"""
Model factories and a shared evaluation harness — WM9PH-15 coursework (v2).

Three classifiers are compared on the CICIDS2017 7-class task:

  * Random Forest        - bagged decision trees, strong tabular baseline,
                           naturally handles non-linear feature interactions.
  * XGBoost              - gradient-boosted trees, usually state of the art on
                           tabular intrusion-detection data.
  * MLP (neural network) - a fully-connected feed-forward network
                           (scikit-learn's MLPClassifier), the "deep learning"
                           comparator.

All three expose the standard scikit-learn ``fit`` / ``predict`` /
``predict_proba`` API so the evaluation code below is model-agnostic.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_random_forest(random_state: int = 42, **kw) -> RandomForestClassifier:
    """Random Forest with balanced class weights to counter class imbalance."""
    params = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    params.update(kw)
    return RandomForestClassifier(**params)


def build_xgboost(num_classes: int = 7, random_state: int = 42, **kw):
    """Multi-class XGBoost.  Class imbalance is handled via per-sample weights
    passed at ``fit`` time (see ``balanced_sample_weight``)."""
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed in this environment.")
    params = dict(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.15,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=random_state,
    )
    params.update(kw)
    return XGBClassifier(**params)


def train_tabnet(Xtr, ytr, Xval, yval, max_epochs=40, patience=8,
                 batch_size=4096, random_state=42, **kw):
    """Train a TabNet classifier (Arik & Pfister, 2021) — a deep, attention-based
    network designed for tabular data, included as the 'bigger model' comparator.

    Requires the optional ``pytorch-tabnet`` package; raises a clear error if it
    is missing so the rest of the pipeline still runs without it.
    """
    try:
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as e:  # pragma: no cover
        raise ImportError("TabNet needs `pip install torch pytorch-tabnet`.") from e

    params = dict(n_d=32, n_a=32, n_steps=4, gamma=1.5,
                  n_independent=2, n_shared=2, lambda_sparse=1e-4,
                  optimizer_params=dict(lr=2e-2),
                  scheduler_fn=torch.optim.lr_scheduler.StepLR,
                  scheduler_params=dict(step_size=10, gamma=0.9),
                  mask_type="entmax", seed=random_state, verbose=0,
                  device_name="cpu")
    params.update(kw)
    clf = TabNetClassifier(**params)
    clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_name=["val"],
            eval_metric=["accuracy"], max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, virtual_batch_size=batch_size // 8,
            weights=1)   # weights=1 -> inverse-frequency class balancing
    return clf


class _LSTMWrapper:
    """Thin sklearn-style wrapper around a trained PyTorch LSTM (predict /
    predict_proba on numpy arrays), so it plugs into :func:`evaluate`."""
    def __init__(self, model, n_features):
        self.model = model
        self.n_features = n_features

    def _logits(self, X):
        import torch
        self.model.eval()
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(-1)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 4096):
                out.append(self.model(Xt[i:i+4096]))
        return torch.cat(out)

    def predict(self, X):
        return self._logits(X).argmax(1).numpy()

    def predict_proba(self, X):
        import torch
        return torch.softmax(self._logits(X), dim=1).numpy()


def train_lstm(Xtr, ytr, Xval, yval, n_classes=7, hidden=64, max_epochs=20,
               patience=5, batch_size=1024, random_state=42):
    """Train an LSTM over features-as-pseudo-timesteps.

    NOTE: the flow features have **no temporal order**, so an LSTM is not the
    natural model here.  It is included only as a deliberate *negative control*
    to test whether imposing a sequence structure helps (it does not).  A genuine
    LSTM would require packet-level sequences, which this dataset release lacks.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:  # pragma: no cover
        raise ImportError("LSTM needs `pip install torch`.") from e
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight

    torch.manual_seed(random_state); np.random.seed(random_state)
    torch.set_num_threads(os.cpu_count() or 4)
    nfeat = Xtr.shape[1]

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(64, n_classes))
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1])

    model = _Net()
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1)
    ytr_t = torch.tensor(ytr)
    wrap = _LSTMWrapper(model, nfeat)

    best_f1, best_state, bad = 0.0, None, 0
    for _ in range(max_epochs):
        model.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            crit(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        f1 = f1_score(yval, wrap.predict(Xval), average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return wrap


# --------------------------------------------------------------------------- #
# 1D-CNN  (the PRIMARY model for the CNN-centric project)
# --------------------------------------------------------------------------- #
# A flow is a vector of numeric features with no spatial order, but a 1D-CNN over
# the feature axis is a standard, widely-published NIDS approach: local
# convolutions learn combinations of neighbouring flow statistics. Being a
# differentiable network, the CNN also supports principled gradient-based
# adversarial attacks and adversarial training (see security.py) — the key
# advantage that motivates choosing it as the detector to develop in depth.
def _build_cnn1d_net(n_features, n_classes):
    import torch.nn as nn
    return nn.Sequential(
        nn.Unflatten(1, (1, n_features)),               # (B, feat) -> (B, 1, feat)
        nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
        nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, n_classes),
    )


class CNN1DClassifier:
    """sklearn-style wrapper around a trained 1D-CNN.

    Exposes ``predict`` / ``predict_proba`` on numpy arrays (so it plugs into
    :func:`evaluate`) and keeps the underlying differentiable ``net`` accessible
    for gradient-based adversarial attacks in ``security.py``. Input arrays are
    plain (n_samples, n_features); the channel axis is added inside the net.
    """
    def __init__(self, net, n_features):
        self.net = net
        self.n_features = n_features

    def _logits(self, X):
        import torch
        self.net.eval()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 8192):
                out.append(self.net(Xt[i:i+8192]))
        return torch.cat(out)

    def predict(self, X):
        return self._logits(X).argmax(1).numpy()

    def predict_proba(self, X):
        import torch
        return torch.softmax(self._logits(X), dim=1).numpy()


def train_cnn1d(Xtr, ytr, Xval, yval, n_classes=7, max_epochs=30, patience=6,
                batch_size=2048, lr=1e-3, weight_scheme="sqrt",
                random_state=42, verbose=False):
    """Train the 1D-CNN with class-weighted cross-entropy and early stopping on
    validation macro-F1. Returns a :class:`CNN1DClassifier`."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:  # pragma: no cover
        raise ImportError("CNN needs `pip install torch`.") from e

    torch.manual_seed(random_state); np.random.seed(random_state)
    torch.set_num_threads(os.cpu_count() or 4)
    nfeat = Xtr.shape[1]
    net = _build_cnn1d_net(nfeat, n_classes)

    # per-class weights for the cross-entropy loss
    _, counts = np.unique(ytr, return_counts=True)
    base = len(ytr) / (n_classes * counts)
    cls_w = np.sqrt(base) if weight_scheme == "sqrt" else base
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr)
    wrap = CNN1DClassifier(net, nfeat)
    wrap.history = []          # per-epoch validation macro-F1, for plotting

    best_f1, best_state, bad = 0.0, None, 0
    for ep in range(max_epochs):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            crit(net(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        sched.step()
        f1 = f1_score(yval, wrap.predict(Xval), average="macro")
        wrap.history.append(float(f1))
        if verbose:
            print(f"  cnn epoch {ep:2d}  val macroF1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return wrap


def train_cnn1d_adversarial(Xtr, ytr, Xval, yval, n_classes=7, max_epochs=20,
                            patience=6, batch_size=2048, lr=1e-3, eps=0.1,
                            weight_scheme="sqrt", random_state=42, verbose=False):
    """Adversarially-trained 1D-CNN (FGSM adversarial training).

    Each batch generates an FGSM adversarial example using the CNN's true input
    gradient, and the network is trained on a 50/50 mix of clean and adversarial
    inputs. This is the standard, principled hardening method for neural networks
    and exploits the CNN's differentiability (impossible for the tree models).
    Returns a :class:`CNN1DClassifier`.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:  # pragma: no cover
        raise ImportError("CNN needs `pip install torch`.") from e

    torch.manual_seed(random_state); np.random.seed(random_state)
    torch.set_num_threads(os.cpu_count() or 4)
    nfeat = Xtr.shape[1]
    net = _build_cnn1d_net(nfeat, n_classes)
    _, counts = np.unique(ytr, return_counts=True)
    base = len(ytr) / (n_classes * counts)
    cls_w = np.sqrt(base) if weight_scheme == "sqrt" else base
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr)
    wrap = CNN1DClassifier(net, nfeat)

    best_f1, best_state, bad = 0.0, None, 0
    for ep in range(max_epochs):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            # FGSM adversarial example for this batch (true gradient)
            xb_adv = xb.clone().requires_grad_(True)
            g = torch.autograd.grad(crit(net(xb_adv), yb), xb_adv)[0]
            xb_adv = (xb + eps * g.sign()).clamp(-10, 10).detach()
            # train on clean + adversarial
            opt.zero_grad()
            loss = 0.5 * crit(net(xb), yb) + 0.5 * crit(net(xb_adv), yb)
            loss.backward(); opt.step()
        sched.step()
        f1 = f1_score(yval, wrap.predict(Xval), average="macro")
        if verbose:
            print(f"  adv-cnn epoch {ep:2d}  val macroF1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return wrap


def build_mlp(random_state: int = 42, **kw) -> MLPClassifier:
    """Fully-connected neural network (128-64 hidden units, ReLU, Adam).

    Supervised multi-class classifier trained with the cross-entropy
    (log-loss) objective; early stopping on an internal validation slice
    guards against over-fitting.
    """
    params = dict(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,                # L2 regularisation
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=random_state,
    )
    params.update(kw)
    return MLPClassifier(**params)


def train_xgboost(Xtr, ytr, Xval=None, yval=None, num_classes=7,
                  early_stopping_rounds=30, sample_weight=None,
                  random_state=42, **params):
    """Build and fit an XGBoost model, with early stopping on a validation set.

    If ``Xval``/``yval`` are given, training stops when validation mlogloss has
    not improved for ``early_stopping_rounds`` rounds, and prediction uses the
    best iteration — so ``n_estimators`` becomes an upper bound rather than a
    value to tune by hand.  Class imbalance is handled via ``sample_weight``.
    """
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed in this environment.")
    defaults = dict(n_estimators=1000)
    defaults.update(params)
    if Xval is not None:
        defaults["early_stopping_rounds"] = early_stopping_rounds
    model = build_xgboost(num_classes=num_classes, random_state=random_state,
                          **defaults)
    if sample_weight is None:
        sample_weight = balanced_sample_weight(ytr)
    fit_kw = {"sample_weight": sample_weight, "verbose": False}
    if Xval is not None:
        fit_kw["eval_set"] = [(Xval, yval)]
    model.fit(Xtr, ytr, **fit_kw)
    return model


def save_cnn(cnn, path, n_classes=7):
    """Persist a CNN1DClassifier: torch weights + the shape needed to rebuild."""
    import torch
    torch.save({"state_dict": cnn.net.state_dict(),
                "n_features": int(cnn.n_features),
                "n_classes": int(n_classes)}, path)


def load_cnn(path):
    """Reload a CNN1DClassifier saved by :func:`save_cnn`."""
    import torch
    ckpt = torch.load(path, weights_only=True)
    net = _build_cnn1d_net(ckpt["n_features"], ckpt["n_classes"])
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return CNN1DClassifier(net, ckpt["n_features"])


# --------------------------------------------------------------------------- #
# Imbalance helper
# --------------------------------------------------------------------------- #
def balanced_sample_weight(y_int: np.ndarray, scheme: str = "balanced") -> np.ndarray:
    """Per-sample weights to counter class imbalance.

    scheme:
      * "balanced" - weight = N / (K * n_c), the standard inverse-frequency
        weighting. Strong; ideal when classes are only mildly imbalanced.
      * "sqrt"     - weight = sqrt(N / (K * n_c)), a milder weighting. Preferred
        under *extreme* imbalance (e.g. the full dataset's ~1000:1) where full
        inverse-frequency over-boosts the rare classes and floods them with
        false positives, collapsing their precision.

    Used to give XGBoost the same balancing that RF/MLP get internally.
    """
    classes, counts = np.unique(y_int, return_counts=True)
    freq = {c: n for c, n in zip(classes, counts)}
    n_total = len(y_int)
    n_classes = len(classes)
    base = {c: n_total / (n_classes * freq[c]) for c in classes}
    if scheme == "sqrt":
        w = {c: float(np.sqrt(v)) for c, v in base.items()}
    elif scheme == "balanced":
        w = base
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return np.array([w[c] for c in y_int], dtype="float32")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, X, y_true, class_names) -> dict:
    """Return a dictionary of headline metrics + per-class table + arrays.

    Keys: accuracy, macro_f1, weighted_f1, report (str), report_dict,
          confusion (counts), confusion_norm (row-normalised),
          per_class (dict label -> {precision, recall, f1, support}),
          y_pred.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]), "support": int(s[i]),
        }
        for i in labels
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            digits=4, zero_division=0,
        ),
        "report_dict": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
        "confusion": cm,
        "confusion_norm": cm_norm,
        "per_class": per_class,
        "y_pred": y_pred,
    }


def plot_confusion(cm_norm, class_names, title, ax=None, save_path=None):
    """Draw a row-normalised confusion-matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6))
    else:
        fig = ax.figure
    sns.heatmap(
        cm_norm, annot=True, fmt=".1%", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, cbar_kws={"label": "Row-normalised rate"}, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
