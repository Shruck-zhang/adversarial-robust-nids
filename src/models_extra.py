# -*- coding: utf-8 -*-
"""Extra detection models for the comprehensive comparison (O1).

Adds a deep DNN, LSTM, GRU and CNN-LSTM hybrid (all reusing the CNN1DClassifier
wrapper via a shared class-weighted training loop), plus classical Logistic
Regression and Decision Tree references. This lets the thesis compare many model
families on the same data rather than assuming a single main model.
"""
import os
import numpy as np

from models import CNN1DClassifier


# ---- classical references ------------------------------------------------- #
def build_logistic(random_state=42, **kw):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=random_state, **kw)


def build_decision_tree(random_state=42, **kw):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(class_weight="balanced", random_state=random_state, **kw)


# ---- deep-network builders (all take (B, n_features) -> logits) ------------ #
def build_dnn_net(n_features, n_classes):
    """Deep feed-forward network (deeper than the shallow MLP baseline)."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, n_classes),
    )


def build_lstm_net(n_features, n_classes, hidden=64):
    import torch.nn as nn
    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(64, n_classes))
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(-1)                 # (B, feat) -> (B, feat, 1)
            _, (h, _) = self.lstm(x)
            return self.head(h[-1])
    return _LSTM()


def build_gru_net(n_features, n_classes, hidden=64):
    import torch.nn as nn
    class _GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(1, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(64, n_classes))
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])
    return _GRU()


def build_cnn_lstm_net(n_features, n_classes, hidden=64):
    """CNN-LSTM hybrid: 1-D convolutions extract local features, an LSTM models the
    resulting sequence (the HAST-NAD-style spatial-temporal design)."""
    import torch.nn as nn
    class _CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2))
            self.lstm = nn.LSTM(64, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(64, n_classes))
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)                  # (B, feat) -> (B, 1, feat)
            c = self.conv(x).transpose(1, 2)        # (B, seq, 64)
            _, (h, _) = self.lstm(c)
            return self.head(h[-1])
    return _CNNLSTM()


# ---- shared class-weighted training loop ---------------------------------- #
def train_net(net, Xtr, ytr, Xval, yval, n_classes, weight_scheme="sqrt",
              max_epochs=25, patience=6, batch_size=2048, lr=1e-3,
              random_state=42, verbose=False, tag="net"):
    """Train any (B, feat)->logits torch net with class-weighted cross-entropy and
    early stopping on validation macro-F1. Returns a CNN1DClassifier."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    torch.manual_seed(random_state); np.random.seed(random_state)
    torch.set_num_threads(os.cpu_count() or 4)
    nfeat = Xtr.shape[1]

    _, counts = np.unique(ytr, return_counts=True)
    base = len(ytr) / (n_classes * counts)
    cls_w = np.sqrt(base) if weight_scheme == "sqrt" else base
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)

    Xtr_t = torch.tensor(np.asarray(Xtr), dtype=torch.float32)
    ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.long)
    wrap = CNN1DClassifier(net, nfeat); wrap.history = []
    best, best_state, bad = 0.0, None, 0
    for ep in range(max_epochs):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            crit(net(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        sched.step()
        f1 = f1_score(yval, wrap.predict(Xval), average="macro")
        wrap.history.append(float(f1))
        if verbose:
            print(f"  {tag} ep{ep:2d} valF1={f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return wrap


def save_dnn(clf, path, n_classes):
    """Persist a DNN CNN1DClassifier (torch weights + shape to rebuild)."""
    import torch
    torch.save({"state_dict": clf.net.state_dict(),
                "n_features": int(clf.n_features), "n_classes": int(n_classes)}, path)


def load_dnn(path):
    """Reload a DNN saved by save_dnn."""
    import torch
    ck = torch.load(path, weights_only=True)
    net = build_dnn_net(ck["n_features"], ck["n_classes"])
    net.load_state_dict(ck["state_dict"]); net.eval()
    return CNN1DClassifier(net, ck["n_features"])


def build_mlp_net(n_features, n_classes):
    """A shallow feed-forward net (torch), so the MLP is also gradient-attackable."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, n_classes),
    )


def train_net_adversarial(net, Xtr, ytr, Xval, yval, n_classes, eps=0.1,
                          weight_scheme="sqrt", max_epochs=20, patience=6,
                          batch_size=2048, lr=1e-3, random_state=42,
                          verbose=False, tag="adv"):
    """Generic FGSM adversarial training for any (B, feat)->logits net: each batch
    is augmented with an FGSM adversarial example from the net's true input
    gradient and the loss is a 50/50 clean+adversarial mix. Returns CNN1DClassifier."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    torch.manual_seed(random_state); np.random.seed(random_state)
    torch.set_num_threads(os.cpu_count() or 4)
    nfeat = Xtr.shape[1]
    _, counts = np.unique(ytr, return_counts=True)
    base = len(ytr) / (n_classes * counts)
    cls_w = np.sqrt(base) if weight_scheme == "sqrt" else base
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)
    Xtr_t = torch.tensor(np.asarray(Xtr), dtype=torch.float32)
    ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.long)
    wrap = CNN1DClassifier(net, nfeat)
    best, best_state, bad = 0.0, None, 0
    for ep in range(max_epochs):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), batch_size):
            idx = perm[i:i + batch_size]; xb, yb = Xtr_t[idx], ytr_t[idx]
            xb_adv = xb.clone().requires_grad_(True)
            g = torch.autograd.grad(crit(net(xb_adv), yb), xb_adv)[0]
            xb_adv = (xb + eps * g.sign()).clamp(-10, 10).detach()
            opt.zero_grad()
            loss = 0.5 * crit(net(xb), yb) + 0.5 * crit(net(xb_adv), yb)
            loss.backward(); opt.step()
        sched.step()
        f1 = f1_score(yval, wrap.predict(Xval), average="macro")
        if verbose:
            print(f"  {tag} ep{ep:2d} valF1={f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return wrap
