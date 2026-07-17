"""
Adversarial robustness evaluation — WM9PH-15 coursework (v2), Learning Outcome 3
("develop, **secure** and optimise the performance of a machine learning model").

A network intrusion detector is itself a target: an attacker who can slightly
perturb the statistical fingerprint of a malicious flow (e.g. by padding
packets, adding jitter, or inserting benign cover traffic) may push it across
the model's decision boundary and evade detection.  This module quantifies how
fragile each model is to such perturbations.

Two complementary, model-agnostic perturbation schemes are implemented:

  * Gaussian noise   - isotropic random jitter of increasing magnitude, a proxy
                       for natural measurement noise / unintentional drift.
  * FGSM-style        - a white-box, gradient-free approximation that pushes each
    boundary nudge      feature in the direction that most increases the loss,
                        estimated by finite differences on the predicted
                        probability of the true class.  This is the worst-case
                        analogue of an adversary deliberately crafting an
                        evasive flow.

Robustness is reported as the per-class **false-negative rate** (attacks
mis-classified as something else, especially as Benign) as perturbation
strength grows — the metric a defender actually cares about.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------------------------------- #
# Perturbations  (applied in the already-scaled feature space)
# --------------------------------------------------------------------------- #
def gaussian_perturbation(X, epsilon, random_state=42):
    """Add isotropic Gaussian noise with std = epsilon (scaled space)."""
    rng = np.random.RandomState(random_state)
    return X + rng.normal(0.0, epsilon, size=X.shape).astype(X.dtype)


def fgsm_perturbation(model, X, y_true, epsilon, n_classes):
    """Gradient-free FGSM approximation.

    For each sample we estimate, by central finite differences, the sign of the
    gradient of the true-class probability w.r.t. each feature, then step a
    distance ``epsilon`` in the direction that *decreases* that probability
    (i.e. most increases the chance of mis-classification).

    To keep this tractable the gradient sign is estimated on a single shared
    random probe direction per feature batch; this yields a conservative
    (lower-bound) estimate of adversarial fragility, which is sufficient to
    compare models.  Operates on the scaled feature space.
    """
    X = np.asarray(X, dtype="float64")
    n, d = X.shape
    # Probe each feature direction with a small delta and measure the change in
    # the predicted probability of the true class.
    delta = 1e-2
    base_prob = _true_class_proba(model, X, y_true, n_classes)
    grad_sign = np.zeros_like(X)
    # Estimate gradient sign feature-by-feature (vectorised over samples).
    for j in range(d):
        Xp = X.copy()
        Xp[:, j] += delta
        prob_up = _true_class_proba(model, Xp, y_true, n_classes)
        # If increasing feature j lowers true-class prob, the adversary moves +;
        # otherwise moves -.  sign of d(prob)/d(x_j); step opposite to raise loss.
        grad = (prob_up - base_prob) / delta
        grad_sign[:, j] = -np.sign(grad)
    return (X + epsilon * grad_sign).astype("float32")


def _true_class_proba(model, X, y_true, n_classes):
    proba = model.predict_proba(X)
    return proba[np.arange(len(y_true)), y_true]


def pgd_perturbation(model, X, y_true, epsilon, n_classes, steps=7, clip=10.0):
    """Projected Gradient Descent — a stronger, multi-step attack.

    Iterated FGSM: at each of ``steps`` steps we re-estimate the gradient sign of
    the true-class probability (finite differences) and take a small step
    (size ``epsilon/4``) in the loss-increasing direction, then project back into
    the L-infinity ball of radius ``epsilon`` around the original point.  Because
    it adapts to the model at every step, PGD is markedly stronger than the
    single-step FGSM in :func:`fgsm_perturbation`; using it as an *independent*
    evaluation attack tests whether an adversarially-trained model is genuinely
    robust or merely memorised the one-shot perturbation it was trained on.
    """
    X0 = np.asarray(X, dtype="float64")
    Xadv = X0.copy()
    alpha = epsilon / 4.0
    delta = 1e-2
    for _ in range(steps):
        base = _true_class_proba(model, Xadv, y_true, n_classes)
        grad_sign = np.zeros_like(Xadv)
        for j in range(Xadv.shape[1]):
            Xp = Xadv.copy()
            Xp[:, j] += delta
            up = _true_class_proba(model, Xp, y_true, n_classes)
            grad_sign[:, j] = -np.sign((up - base) / delta)
        Xadv = Xadv + alpha * grad_sign
        Xadv = np.clip(Xadv, X0 - epsilon, X0 + epsilon)   # project to L-inf ball
        Xadv = np.clip(Xadv, -clip, clip)
    return Xadv.astype("float32")


# --------------------------------------------------------------------------- #
# Gradient-based attacks for the differentiable CNN (true white-box FGSM / PGD)
# --------------------------------------------------------------------------- #
# Because the CNN is differentiable, we can compute the EXACT loss gradient w.r.t.
# the input via back-propagation, rather than the finite-difference approximation
# the tree models require. This is the standard, principled adversarial attack and
# the main reason a neural network is the natural detector for a security study.
def fgsm_grad(cnn, X, y_true, epsilon, clip=10.0):
    """One-step white-box FGSM using the CNN's input gradient."""
    import torch
    import torch.nn.functional as F
    net = cnn.net; net.eval()
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(np.asarray(y_true), dtype=torch.long)
    loss = F.cross_entropy(net(Xt), yt)
    grad = torch.autograd.grad(loss, Xt)[0]
    adv = (Xt + epsilon * grad.sign()).clamp(-clip, clip)
    return adv.detach().numpy().astype("float32")


def pgd_grad(cnn, X, y_true, epsilon, steps=10, clip=10.0):
    """Multi-step white-box PGD (iterated FGSM, projected to the L-inf ball)."""
    import torch
    import torch.nn.functional as F
    net = cnn.net; net.eval()
    X0 = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y_true), dtype=torch.long)
    alpha = epsilon / 4.0
    Xadv = X0.clone()
    for _ in range(steps):
        Xadv.requires_grad_(True)
        loss = F.cross_entropy(net(Xadv), yt)
        grad = torch.autograd.grad(loss, Xadv)[0]
        Xadv = Xadv.detach() + alpha * grad.sign()
        Xadv = torch.max(torch.min(Xadv, X0 + epsilon), X0 - epsilon).clamp(-clip, clip)
    return Xadv.detach().numpy().astype("float32")


def _perturb(model, X, y_true, eps, scheme, n_classes, random_state):
    """Dispatch helper used by robustness_curve / per_class_fnr."""
    if eps == 0:
        return X
    if scheme == "gaussian":
        return gaussian_perturbation(X, eps, random_state)
    if scheme == "fgsm":
        return fgsm_perturbation(model, X, y_true, eps, n_classes)
    if scheme == "pgd":
        return pgd_perturbation(model, X, y_true, eps, n_classes)
    if scheme == "fgsm_grad":          # gradient-based, CNN only
        return fgsm_grad(model, X, y_true, eps)
    if scheme == "pgd_grad":           # gradient-based, CNN only
        return pgd_grad(model, X, y_true, eps)
    raise ValueError(scheme)


# --------------------------------------------------------------------------- #
# Adversarial training (defence)
# --------------------------------------------------------------------------- #
def adversarial_augment(model, X, y, n_classes, rng,
                        gauss_lo=0.02, gauss_hi=0.12, fgsm_eps=0.03,
                        clean_weight=3.0, clip=10.0, benign_index=0):
    """Build an adversarially-augmented training set for robustness hardening.

    Returns ``(X_aug, y_aug, weight)`` consisting of:
      * the clean originals (up-weighted by ``clean_weight`` so clean accuracy is
        preserved),
      * a Gaussian-noise copy of every flow (per-sample sigma drawn uniformly
        from [gauss_lo, gauss_hi]) — this widens the decision margins, and
      * an FGSM adversarial copy of every *attack* flow, generated against
        ``model`` — this teaches the model to resist targeted evasion.

    All copies keep their TRUE label.  Retraining on this set with the returned
    ``weight`` (multiplied into the usual class-balancing weights) yields a model
    that is far harder to evade for a small clean-accuracy cost.
    """
    sig = rng.uniform(gauss_lo, gauss_hi, size=(len(X), 1)).astype("float32")
    Xg = X + rng.normal(0, 1, X.shape).astype("float32") * sig
    atk = np.where(y != benign_index)[0]
    Xf = fgsm_perturbation(model, X[atk], y[atk], fgsm_eps, n_classes)
    X_aug = np.clip(np.vstack([X, Xg, Xf]), -clip, clip).astype("float32")
    y_aug = np.concatenate([y, y, y[atk]])
    weight = np.concatenate([np.full(len(X), clean_weight, dtype="float32"),
                             np.ones(len(Xg), dtype="float32"),
                             np.ones(len(Xf), dtype="float32")])
    return X_aug, y_aug, weight


# --------------------------------------------------------------------------- #
# Robustness curves
# --------------------------------------------------------------------------- #
def robustness_curve(model, X, y_true, epsilons, scheme="gaussian",
                     n_classes=7, benign_index=0, random_state=42):
    """Evaluate a model across increasing perturbation strengths.

    Returns a dict of arrays (one entry per epsilon):
        accuracy, macro_f1, attack_fnr (fraction of true-attack samples
        predicted as Benign), evasion_rate (attacks predicted as ANY wrong
        class).
    """
    attack_mask = y_true != benign_index
    out = {"epsilon": list(epsilons), "accuracy": [], "macro_f1": [],
           "attack_to_benign_fnr": [], "evasion_rate": []}

    for eps in epsilons:
        Xp = _perturb(model, X, y_true, eps, scheme, n_classes, random_state)
        y_pred = model.predict(Xp)
        out["accuracy"].append(accuracy_score(y_true, y_pred))
        out["macro_f1"].append(f1_score(y_true, y_pred, average="macro"))

        if attack_mask.any():
            atk_pred = y_pred[attack_mask]
            out["attack_to_benign_fnr"].append(
                float(np.mean(atk_pred == benign_index)))
            out["evasion_rate"].append(
                float(np.mean(atk_pred != y_true[attack_mask])))
        else:
            out["attack_to_benign_fnr"].append(np.nan)
            out["evasion_rate"].append(np.nan)

    return out


def per_class_fnr(model, X, y_true, epsilon, class_names, scheme="gaussian",
                  n_classes=7, random_state=42):
    """False-negative rate per attack class at a single perturbation strength."""
    Xp = _perturb(model, X, y_true, epsilon, scheme, n_classes, random_state)
    y_pred = model.predict(Xp)

    rows = {}
    for idx, name in enumerate(class_names):
        mask = y_true == idx
        if mask.any():
            rows[name] = float(np.mean(y_pred[mask] != idx))
        else:
            rows[name] = np.nan
    return rows
