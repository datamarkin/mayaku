"""SGD with an orthogonalised update direction for conv weights.

Written from the public description of Muon (K. Jordan, 2024, "Muon: an
optimizer for hidden layers in neural networks"): take the momentum update of a
weight matrix and replace it with its nearest semi-orthogonal matrix, computed
by a few Newton-Schulz iterations, so every singular direction of the update
moves at the same rate instead of the few dominant ones.

Two choices keep the rest of the recipe meaningful:

  magnitude from SGD   the orthogonalised direction is rescaled to the
                       Frobenius norm of the SGD step it replaces, so lr, the
                       cosine, warmup, weight decay and the clip keep their
                       meaning. Only the direction changes.
  a mix, not a swap    step = mix * orth + (1 - mix) * sgd. mix 0 is plain
                       Nesterov SGD, exactly.

Muon is for hidden-layer matrices. Parameters that are not 4-D conv weights,
the first convolution on RGB and the output projections take the plain SGD
step (the caller marks them with `muon: False` in their group).
"""

import torch

# Quintic Newton-Schulz coefficients from the public Muon description: they
# push singular values into roughly [0.7, 1.2] in five steps rather than
# converging exactly, which is all the update direction needs.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def orthogonalise(g, steps=5, eps=1e-7):
    """Approximate U V^T of `g` (2-D) by Newton-Schulz iteration."""
    a, b, c = NS_COEFFS
    x = g.float()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return (x.T if tall else x).to(g.dtype)


class MuonSGD(torch.optim.Optimizer):
    """Nesterov SGD whose conv-weight steps are partly orthogonalised.

    Group keys: lr, momentum, weight_decay (coupled, as torch SGD), muon
    (bool), mix (float in [0, 1]).
    """

    def __init__(self, params, lr, momentum, mix=0.5, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=0.0, muon=False, mix=mix)
        super().__init__(params, defaults)
        self.ns_steps = ns_steps

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, mom, wd = g["lr"], g["momentum"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if wd:
                    d = d.add(p, alpha=wd)
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = d.clone()
                else:
                    state["buf"].mul_(mom).add_(d)
                step = d.add(state["buf"], alpha=mom)   # nesterov, as torch SGD
                if g["muon"] and g["mix"] > 0 and p.ndim == 4:
                    flat = step.reshape(step.shape[0], -1)
                    o = orthogonalise(flat, self.ns_steps)
                    o = o * (flat.norm() / (o.norm() + 1e-12))
                    step = torch.lerp(step, o.reshape_as(step), g["mix"])
                p.add_(step, alpha=-lr)
        return loss
