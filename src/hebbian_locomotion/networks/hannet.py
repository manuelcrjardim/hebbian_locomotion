"""
Hebbian Attractor Network (HAN).

A drop-in extension of the existing ``HebbianNet`` (ABCD plastic feedforward net)
that adds the two mechanisms from Dittrich et al., "Hebbian Attractor Networks
for Robot Locomotion":

  (1) Temporal averaging (M):  the Hebbian update is driven by a moving average
      of the pre- and post-synaptic activations over the last M forward passes
      (HAN Eq. 5), instead of the instantaneous activations.

  (2) Dual-timescale (tau_hebb):  the forward pass / action inference still runs
      every step at f_NN, but Hebbian weight updates are applied only every
      tau_hebb = floor(f_NN / f_hebb) steps (HAN Eq. 8).

Both M and tau_hebb are HYPERPARAMETERS, not evolved. The evolved parameters
(A, B, C, D, lr) are exactly those of ``HebbianNet``, so:
  * the ES search dimensionality is unchanged (drop-in for run_es_dynamic.py),
  * get_models_params / set_models_params / get_a_model_params are inherited
    verbatim and remain compatible with your solver.

Setting M=1 and tau_hebb=1 recovers the instantaneous ABCD baseline exactly,
so this single class also covers the (A)/(B) conditions of the ablation ladder.

Ready-made configs (HAN paper Table I; superscript = M, subscript = f_NN/f_hebb):

    condition          norm_mode   M    tau_hebb
    (A) HNN (no MN)     'none'      1    1
    (B) HAN M=1 f=1     'max'       1    1
    (C) HAN M=1 f=4     'max'       1    4
    (D) HAN M=10 f=1    'max'      10    1
    (E) HAN M=10 f=4    'max'      10    4     <- strongest fixed-point config

Usage (in run_es_dynamic.py, replacing the HebbianNet instantiation):

    from hebbian_locomotion.networks.han_net import HANNet
    models = HANNet(POPSIZE, sizes, norm_mode='max', M=10, tau_hebb=4)
    # ... rest of the training loop is unchanged ...

The freeze() / unfreeze() methods support the freeze-control experiment: call
models.freeze() at the perturbation step to disable plasticity while the forward
pass keeps running, isolating online adaptation from static robustness.
"""

import torch

try:
    # absolute import matching the package layout in scripts/
    from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet
except ImportError:  # pragma: no cover - fallback for relative import contexts
    from .hebbian_neural_net import HebbianNet


def _identity_norm(w, eps=1e-5):
    """No-op normalization, for the (A) HNN-without-MN condition."""
    return w


class HANNet(HebbianNet):
    """Hebbian Attractor Network: HebbianNet + moving average + dual-timescale.

    Args:
        popsize:    Number of parallel individuals (matches num_envs).
        sizes:      Layer sizes, e.g. [obs_dim, 64, 32, action_dim].
        init_noise: Uniform range for initial connection weights.
        norm_mode:  'max' (recommended for HANs), 'var', 'clip', or 'none'.
        M:          Moving-average window length over activations (>= 1).
        tau_hebb:   Hebbian update period in forward passes (>= 1).
                    tau_hebb = floor(f_NN / f_hebb); use tau_from_freq() to derive.
    """

    def __init__(self,
                 popsize,
                 sizes,
                 init_noise=0.01,
                 norm_mode='max',
                 M=10,
                 tau_hebb=1):
        super().__init__(popsize, sizes, init_noise, norm_mode)

        # 'none' is not handled by the parent; wire it up here so condition (A)
        # (HNN without max-normalization) is available from the same class.
        if norm_mode == 'none':
            self.WeightStand = _identity_norm

        self.M = max(1, int(M))
        self.tau_hebb = max(1, int(tau_hebb))

        # One activation "node" per layer boundary: x^0 (obs) ... x^L (action).
        self.n_nodes = len(sizes)

        # Episode-lifetime state (reset every rollout via reset_weights()).
        self.plastic = True          # toggled by freeze()/unfreeze()
        self.t = 0                   # forward-pass counter within the episode
        self._init_buffers()

    # ------------------------------------------------------------------
    # Activation ring buffers (one per node) for the moving average
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """(Re)allocate per-node ring buffers and reset their pointers/counts."""
        device = self.weights[0].device
        self.act_buf = [
            torch.zeros(self.M, self.popsize, self.architecture[n], device=device)
            for n in range(self.n_nodes)
        ]
        self.buf_ptr = [0] * self.n_nodes   # next write index per node
        self.buf_cnt = [0] * self.n_nodes   # number of valid samples per node

    def _push(self, n, x):
        """Write the latest activation x (popsize, dim) into node n's buffer."""
        self.act_buf[n][self.buf_ptr[n]] = x
        self.buf_ptr[n] = (self.buf_ptr[n] + 1) % self.M
        self.buf_cnt[n] = min(self.buf_cnt[n] + 1, self.M)

    def _mean(self, n):
        """Moving average over the up-to-M valid samples in node n's buffer."""
        c = self.buf_cnt[n]
        return self.act_buf[n][:c].mean(dim=0)

    # ------------------------------------------------------------------
    # Plasticity toggle (for the freeze-control experiment)
    # ------------------------------------------------------------------

    def freeze(self):
        """Stop Hebbian updates; the forward pass keeps running on frozen weights."""
        self.plastic = False

    def unfreeze(self):
        """Resume Hebbian updates."""
        self.plastic = True

    # ------------------------------------------------------------------
    # Rollout reset
    # ------------------------------------------------------------------

    def reset_weights(self):
        """Re-init plastic weights AND the activation buffers / step counter.

        Called once before each rollout. Extends the parent so that the moving
        average and dual-timescale clock start fresh every generation.
        """
        super().reset_weights()
        self._init_buffers()
        self.t = 0
        self.plastic = True

    # ------------------------------------------------------------------
    # Forward pass + (dual-timescale, averaged) Hebbian update
    # ------------------------------------------------------------------

    def forward(self, obs):
        """Run one forward pass and, on update steps, one HAN Hebbian update.

        Forward propagation uses INSTANTANEOUS activations every step (HAN Eq. 2);
        the Hebbian update uses the MOVING-AVERAGE activations (Eq. 5) and only
        fires every tau_hebb steps (Eq. 8).

        Args:
            obs: Observation tensor of shape (popsize, obs_dim).

        Returns:
            Action tensor of shape (popsize, action_dim), values in (-1, 1).
        """
        with torch.no_grad():
            # --- Instantaneous forward propagation, recording node activations ---
            acts = [obs]
            x = obs
            for i in range(self.n_layers):
                x = torch.tanh(torch.einsum('ij, ijk -> ik', x, self.weights[i].float()))
                acts.append(x)
            action = acts[-1]

            # --- Update each node's moving-average buffer with this step ---
            for n in range(self.n_nodes):
                self._push(n, acts[n])

            # --- Dual-timescale Hebbian update on averaged activations ---
            if self.plastic and (self.t % self.tau_hebb == 0):
                avg = [self._mean(n) for n in range(self.n_nodes)]
                for i in range(self.n_layers):
                    self.weights[i] = self.hebbian_update(
                        i, self.weights[i], avg[i], avg[i + 1],
                        self.A[i], self.B[i], self.C[i], self.D[i], self.lr[i]
                    )

            self.t += 1

        return action.float().detach()

    # ------------------------------------------------------------------
    # Helpers for attractor analysis (PCA / summed-l2 weight change)
    # ------------------------------------------------------------------

    def get_weight_snapshot(self, idx=0):
        """Flat copy of all plastic weights for individual `idx` (numpy, on CPU).

        Collect these per step during an evaluation rollout to build the weight
        trajectory for PCA, summed-l2 change, and Fourier analysis.
        """
        return torch.cat([w[idx].flatten() for w in self.weights]).detach().cpu().numpy()

    @staticmethod
    def tau_from_freq(f_nn, f_hebb):
        """tau_hebb = floor(f_NN / f_hebb), clamped to >= 1."""
        return max(1, int(f_nn // f_hebb))