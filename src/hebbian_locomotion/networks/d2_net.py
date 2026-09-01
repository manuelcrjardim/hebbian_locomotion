"""
D2 network: Hebbian Attractor Network with a META-LEARNED averaging window.

HANNet averages pre-/post-synaptic activations over a FIXED integer boxcar
window of M forward passes (HAN Eq. 5). M is a hyperparameter, hand-picked
before training, and the results of Objective 1 show that the choice of M
dominates the plasticity regime: small M gives a co-dynamic limit cycle whose
gait collapses the moment plasticity is frozen, large M gives a converged,
near-static controller that cannot respond to a changed body.

D2Net keeps the boxcar but makes its LENGTH an evolved parameter, so the
temporal window becomes an outcome of the experiment rather than an input to
it.

THE INTEGER PROBLEM AND THE FRACTIONAL BOXCAR
---------------------------------------------
A boxcar averages over a whole number of steps, so a continuously varying
parameter produces a staircase: fitness is flat while the parameter moves
within one integer, then jumps. Evolution strategies estimate a gradient from
perturbations, and on a flat surface that estimate is zero, so the search
stalls between the steps.

The fix is to define the window at non-integer lengths by blending the two
adjacent integer boxcars:

    M_lo = floor(M),  lam = M - M_lo

    xbar_t = (1 - lam) * boxcar_{M_lo}(t)  +  lam * boxcar_{M_lo + 1}(t)

At lam = 0 this is exactly the M_lo boxcar; at lam -> 1 it approaches the
M_lo + 1 boxcar, which is where the next integer's lam = 0 also lands, so the
output is continuous across every boundary. This is the length analogue of the
standard fractional-delay filter construction (Laakso et al., 1996), which
interpolates between adjacent integer delays for the same reason.

Crucially the estimator remains a weighted sum of boxcars, so it retains the
comb structure that motivates the whole analysis: the spectral nulls of the
boxcar slide continuously with M rather than jumping between discrete
positions. An exponential moving average -- the obvious alternative continuous
relaxation -- has no nulls at all, and would not be comparable to Objective 1
on the same axis.

PARAMETERISATION (log-M):

    M = exp(theta)              theta is the evolved parameter, M in STEPS

ES perturbs theta with a single global sigma. Under log-M a perturbation of
size sigma is a MULTIPLICATIVE change of exp(sigma) in the window length,
uniformly across the whole range -- the search is scale-free, which is what is
wanted when the plausible range spans M = 1 to M = 160. Evolving M directly on
a linear scale would search coarsely at the short end and barely move at the
long end.

The evolved M is reported in the same units as the fixed M values of
Objective 1, with no conversion.

PARAMETER LAYOUT:
theta is appended as a contiguous TAIL of the flat parameter vector, so the
parent HebbianNet block is a pure prefix and every existing checkpoint,
solver, and analysis script that indexes the parent block keeps working
unchanged. Per individual the parameter count is n_parent + n_theta.

SHARING (Objective 3 is a one-line config change):
    share='global'  1 theta for the whole network      (Objective 2)
    share='layer'   1 theta per activation node        (Objective 3)
    share='node'    1 theta per unit per node          (Objective 3)

Usage:
    from hebbian_locomotion.networks.d2_net import D2Net
    models = D2Net(POPSIZE, sizes, norm_mode='max', share='global', M_init=20.0)
    # everything downstream (solver, rollout loop, freeze()) is unchanged
"""

import numpy as np
import torch

try:
    from hebbian_locomotion.networks.han_net import HANNet
except ImportError:  # pragma: no cover
    from .han_net import HANNet


# Numerical guard rails on the evolved window, in control steps. M_MIN = 1 is
# the instantaneous limit (the average is the current activation alone).
# M_MAX bounds the ring buffer allocated at construction.
M_MIN = 1.0
M_MAX = 512.0


class D2Net(HANNet):
    """HAN whose boxcar averaging window is evolved rather than hand-set.

    Args:
        popsize:    Number of parallel individuals (matches num_envs).
        sizes:      Layer sizes, e.g. [obs_dim, 64, 32, action_dim].
        init_noise: Uniform range for initial connection weights.
        norm_mode:  'max' throughout this thesis (cross-condition comparability).
        tau_hebb:   Hebbian update period in forward passes. Held at 1 here so
                    that the averaging window is the single manipulated
                    variable.
        share:      'global' | 'layer' | 'node'.
        M_init:     Initial window length in STEPS. Default 20.0 is the
                    paper-faithful HAN condition, so the evolved window is
                    initialised at the Objective 1 optimum and any drift away
                    from it is an informative result.
        M_cap:      Ring-buffer capacity. Windows are clamped to this, so it
                    must exceed any M the search should be able to reach.
        theta_jitter: Std of Gaussian noise added to the initial theta across
                    the population. Small but nonzero so the initial ES
                    gradient on theta is estimable rather than degenerate.
    """

    def __init__(self,
                 popsize,
                 sizes,
                 init_noise=0.01,
                 norm_mode='max',
                 tau_hebb=1,
                 share='global',
                 M_init=20.0,
                 M_cap=None,
                 theta_jitter=0.05):

        self.M_cap = int(M_cap if M_cap is not None else M_MAX)

        # The parent allocates ring buffers of length M; allocate them at the
        # cap once here so nothing reallocates as the evolved window moves.
        super().__init__(popsize, sizes, init_noise=init_noise,
                         norm_mode=norm_mode, M=self.M_cap, tau_hebb=tau_hebb)

        if share not in ('global', 'layer', 'node'):
            raise ValueError(f"share must be 'global'|'layer'|'node', got {share!r}")
        self.share = share

        # --- Map each activation node to its slice of the theta vector ---
        # 'global': every node reads the same scalar.
        # 'layer' : one scalar per node.
        # 'node'  : one scalar per unit, so the slice has the node's width.
        if share == 'global':
            self.theta_slices = [slice(0, 1)] * self.n_nodes
            self.n_theta = 1
        elif share == 'layer':
            self.theta_slices = [slice(n, n + 1) for n in range(self.n_nodes)]
            self.n_theta = self.n_nodes
        else:
            offs, sl = 0, []
            for n in range(self.n_nodes):
                w = self.architecture[n]
                sl.append(slice(offs, offs + w))
                offs += w
            self.theta_slices = sl
            self.n_theta = offs

        device = self.weights[0].device
        theta0 = float(np.log(np.clip(M_init, M_MIN, self.M_cap)))
        self.theta = (theta0 + theta_jitter *
                      torch.randn(popsize, self.n_theta, device=device))

        # Size of the inherited HebbianNet block, computed explicitly rather
        # than by calling the parent getter: that getter is defined as
        # len(self.get_a_model_params()), which dispatches back to the
        # overridden D2Net version and would already include the theta tail.
        # Five coefficient groups (A, B, C, D, lr), one matrix per layer.
        per_layer = sum(sizes[i] * sizes[i + 1] for i in range(self.n_layers))
        self.n_parent_params = 5 * per_layer

        self._init_buffers()

    # ------------------------------------------------------------------
    # The evolved window
    # ------------------------------------------------------------------

    @property
    def M_window(self):
        """Evolved window length in steps, shape (popsize, n_theta), float."""
        return torch.exp(self.theta).clamp(M_MIN, float(self.M_cap))

    def get_M(self):
        """Evolved window lengths in control steps, numpy (popsize, n_theta)."""
        return self.M_window.detach().cpu().numpy()

    def M_summary(self, dt=0.02):
        """Population summary of the evolved window, for per-generation logging.

        Returns a dict of scalars: M in steps and in seconds. The trajectory of
        M over generations is the headline result of Objective 2.
        """
        m = self.get_M()
        return {
            'M_mean': float(m.mean()),
            'M_median': float(np.median(m)),
            'M_std': float(m.std()),
            'M_min': float(m.min()),
            'M_max': float(m.max()),
            'M_seconds_mean': float(m.mean() * dt),
        }

    # ------------------------------------------------------------------
    # Fractional-length boxcar, replacing HANNet's fixed-M ring buffer mean
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """Allocate ring buffers at the cap, plus per-node write pointers."""
        device = self.weights[0].device
        self.act_buf = [
            torch.zeros(self.M_cap, self.popsize, self.architecture[n],
                        device=device)
            for n in range(self.n_nodes)
        ]
        self.buf_ptr = [0] * self.n_nodes   # next write index per node
        self.buf_cnt = [0] * self.n_nodes   # valid samples per node

    def _push(self, n, x):
        """Write the latest activation into node n's ring buffer."""
        self.act_buf[n][self.buf_ptr[n]] = x
        self.buf_ptr[n] = (self.buf_ptr[n] + 1) % self.M_cap
        self.buf_cnt[n] = min(self.buf_cnt[n] + 1, self.M_cap)

    def _tail(self, n, k):
        """The k most recently written rows of node n's buffer, oldest first.

        The buffer is circular, so the newest sample sits at buf_ptr - 1 and
        the k-sample tail may wrap past index 0.
        """
        end = self.buf_ptr[n]
        if k <= end:
            return self.act_buf[n][end - k:end]
        return torch.cat((self.act_buf[n][self.M_cap - (k - end):],
                          self.act_buf[n][:end]), dim=0)

    def _mean(self, n):
        """Fractional-length boxcar mean of node n's activations.

        Blends the boxcars of length M_lo and M_lo + 1 in proportion to the
        fractional part of the evolved window, so the estimator is continuous
        in M. Early in an episode the window is truncated to the number of
        samples actually available, exactly as HANNet's buf_cnt does, so the
        estimate is unbiased from step 0 rather than ramping up from zeros.
        """
        avail = self.buf_cnt[n]
        if avail == 0:
            return torch.zeros(self.popsize, self.architecture[n],
                               device=self.act_buf[n].device)

        m = self.M_window[:, self.theta_slices[n]]          # (pop, 1) or (pop, w)
        m = m.clamp(M_MIN, float(avail))
        m_lo = torch.floor(m)
        lam = m - m_lo                                       # in [0, 1)

        # Cumulative sums over the tail let both boxcars be read off without
        # summing twice: cs[j] is the sum of the j+1 most recent samples.
        k_max = int(m_lo.max().item()) + 1
        k_max = min(k_max, avail)
        tail = self._tail(n, k_max)                          # (k_max, pop, dim)
        cs = torch.cumsum(tail.flip(0), dim=0)               # newest first

        # Gather per individual: index (M_lo - 1) and (M_lo), clamped in range.
        idx_lo = (m_lo.long() - 1).clamp(0, k_max - 1)       # (pop, n_slot)
        idx_hi = m_lo.long().clamp(0, k_max - 1)

        # cs is (k, pop, dim); move the population axis first to gather along k.
        csp = cs.permute(1, 0, 2)                            # (pop, k, dim)
        dim = csp.shape[-1]

        # Under 'global'/'layer' one index serves the whole node and must be
        # broadcast across its units; under 'node' each unit already has its
        # own. Both end up as (pop, 1, dim) gather indices.
        if idx_lo.shape[1] == 1:
            g_lo = idx_lo.view(-1, 1, 1).expand(-1, 1, dim)
            g_hi = idx_hi.view(-1, 1, 1).expand(-1, 1, dim)
        else:
            g_lo = idx_lo.unsqueeze(1)
            g_hi = idx_hi.unsqueeze(1)

        s_lo = torch.gather(csp, 1, g_lo).squeeze(1)         # (pop, dim)
        s_hi = torch.gather(csp, 1, g_hi).squeeze(1)
        d_lo, d_hi = m_lo, m_lo + 1.0

        # Guard the upper boxcar when M_lo + 1 exceeds what is available.
        over = (d_hi > avail)
        s_hi = torch.where(over, s_lo, s_hi)
        d_hi = torch.where(over, d_lo, d_hi)

        return (1.0 - lam) * (s_lo / d_lo) + lam * (s_hi / d_hi)

    # ------------------------------------------------------------------
    # Rollout reset
    # ------------------------------------------------------------------

    def reset_weights(self):
        """Re-init plastic weights AND the averaging buffers / step counter."""
        # HANNet.reset_weights calls _init_buffers, which this class overrides,
        # so the ring buffers are reallocated at the cap rather than at M.
        super().reset_weights()

    # ------------------------------------------------------------------
    # ES interface: parent block as prefix, theta as contiguous tail
    # ------------------------------------------------------------------

    def get_models_params(self):
        """Flat params for ALL individuals: parent block, then theta tail.

        Parent layout is group-major (all individuals for a given group/layer
        block are contiguous), so appending theta as a flat tail of length
        popsize * n_theta preserves that convention.
        """
        parent = super().get_models_params()
        tail = self.theta.detach().cpu().numpy().reshape(-1)
        return np.concatenate([parent, tail])

    def get_a_model_params(self):
        """Flat params for the FIRST individual: parent block, then its theta."""
        parent = super().get_a_model_params()
        tail = self.theta[0].detach().cpu().numpy().reshape(-1)
        return np.concatenate([parent, tail])

    def set_models_params(self, flat_params):
        """Load a population from the solver, shape (popsize, n_parent + n_theta).

        The parent reads by explicit column offsets and ignores trailing
        columns, so it can be handed the full matrix unmodified.
        """
        super().set_models_params(flat_params)
        theta = np.ascontiguousarray(flat_params[:, self.n_parent_params:])
        if theta.shape[1] != self.n_theta:
            raise ValueError(
                f"expected {self.n_theta} theta columns, got {theta.shape[1]}; "
                f"solver num_params must be {self.n_parent_params + self.n_theta}")
        self.theta = torch.from_numpy(theta).float().to(self.weights[0].device)

    def set_a_model_params(self, flat_params):
        """Load one individual and broadcast to all slots (evaluation path)."""
        super().set_a_model_params(flat_params)
        theta = np.ascontiguousarray(flat_params[self.n_parent_params:])
        if theta.shape[0] != self.n_theta:
            raise ValueError(
                f"expected {self.n_theta} theta values, got {theta.shape[0]}")
        t = torch.from_numpy(theta).float().to(self.weights[0].device)
        self.theta = t.unsqueeze(0).repeat(self.popsize, 1)

    def get_n_params_a_model(self):
        return self.n_parent_params + self.n_theta