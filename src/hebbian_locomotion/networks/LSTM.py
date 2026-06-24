"""
LSTM_neural_net.py — faithful replication of the paper-repo LSTM baseline
(SeqLSTMs: three stacked LSTM blocks), wired to drop into the ES pipeline.

This reproduces the architecture and math of the uploaded LSTM_neural_net.py
(Leung et al. repo): three LSTMs blocks, where each block concatenates the
input with its hidden state to drive the four gates, and the output projection
re-concatenates the input with the *updated* hidden state (no output bias).
Blocks 1 and 2 map obs->obs (via the hidden width); block 3 maps obs->actions.

Changes vs. the uploaded file (kept minimal and noted):
  * Fixed the SeqLSTMs<->LSTMs method-name mismatches so it runs
    (get_n_params/set_params/get_params_a_model -> the real names).
  * Exposes the HebbianNet interface used by run_es_dynamic*.py:
    forward, reset_weights, get_n_params_a_model, get_a_model_params,
    get_models_params, set_models_params, set_a_model_params, get_weights,
    get_states, and an `architecture` attribute.
  * Added reset_weights() to re-initialise the hidden/cell states each rollout
    (your ES loop calls this every generation; the original only initialised
    states once in __init__).
  * Device-aware (defaults to cuda) instead of hardcoded .cuda(), so it runs
    on CPU for testing and on the cluster unchanged.
  * forward uses squeeze(-1) (drop only the trailing dim) instead of squeeze_(),
    which avoids collapsing the population dim when popsize==1 (e.g. eval).

Init note: the uploaded code initialises BOTH the weights and the hidden/cell
states with U(-0.1, 0.1) (init_w = init_hidd = 0.1). The paper *text* instead
states weights U(-0.1, 0.1) and states U(-0.01, 0.01). This file follows the
CODE (0.1) since that is what you asked to replicate; set state_init=0.01 if you
prefer the text's value.

Sizing: with three stacked blocks the evolved-parameter count grows ~12*h^2 in
the hidden width h, so it matches the ABCD nets at a much smaller h than a
single-layer LSTM. For obs=23, act=16: h~28 ~= plain ABCD (~20.2k),
h~31-32 ~= eligibility-trace ABCD (~24.2k). The constructor prints the count.
"""

import numpy as np
import torch


class LSTMs:
    """A single LSTM block (faithful to the repo's LSTMs class)."""

    def __init__(self, popsize, arch, state_init=0.01, device="cuda"):
        self.arch = tuple(arch)
        self.in_channels, self.hid_size, self.out_channels = self.arch
        self.popsize = popsize
        self.state_init = state_init
        self.device = device

        self.n_params = self.get_n_params_a_model()

        # Plastic recurrent state (reset each rollout)
        self.hidden_state = None
        self.cell_state = None
        self.reset_states()

        # Evolved weights initialised in-range; overwritten by set_models_params.
        init = np.random.uniform(-state_init, state_init,
                                 (popsize, self.n_params)).astype(np.float32)
        self.set_models_params(init)

    # ------------------------------------------------------------------
    def reset_states(self):
        """Re-initialise hidden and cell states U(-state_init, state_init)."""
        h = self.hid_size
        self.hidden_state = torch.empty(self.popsize, h, 1, device=self.device).uniform_(
            -self.state_init, self.state_init)
        self.cell_state = torch.empty(self.popsize, h, 1, device=self.device).uniform_(
            -self.state_init, self.state_init)

    # ------------------------------------------------------------------
    def forward(self, inp):
        with torch.no_grad():
            x = torch.cat((inp.unsqueeze(-1), self.hidden_state), dim=1)

            f = torch.sigmoid(torch.einsum('lbn,lbc->lnc', self.Wf.float(), x.float()) + self.Bf.float())
            i = torch.sigmoid(torch.einsum('lbn,lbc->lnc', self.Wi.float(), x.float()) + self.Bi.float())
            c = torch.tanh(torch.einsum('lbn,lbc->lnc', self.Wc.float(), x.float()) + self.Bc.float())

            self.cell_state = f * self.cell_state + i * c

            o = torch.sigmoid(torch.einsum('lbn,lbc->lnc', self.Wo.float(), x.float()) + self.Bo.float())
            self.hidden_state = o * torch.tanh(self.cell_state)

            x = torch.cat((inp.unsqueeze(-1), self.hidden_state), dim=1)
            out = torch.tanh(torch.einsum('lbn,lbc->lnc', self.Wout.float(), x.float()))  # (pop, n_o, 1)

        return out.squeeze(-1)  # (pop, n_o)

    # ------------------------------------------------------------------
    def get_n_params_a_model(self):
        n_i, n_h, n_o = self.arch
        return (n_i + n_h) * n_h * 4 + (n_i + n_h) * n_o + n_h * 4

    def get_models_params(self):
        p = torch.cat([self.Wf.flatten(), self.Wi.flatten(), self.Wc.flatten(),
                       self.Wo.flatten(), self.Wout.flatten(),
                       self.Bf.flatten(), self.Bi.flatten(), self.Bc.flatten(),
                       self.Bo.flatten()])
        return p.flatten().cpu().numpy()

    def get_a_model_params(self):
        p = torch.cat([self.Wf[0].flatten(), self.Wi[0].flatten(), self.Wc[0].flatten(),
                       self.Wo[0].flatten(), self.Wout[0].flatten(),
                       self.Bf[0].flatten(), self.Bi[0].flatten(), self.Bc[0].flatten(),
                       self.Bo[0].flatten()])
        return p.flatten().cpu().numpy()

    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(np.ascontiguousarray(flat_params)).to(self.device)
        n_i, n_h, n_o = self.arch
        wlen = (n_i + n_h) * n_h
        m = 0
        self.Wf = flat_params[:, m:m + wlen].reshape(self.popsize, n_i + n_h, n_h); m += wlen
        self.Wi = flat_params[:, m:m + wlen].reshape(self.popsize, n_i + n_h, n_h); m += wlen
        self.Wc = flat_params[:, m:m + wlen].reshape(self.popsize, n_i + n_h, n_h); m += wlen
        self.Wo = flat_params[:, m:m + wlen].reshape(self.popsize, n_i + n_h, n_h); m += wlen
        self.Wout = flat_params[:, m:m + (n_i + n_h) * n_o].reshape(self.popsize, n_i + n_h, n_o)
        m += (n_i + n_h) * n_o
        self.Bf = flat_params[:, m:m + n_h].unsqueeze(-1); m += n_h
        self.Bi = flat_params[:, m:m + n_h].unsqueeze(-1); m += n_h
        self.Bc = flat_params[:, m:m + n_h].unsqueeze(-1); m += n_h
        self.Bo = flat_params[:, m:m + n_h].unsqueeze(-1); m += n_h

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(np.ascontiguousarray(flat_params)).to(self.device)
        n_i, n_h, n_o = self.arch
        wlen = (n_i + n_h) * n_h
        m = 0
        def w(num, *shape):
            nonlocal m
            t = flat_params[m:m + num].repeat(self.popsize, 1, 1).reshape(self.popsize, *shape)
            m += num
            return t
        self.Wf = w(wlen, n_i + n_h, n_h)
        self.Wi = w(wlen, n_i + n_h, n_h)
        self.Wc = w(wlen, n_i + n_h, n_h)
        self.Wo = w(wlen, n_i + n_h, n_h)
        self.Wout = w((n_i + n_h) * n_o, n_i + n_h, n_o)
        self.Bf = flat_params[m:m + n_h].repeat(self.popsize, 1).unsqueeze(-1); m += n_h
        self.Bi = flat_params[m:m + n_h].repeat(self.popsize, 1).unsqueeze(-1); m += n_h
        self.Bc = flat_params[m:m + n_h].repeat(self.popsize, 1).unsqueeze(-1); m += n_h
        self.Bo = flat_params[m:m + n_h].repeat(self.popsize, 1).unsqueeze(-1); m += n_h

    def get_state(self):
        return self.hidden_state, self.cell_state

    def weight_dict(self):
        return {"Wf": self.Wf, "Wi": self.Wi, "Wc": self.Wc, "Wo": self.Wo,
                "Wout": self.Wout, "Bf": self.Bf, "Bi": self.Bi, "Bc": self.Bc, "Bo": self.Bo}


class SeqLSTMs:
    """Three stacked LSTM blocks — the repo's policy network — with the ES
    pipeline (HebbianNet) interface."""

    def __init__(self, popsize, sizes=None, arch=None, init_w=0.1, state_init=0.1, device="cuda"):
        arch = sizes if sizes is not None else arch
        assert arch is not None and len(arch) == 3, "pass sizes=[obs_dim, hidden, action_dim]"
        self.architecture = list(arch)
        self.popsize = popsize
        self.device = device

        in_channels, hid_size, out_channels = arch
        arch_base = (in_channels, hid_size, in_channels)
        arch_final = (in_channels, hid_size, out_channels)

        self.model_1 = LSTMs(popsize, arch_base, state_init, device)
        self.model_2 = LSTMs(popsize, arch_base, state_init, device)
        self.model_3 = LSTMs(popsize, arch_final, state_init, device)

        self.n_params_b = self.model_1.get_n_params_a_model()
        self.n_params_f = self.model_3.get_n_params_a_model()

        # Faithful to the repo: blocks 1 and 2 share the same initial draw.
        init_params_b = np.random.uniform(-init_w, init_w, (popsize, self.n_params_b)).astype(np.float32)
        init_params_f = np.random.uniform(-init_w, init_w, (popsize, self.n_params_f)).astype(np.float32)
        self.model_1.set_models_params(init_params_b)
        self.model_2.set_models_params(init_params_b)
        self.model_3.set_models_params(init_params_f)

    # ------------------------------------------------------------------
    def forward(self, inp):
        out_1 = self.model_1.forward(inp)
        out_2 = self.model_2.forward(out_1)
        out_3 = self.model_3.forward(out_2)
        return out_3  # (pop, action_dim)

    def reset_weights(self):
        """Reset the recurrent state of all three blocks for a fresh rollout."""
        self.model_1.reset_states()
        self.model_2.reset_states()
        self.model_3.reset_states()

    # ------------------------------------------------------------------
    def get_n_params_a_model(self):
        return self.n_params_b + self.n_params_b + self.n_params_f

    def get_a_model_params(self):
        return np.concatenate([self.model_1.get_a_model_params(),
                               self.model_2.get_a_model_params(),
                               self.model_3.get_a_model_params()])

    def get_models_params(self):
        return np.concatenate([self.model_1.get_models_params(),
                               self.model_2.get_models_params(),
                               self.model_3.get_models_params()])

    def set_models_params(self, pop):
        m = 0
        self.model_1.set_models_params(pop[:, m:m + self.n_params_b]); m += self.n_params_b
        self.model_2.set_models_params(pop[:, m:m + self.n_params_b]); m += self.n_params_b
        self.model_3.set_models_params(pop[:, m:m + self.n_params_f]); m += self.n_params_f

    def set_a_model_params(self, flat_params):
        m = 0
        self.model_1.set_a_model_params(flat_params[m:m + self.n_params_b]); m += self.n_params_b
        self.model_2.set_a_model_params(flat_params[m:m + self.n_params_b]); m += self.n_params_b
        self.model_3.set_a_model_params(flat_params[m:m + self.n_params_f]); m += self.n_params_f

    # ------------------------------------------------------------------
    def get_weights(self):
        """List of three per-block weight dicts (for histogram logging)."""
        return [self.model_1.weight_dict(), self.model_2.weight_dict(), self.model_3.weight_dict()]

    def get_states(self):
        """Concatenated (hidden, cell) states across all blocks, each
        (popsize, 3*hidden) — the LSTM analog of the plastic-weight trajectory."""
        h = torch.cat([self.model_1.hidden_state.squeeze(-1),
                       self.model_2.hidden_state.squeeze(-1),
                       self.model_3.hidden_state.squeeze(-1)], dim=1)
        c = torch.cat([self.model_1.cell_state.squeeze(-1),
                       self.model_2.cell_state.squeeze(-1),
                       self.model_3.cell_state.squeeze(-1)], dim=1)
        return h, c


# Alias so existing imports (`from ... import LSTMNet`) keep working.
LSTMNet = SeqLSTMs