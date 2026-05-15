import numpy as np
import torch


def var_norm(w, eps=1e-5):
    """Normalize weights by subtracting mean and dividing by standard deviation (layer-wise).
    
    Corresponds to Eq. 3 in the paper:
        w_ij^k = (w_ij^k - mean(W^k)) / std(W^k)
    """
    mean = torch.mean(input=w, dim=[1, 2], keepdim=True)
    var = torch.var(input=w, dim=[1, 2], keepdim=True)
    w = (w - mean) / torch.sqrt(var + eps)
    return w


def max_norm(w, eps=1e-5):
    """Normalize weights by dividing by the maximum absolute value (layer-wise).
    
    Corresponds to Eq. 2 in the paper:
        w_ij^k = w_ij^k / max(abs(W^k))
    """
    max_val = torch.max(torch.abs(w).flatten(start_dim=1, end_dim=2), dim=1)
    max_val = max_val[0].unsqueeze(1).unsqueeze(2)
    w = w / max_val
    return w


def clip_weights(w, eps=1e-5):
    """Clip weights to the range [-1.0, 1.0]."""
    w = torch.clamp(w, min=-1.0, max=1.0)
    return w


class HebbianNet:
    def __init__(self,
                 popsize,
                 sizes,
                 init_noise=0.01,
                 norm_mode='var'):
        """
        A feedforward network with ABCD Hebbian plasticity (Eq. 1 in the paper).

        The ES optimizer evolves the Hebbian coefficients (A, B, C, D) and learning
        rates (lr) for each synapse. During a rollout, the actual connection weights
        are updated at every forward pass using the Hebbian rule. The weights must be
        re-initialized before each new rollout via reset_weights().

        Args:
            popsize:    Number of parallel individuals (matches num_envs).
            sizes:      Layer sizes, e.g. [obs_dim, 64, 32, action_dim].
            init_noise: Uniform range for initial connection weights.
            norm_mode:  Weight normalization method ('var', 'max', or 'clip').
        """
        self.popsize = popsize
        self.architecture = sizes
        self.init_noise = init_noise
        self.n_layers = len(sizes) - 1

        # --- Connection weights (plastic, reset each episode) ---
        self.weights = []
        for i in range(self.n_layers):
            w = torch.Tensor(popsize, self.architecture[i], self.architecture[i + 1]).uniform_(-init_noise, init_noise).cuda()
            self.weights.append(w)

        # --- Pre-allocated ones array for broadcasting in hebbian_update ---
        self.one_array = []
        for i in range(self.n_layers):
            ones = torch.ones(popsize, sizes[i], sizes[i + 1]).cuda()
            self.one_array.append(ones)

        # --- Hebbian coefficients (evolved by ES, fixed within an episode) ---
        self.A = self._init_hebbian_coeffs(popsize, sizes)
        self.B = self._init_hebbian_coeffs(popsize, sizes)
        self.C = self._init_hebbian_coeffs(popsize, sizes)
        self.D = self._init_hebbian_coeffs(popsize, sizes)
        self.lr = self._init_hebbian_coeffs(popsize, sizes)

        # --- Select normalization function ---
        if norm_mode == 'var':
            self.WeightStand = var_norm
        elif norm_mode == 'max':
            self.WeightStand = max_norm
        elif norm_mode == 'clip':
            self.WeightStand = clip_weights

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_hebbian_coeffs(self, popsize, sizes):
        """Initialize one set of Hebbian coefficient tensors (one per layer).
        
        Paper: "Hebbian coefficients are initialized with a normal distribution
        with a mean of zero and a standard deviation of 0.01."
        (The code uses 0.025 — kept as-is from the original implementation.)
        """
        coeffs = []
        for i in range(self.n_layers):
            c = torch.normal(0, 0.01, (popsize, sizes[i], sizes[i + 1])).cuda()
            coeffs.append(c)
        return coeffs

    def reset_weights(self):
        """Re-initialize connection weights to small random values.
        
        MUST be called before each new rollout so that the Hebbian rules
        transform a fresh set of weights rather than carrying over state
        from the previous generation.
        """
        for i in range(self.n_layers):
            n_in = self.architecture[i]
            n_out = self.architecture[i + 1]
            self.weights[i] = torch.Tensor(
                self.popsize, n_in, n_out
            ).uniform_(-self.init_noise, self.init_noise).cuda()

    # ------------------------------------------------------------------
    # Forward pass + Hebbian update
    # ------------------------------------------------------------------

    def forward(self, pre):
        """Run a forward pass through all layers, applying Hebbian updates.

        Args:
            pre: Observation tensor of shape (popsize, obs_dim).

        Returns:
            Output tensor of shape (popsize, action_dim), values in (-1, 1).
        """
        with torch.no_grad():
            for i in range(self.n_layers):
                W = self.weights[i]

                # Linear transform + tanh activation
                post = torch.tanh(torch.einsum('ij, ijk -> ik', pre, W.float()))

                # Apply ABCD Hebbian update (Eq. 1) and normalize
                self.weights[i] = self.hebbian_update(
                    i, W, pre, post,
                    self.A[i], self.B[i], self.C[i], self.D[i], self.lr[i]
                )

                # Output of this layer becomes input to the next
                pre = post

        return post.float().detach()

    def hebbian_update(self, layer_idx, weights, pre, post, A, B, C, D, lr):
        """Apply the ABCD Hebbian learning rule (Eq. 1 in the paper).

        delta_w_ij = lr * (A * o_i * o_j  +  B * o_i  +  C * o_j  +  D)

        Args:
            layer_idx: Index of the current layer (for one_array lookup).
            weights:   Current weight tensor, shape (pop, n_in, n_out).
            pre:       Pre-synaptic activations, shape (pop, n_in).
            post:      Post-synaptic activations, shape (pop, n_out).
            A, B, C, D, lr: Evolved Hebbian coefficients, each (pop, n_in, n_out).

        Returns:
            Updated and normalized weight tensor.
        """
        # Broadcast pre-synaptic activations: (pop, n_in) -> (pop, n_in, n_out)
        i = self.one_array[layer_idx] * pre.unsqueeze(2)

        # Broadcast post-synaptic activations: (pop, n_out) -> (pop, n_in, n_out)
        j = post.unsqueeze(2).expand(-1, -1, weights.shape[1]).transpose(1, 2)

        # Correlation term
        ij = i * j

        # ABCD update rule
        weights = weights + lr * (A * ij + B * i + C * j + D)

        # Normalize to prevent divergence
        weights = self.WeightStand(weights)

        return weights

    # ------------------------------------------------------------------
    # Parameter getters (for ES interface)
    # ------------------------------------------------------------------

    def get_weights(self):
        """Return the current connection weight tensors."""
        result = []
        for w in self.weights:
            result.append(w)
        return result

    def get_n_params_a_model(self):
        """Return the number of evolvable parameters for a single individual."""
        return len(self.get_a_model_params())

    def get_models_params(self):
        """Flatten ALL individuals' Hebbian coefficients into a single numpy array.
        
        Layout: [A_layer0, A_layer1, ..., B_layer0, ..., D_layerN, lr_layer0, ..., lr_layerN]
        Shape: (popsize * total_coeffs_per_individual,)
        """
        all_param_groups = [self.A, self.B, self.C, self.D, self.lr]
        flat_parts = []
        for param_group in all_param_groups:
            for layer_params in param_group:
                flat_parts.append(layer_params.flatten())
        p = torch.cat(flat_parts)
        return p.cpu().numpy()

    def get_a_model_params(self):
        """Flatten the FIRST individual's Hebbian coefficients into a numpy array.
        
        Used to initialize the ES solver's mu with a representative set of params.
        """
        all_param_groups = [self.A, self.B, self.C, self.D, self.lr]
        flat_parts = []
        for param_group in all_param_groups:
            for layer_params in param_group:
                flat_parts.append(layer_params[0].flatten())
        p = torch.cat(flat_parts)
        return p.cpu().numpy()

    # ------------------------------------------------------------------
    # Parameter setters (for ES interface)
    # ------------------------------------------------------------------

    def update_params(self, hebb_list, flat_params, start_index):
        """Unpack a slice of the flat parameter matrix into one Hebbian coefficient list.
        
        Each row of flat_params corresponds to one individual in the population.
        
        Args:
            hebb_list:    List of tensors to overwrite (e.g. self.A).
            flat_params:  Full parameter matrix, shape (popsize, total_params).
            start_index:  Column offset to start reading from.

        Returns:
            Updated column offset after consuming this coefficient's parameters.
        """
        m = start_index
        for i in range(len(hebb_list)):
            pop, n_in, n_out = hebb_list[i].shape
            num_elements = n_in * n_out
            hebb_list[i] = flat_params[:, m:m + num_elements].reshape(pop, n_in, n_out).cuda()
            m += num_elements
        return m

    def update_a_model_params(self, hebb_list, flat_params, start_index):
        """Unpack a single individual's flat params and broadcast to all individuals.
        
        Used for loading a trained checkpoint into all population slots.
        
        Args:
            hebb_list:    List of tensors to overwrite.
            flat_params:  Single individual's flat params, shape (total_params,).
            start_index:  Offset to start reading from.

        Returns:
            Updated offset.
        """
        m = start_index
        for i in range(len(hebb_list)):
            pop, n_in, n_out = hebb_list[i].shape
            num_elements = n_in * n_out
            hebb_list[i] = flat_params[m:m + num_elements].repeat(pop, 1, 1).reshape(pop, n_in, n_out).cuda()
            m += num_elements
        return m

    def set_models_params(self, flat_params):
        """Load a full population of Hebbian coefficients from the ES solver.
        
        Args:
            flat_params: numpy array of shape (popsize, n_params_per_individual).
        """
        flat_params = torch.from_numpy(flat_params)

        m = 0
        m = self.update_params(self.A, flat_params, m)
        m = self.update_params(self.B, flat_params, m)
        m = self.update_params(self.C, flat_params, m)
        m = self.update_params(self.D, flat_params, m)
        m = self.update_params(self.lr, flat_params, m)

    def set_a_model_params(self, flat_params):
        """Load a single individual's Hebbian coefficients and broadcast to all slots.
        
        Used for evaluation / testing a trained model.
        
        Args:
            flat_params: numpy array of shape (n_params_per_individual,).
        """
        flat_params = torch.from_numpy(flat_params)

        m = 0
        m = self.update_a_model_params(self.A, flat_params, m)
        m = self.update_a_model_params(self.B, flat_params, m)
        m = self.update_a_model_params(self.C, flat_params, m)
        m = self.update_a_model_params(self.D, flat_params, m)
        m = self.update_a_model_params(self.lr, flat_params, m)