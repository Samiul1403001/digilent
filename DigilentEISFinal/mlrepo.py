import numpy as np

class NumpySimpleSoHLSTM:
    def __init__(self, npz_path):
        weights = np.load(npz_path)
        
        # Extract LSTM Layer 0
        self.w_ih = weights['lstm.weight_ih_l0']
        self.w_hh = weights['lstm.weight_hh_l0']
        self.b_ih = weights['lstm.bias_ih_l0']
        self.b_hh = weights['lstm.bias_hh_l0']
        
        # Extract Linear Layers
        self.fc1_w = weights['fc1.weight']
        self.fc1_b = weights['fc1.bias']
        self.fc2_w = weights['fc2.weight']
        self.fc2_b = weights['fc2.bias']
        
        self.hidden_size = self.w_hh.shape[1]

    def sigmoid(self, x):
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def leaky_relu(self, x, negative_slope=0.01):
        return np.where(x > 0, x, x * negative_slope)

    def lstm_step(self, x_t, h_prev, c_prev):
        gates = np.dot(self.w_ih, x_t) + self.b_ih + np.dot(self.w_hh, h_prev) + self.b_hh
        i_gate, f_gate, g_gate, o_gate = np.split(gates, 4)
        
        i = self.sigmoid(i_gate)
        f = self.sigmoid(f_gate)
        g = np.tanh(g_gate)
        o = self.sigmoid(o_gate)
        
        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)
        
        return h_next, c_next

    def predict(self, sample):
        """
        sample: NumPy array of shape (1, 6, 31)
                (Batch_size, Features, Seq_Length)
        """
        # 1. Remove the batch dimension -> shape becomes (6, 31)
        # 2. Transpose to iterate over time steps -> shape becomes (31, 6)
        sequence = sample[0].T
        sequence[1] = np.log10(sequence[1])
        
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        
        for x_t in sequence:
            # 3. Check for zero-padding. If the row is all zeros, 
            # we've reached the end of the actual data. Break early.
            if np.all(x_t == 0):
                break
                
            h, c = self.lstm_step(x_t, h, c)
            
        # 4. Pass the LAST valid hidden state through linear layers
        out = np.dot(self.fc1_w, h) + self.fc1_b
        out = self.leaky_relu(out)
        out = np.dot(self.fc2_w, out) + self.fc2_b
        
        return float(out[0])

# # LSTM parameters (PyTorch-style)
# W_ih = np.load("npweights/lstm_weight_ih_l0.npy")   # (4*H, input_size)
# W_hh = np.load("npweights/lstm_weight_hh_l0.npy")   # (4*H, H)
# b_ih = np.load("npweights/lstm_bias_ih_l0.npy")     # (4*H,)
# b_hh = np.load("npweights/lstm_bias_hh_l0.npy")     # (4*H,)

# # Fully connected layer 1
# fc1_W = np.load("npweights/fc1_weight.npy")         # (fc1_out, H)
# fc1_b = np.load("npweights/fc1_bias.npy")           # (fc1_out,)

# # Output layer
# out_W = np.load("npweights/olayer_weight.npy")      # (out_dim, fc1_out)
# out_b = np.load("npweights/olayer_bias.npy")        # (out_dim,)

# # ----------------------------------------------------------
# # Utility activation functions
# # ----------------------------------------------------------
# def gelu(x):
#     return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # ----------------------------------------------------------
# # LSTM forward pass (PyTorch-compatible)
# # ----------------------------------------------------------
# def lstm_forward(X, W_ih, W_hh, b_ih, b_hh, hidden_size=10):
#     seq_len = X.shape[0]

#     h = np.zeros((hidden_size,))
#     c = np.zeros((hidden_size,))

#     # PyTorch gate order: input, forget, cell, output
#     W_i, W_f, W_g, W_o = np.split(W_ih, 4)
#     U_i, U_f, U_g, U_o = np.split(W_hh, 4)
#     b_i, b_f, b_g, b_o = np.split(b_ih + b_hh, 4)

#     for t in range(seq_len):
#         x_t = X[t]

#         i = sigmoid(W_i @ x_t + U_i @ h + b_i)
#         f = sigmoid(W_f @ x_t + U_f @ h + b_f)
#         g = np.tanh(W_g @ x_t + U_g @ h + b_g)
#         o = sigmoid(W_o @ x_t + U_o @ h + b_o)

#         c = f * c + i * g
#         h = o * np.tanh(c)

#     return h  # final hidden state

# # ----------------------------------------------------------
# # Fully connected layers
# # ----------------------------------------------------------
# def fc1_forward(h, W, b):
#     return gelu(W @ h + b)

# def out_forward(h, W, b):
#     return sigmoid(W @ h + b)

# # ----------------------------------------------------------
# # Full model forward
# # ----------------------------------------------------------
# def model_forward(sample,
#                   W_ih, W_hh, b_ih, b_hh,
#                   fc1_W, fc1_b,
#                   out_W, out_b):

#     # sample shape: (1, 3, 61)
#     X = sample[0].T  # => (61, 3) sequence-first

#     h = lstm_forward(X, W_ih, W_hh, b_ih, b_hh)
#     z = fc1_forward(h, fc1_W, fc1_b)
#     y = out_forward(z, out_W, out_b)
#     return y
