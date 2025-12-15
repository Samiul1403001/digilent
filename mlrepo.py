import numpy as np

# LSTM parameters (PyTorch-style)
W_ih = np.load("npweights/lstm.weight_ih_l0.npy")   # (4*H, input_size)
W_hh = np.load("npweights/lstm.weight_hh_l0.npy")   # (4*H, H)
b_ih = np.load("npweights/lstm.bias_ih_l0.npy")     # (4*H,)
b_hh = np.load("npweights/lstm.bias_hh_l0.npy")     # (4*H,)

# Fully connected layer 1
fc1_W = np.load("npweights/fc1.weight.npy")         # (fc1_out, H)
fc1_b = np.load("npweights/fc1.bias.npy")           # (fc1_out,)

# Output layer
out_W = np.load("npweights/olayer.weight.npy")      # (out_dim, fc1_out)
out_b = np.load("npweights/olayer.bias.npy")        # (out_dim,)

# ----------------------------------------------------------
# Utility activation functions
# ----------------------------------------------------------
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----------------------------------------------------------
# LSTM forward pass (PyTorch-compatible)
# ----------------------------------------------------------
def lstm_forward(X, W_ih, W_hh, b_ih, b_hh, hidden_size=20):
    seq_len = X.shape[0]

    h = np.zeros((hidden_size,))
    c = np.zeros((hidden_size,))

    # PyTorch gate order: input, forget, cell, output
    W_i, W_f, W_g, W_o = np.split(W_ih, 4)
    U_i, U_f, U_g, U_o = np.split(W_hh, 4)
    b_i, b_f, b_g, b_o = np.split(b_ih + b_hh, 4)

    for t in range(seq_len):
        x_t = X[t]

        i = sigmoid(W_i @ x_t + U_i @ h + b_i)
        f = sigmoid(W_f @ x_t + U_f @ h + b_f)
        g = np.tanh(W_g @ x_t + U_g @ h + b_g)
        o = sigmoid(W_o @ x_t + U_o @ h + b_o)

        c = f * c + i * g
        h = o * np.tanh(c)

    return h  # final hidden state

# ----------------------------------------------------------
# Fully connected layers
# ----------------------------------------------------------
def fc1_forward(h, W, b):
    return gelu(W @ h + b)

def out_forward(h, W, b):
    return sigmoid(W @ h + b)

# ----------------------------------------------------------
# Full model forward
# ----------------------------------------------------------
def model_forward(sample,
                  W_ih, W_hh, b_ih, b_hh,
                  fc1_W, fc1_b,
                  out_W, out_b):

    # sample shape: (1, 3, 61)
    X = sample[0].T  # => (61, 3) sequence-first

    h = lstm_forward(X, W_ih, W_hh, b_ih, b_hh)
    z = fc1_forward(h, fc1_W, fc1_b)
    y = out_forward(z, out_W, out_b)
    return y
