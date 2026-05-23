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
