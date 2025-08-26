import numpy as np

# ----- utils -----
def he_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

def one_hot(y, num_classes):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1.0
    return oh

# ----- model -----
def init_mlp(sizes, seed=0):
    """
    sizes: [in_dim, h1, h2, ..., out_dim]
    returns params dict of W_i, b_i for each layer i=1..L-1
    """
    rng = np.random.default_rng(seed)
    params = {}
    for i in range(len(sizes)-1):
        params[f"W{i+1}"] = he_init(sizes[i], sizes[i+1])
        params[f"b{i+1}"] = np.zeros((1, sizes[i+1]))
    return params

def forward(X, params):
    """
    Returns logits and cache for backprop.
    cache contains layer inputs/outputs needed for gradients.
    """
    A = X
    cache = {"A0": X}
    L = len(params) // 2
    for i in range(1, L):
        Z = A @ params[f"W{i}"] + params[f"b{i}"]
        A = np.maximum(0, Z)  # ReLU
        cache[f"Z{i}"] = Z
        cache[f"A{i}"] = A
    # last layer (linear → logits)
    ZL = A @ params[f"W{L}"] + params[f"b{L}"]
    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = ZL  # logits
    return ZL, cache

def softmax_cross_entropy(logits, y):
    """
    y: int labels shape (N,)
    returns (loss, probs)
    """
    # stable softmax
    logits_shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shift)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    return loss, probs

def backward(params, cache, probs, y):
    """
    Backprop through softmax CE and ReLU MLP.
    Returns grads dict matching params keys.
    """
    grads = {}
    L = len(params) // 2
    N = probs.shape[0]
    Y = one_hot(y, probs.shape[1])

    # dLoss/dLogits
    dZ = (probs - Y) / N

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]
        grads[f"W{i}"] = A_prev.T @ dZ
        grads[f"b{i}"] = dZ.sum(axis=0, keepdims=True)

        if i > 1:
            dA_prev = dZ @ params[f"W{i}"].T
            Z_prev = cache[f"Z{i-1}"]
            dZ = dA_prev * (Z_prev > 0)  # ReLU'
    return grads

# ----- optimizers -----
class SGD:
    def __init__(self, params, lr=1e-2, weight_decay=0.0, momentum=0.0):
        self.lr = lr
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()} if momentum else None
        self.mu = momentum

    def step(self, params, grads):
        for k in params:
            g = grads[k]
            if self.wd:
                g = g + self.wd * params[k]
            if self.m is not None:
                self.m[k] = self.mu * self.m[k] + (1 - self.mu) * g
                g = self.m[k]
            params[k] -= self.lr * g

# ----- training loop -----
def train(X, y, sizes, epochs=200, batch_size=128, lr=1e-3,
          weight_decay=0.0, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    params = init_mlp(sizes, seed=seed)
    opt = SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)

    N = X.shape[0]
    for ep in range(1, epochs + 1):
        # shuffle
        idx = rng.permutation(N)
        Xs, ys = X[idx], y[idx]

        # mini-batches
        for i in range(0, N, batch_size):
            xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]

            # forward
            logits, cache = forward(xb, params)
            loss, probs = softmax_cross_entropy(logits, yb)

            # backward
            grads = backward(params, cache, probs, yb)

            # (optional) gradient clipping
            for k in grads: grads[k] = np.clip(grads[k], -1.0, 1.0)

            # optimizer step
            opt.step(params, grads)

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
            # simple train accuracy
            logits, _ = forward(X, params)
            pred = logits.argmax(axis=1)
            acc = (pred == y).mean()
            print(f"epoch {ep:4d} | loss {loss:.4f} | acc {acc:.3f}")

    return params

# ----- demo on synthetic data -----
if __name__ == "__main__":
    # toy 3-class dataset
    rng = np.random.default_rng(1)
    N, D, C = 1500, 10, 3
    X = rng.normal(size=(N, D))
    true_W = rng.normal(size=(D, C))
    y = (X @ true_W + 0.5 * rng.normal(size=(N, C))).argmax(axis=1)

    sizes = [D, 64, 64, C]  # MLP architecture
    params = train(X, y, sizes, epochs=100, batch_size=128, lr=1e-3)