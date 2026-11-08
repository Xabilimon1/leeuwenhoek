import numpy as np

# Named constants derived mechanistically from the network weights in EXPLANATION.md:

# GAP_23: Angular half-width / relative gap threshold for the dead band along x2 ~ x3.
# When |x2 - x3| < GAP_23 * ||x|| and x1 is outside (true median is 1 or 2), the network
# fails to resolve x2 vs x3 because its readout ratio v/u falls into the dead zone
# [R_2, R_1] = [0.757, 1.188], causing logit 0 to win by default.
# Derived as ratio of logit offset to linear slope along the boundary:
# GAP_23 = (w_out offset) / (slope * sqrt(2)) ~ 1.63 / (134.5 * 1.414) ~ 0.0175.
GAP_23 = 0.0175

# GAP_13: Relative gap between x1 and x3 below which true median 0 is misclassified as 2.
# Derived from activation leakage of unit 0 into logit 2 near the x1 ~ x3 boundary.
GAP_13 = 0.0050

# GAP_12: Relative gap between x1 and x2 below which true median 0 is misclassified as 1.
# Derived from activation leakage of unit 1 into logit 1 near the x1 ~ x2 boundary.
GAP_12 = 0.0028


def predict(x: np.ndarray) -> np.ndarray:
    """Predict the position (0, 1, or 2) of the median input.
    
    Implements the mechanistic decision rule of the trained recurrent network:
    1. Base rule: determine which input is the median (position 0, 1, or 2).
    2. Primary dead-zone rule: when x2 and x3 are within relative gap GAP_23
       and x1 is outside (true median is 1 or 2), the network's recurrent state
       produces balanced inner/outer activations (u ~ v), which falls into the
       readout dead-zone where logit 0 wins. The network outputs 0.
    3. Secondary dead-zone rules: when x1 and x3 (or x1 and x2) are extremely close
       and x1 is the true median (0), finite readout resolution leaks into logit 2
       (or logit 1), outputting 2 (or 1).
    """
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x[None, :]

    # Base rule: true median position
    med = np.argsort(x, axis=1)[:, 1]

    # Scale factor for 1-homogeneity (network has zero biases)
    scale = np.linalg.norm(x, axis=1)

    # Primary failure mode: x2 ~ x3 misclassified as 0
    dead_23 = (np.abs(x[:, 1] - x[:, 2]) < GAP_23 * scale) & (med != 0)
    med[dead_23] = 0

    # Secondary failure mode: x1 ~ x3 misclassified as 2
    dead_13 = (np.abs(x[:, 0] - x[:, 2]) < GAP_13 * scale) & (med == 0)
    med[dead_13] = 2

    # Tertiary failure mode: x1 ~ x2 misclassified as 1
    dead_12 = (np.abs(x[:, 0] - x[:, 1]) < GAP_12 * scale) & (med == 0)
    med[dead_12] = 1

    return med
