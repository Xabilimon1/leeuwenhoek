"""Hand-written twin re-implementation of the trained median network.

Built mechanistically from the decision boundaries analyzed in EXPLANATION.md.
Predicts the position (0, 1, or 2) of the median input, matching the trained
neural network's behavior and its subtle boundary offsets without copying
the network weights or compiling linear regions.
"""

import numpy as np

# Named boundary constants derived analytically from the network weights
# at each pairwise decision interface (see EXPLANATION.md for derivations).

# Boundary B3: Decision threshold between x0 and x1 when x2 < min(x0, x1)
SLOPE_01_LOW_X2 = 0.999608

# Boundary B6: Decision threshold between x0 and x1 when x2 > max(x0, x1)
SLOPE_01_HIGH_X2 = 1.000544

# Boundary B2: Interface between predicting 2 and 0 when x0 < x1
# (x2 near x0, candidate minimum)
SLOPE_B2_X0 = 1.000902
COEFF_B2_X1 = -0.000132

# Boundary B1: Interface between predicting 2 and 1 when x0 < x1
# (x2 near x1, candidate maximum)
SLOPE_B1_X1 = 0.999642
COEFF_B1_X0 = 0.000930

# Boundary B4: Interface between predicting 1 and 2 when x1 < x0
# (x2 near x1, candidate minimum)
SLOPE_B4_X1 = 1.001105
COEFF_B4_X0 = -0.000432

# Boundary B5: Interface between predicting 2 and 0 when x1 < x0
# (x2 near x0, candidate maximum)
SLOPE_B5_X0 = 1.000779
COEFF_B5_X1 = -0.000982


def predict(x: np.ndarray) -> np.ndarray:
    """Predict median position (0..2) for input sequence x.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n, 3) or (3,) containing float inputs.

    Returns
    -------
    np.ndarray
        Array of shape (n,) or (1,) of integer positions in {0, 1, 2}.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]

    x0 = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]
    n = x.shape[0]
    out = np.zeros(n, dtype=np.int64)

    # Step 1 & 2: Compare x0 and x1 with boundary tilt depending on x2
    thresh_01 = np.where(x2 < x0, SLOPE_01_LOW_X2 * x0, SLOPE_01_HIGH_X2 * x0)
    mask_0_lt_1 = x1 > thresh_01

    # Case A: x0 < x1 (x0 is candidate min, x1 is candidate max)
    b_min = SLOPE_B2_X0 * x0 + COEFF_B2_X1 * x1
    b_max = SLOPE_B1_X1 * x1 + COEFF_B1_X0 * x0

    pred0_a = mask_0_lt_1 & (x2 < b_min)
    pred1_a = mask_0_lt_1 & (x2 > b_max)
    pred2_a = mask_0_lt_1 & (~pred0_a) & (~pred1_a)

    out[pred0_a] = 0
    out[pred1_a] = 1
    out[pred2_a] = 2

    # Case B: x1 < x0 (x1 is candidate min, x0 is candidate max)
    mask_1_lt_0 = ~mask_0_lt_1
    b_min_b = SLOPE_B4_X1 * x1 + COEFF_B4_X0 * x0
    b_max_b = SLOPE_B5_X0 * x0 + COEFF_B5_X1 * x1

    pred1_b = mask_1_lt_0 & (x2 < b_min_b)
    pred0_b = mask_1_lt_0 & (x2 > b_max_b)
    pred2_b = mask_1_lt_0 & (~pred1_b) & (~pred0_b)

    out[pred1_b] = 1
    out[pred0_b] = 0
    out[pred2_b] = 2

    return out
