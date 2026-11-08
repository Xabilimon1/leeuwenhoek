import numpy as np

# Named constants derived mechanistically from the weights in EXPLANATION.md:
# G_R: Relative gap threshold when x_2 > x_1 where cancellation of (x_1, x_2)
#      causes fallback to x_0 memory (predicting position 0).
G_R = 0.0232

# G_L: Relative gap threshold when x_2 < x_1 where cancellation of (x_1, x_2)
#      causes fallback to x_0 memory (predicting position 0).
G_L = 0.0243

# G_20: Small boundary shift / overshoot of position 2 into position 0 near x_2 ~ x_0.
G_20 = 0.0069

# G_10: Small boundary shift / overshoot of position 1 into position 0 near x_1 ~ x_0.
G_10 = 0.0043


def predict(x: np.ndarray) -> np.ndarray:
    """Predict median position (0..2) matching the RNN model's behaviour.
    
    Parameters
    ----------
    x : np.ndarray of shape (n, 3) or (3,)
        Input sequences of 3 real numbers.
        
    Returns
    -------
    np.ndarray of shape (n,) containing integer predictions in {0, 1, 2}.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
        
    x0 = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]
    
    # Baseline decision rule: find the true median position
    # The true median is at:
    #   position 1 if x1 is between x0 and x2
    #   position 2 if x2 is between x0 and x1
    #   position 0 if x0 is between x1 and x2
    ranks = np.argsort(x, axis=1)
    pred = ranks[:, 1].copy()
    
    # 1. Primary error mechanism (Cancellation & fallback to x_0):
    # When x1 and x2 are very close, the step-3 feedforward input cancels the
    # step-2 recurrent memory of x1. The network loses the ability to resolve
    # the small gap between x1 and x2, leaving only the step-1 residual of x0,
    # causing the readout to predict position 0.
    
    # Positive half-space (x0 < min(x1, x2)):
    mask_r_pos = (x0 < x1) & (x1 < x2) & ((x2 - x1) < G_R * (x1 - x0))
    mask_l_pos = (x0 < x2) & (x2 < x1) & ((x1 - x2) < G_L * (x2 - x0))
    pred[mask_r_pos] = 0
    pred[mask_l_pos] = 0
    
    # Negative half-space by sign-flip symmetry (x0 > max(x1, x2)):
    mask_r_neg = (x2 < x1) & (x1 < x0) & ((x1 - x2) < G_R * (x0 - x1))
    mask_l_neg = (x1 < x2) & (x2 < x0) & ((x2 - x1) < G_L * (x0 - x2))
    pred[mask_r_neg] = 0
    pred[mask_l_neg] = 0
    
    # 2. Secondary error mechanisms (Readout boundary shifts when x0 is the median):
    # Due to small asymmetric offsets in the output layer weights:
    
    # Position 2 slightly overshoots into position 0 near x2 ~ x0:
    mask_20_pos = (x1 < x0) & (x0 < x2) & ((x2 - x0) < G_20 * (x0 - x1))
    mask_20_neg = (x2 < x0) & (x0 < x1) & ((x0 - x2) < G_20 * (x1 - x0))
    pred[mask_20_pos] = 2
    pred[mask_20_neg] = 2
    
    # Position 1 slightly overshoots into position 0 near x1 ~ x0:
    mask_10_pos = (x2 < x0) & (x0 < x1) & ((x1 - x0) < G_10 * (x0 - x2))
    mask_10_neg = (x1 < x0) & (x0 < x2) & ((x0 - x1) < G_10 * (x2 - x0))
    pred[mask_10_pos] = 1
    pred[mask_10_neg] = 1
    
    return pred
