"""Twin implementation of the argmedian neural network.

This re-implementation is built mechanistically from the analysis of the
trained 8-unit recurrent neural network with ReLU activations and no biases.

The network computes the argmedian (index of the median element in a 3-element sequence)
via pairwise comparisons between inputs. Because recurrent propagation across time
steps incurs slight gain mismatches relative to the direct input feedthrough, the
decision boundaries between classes are slightly tilted relative hyperplanes.

The named constants below represent these relative boundary offsets (gaps),
derived analytically from the weight matrices in EXPLANATION.md:
  - GAP_01: relative gap for the comparison between x0 and x1
  - GAP_02: relative gap for the comparison between x0 and x2
  - GAP_12: relative gap for the comparison between x1 and x2
"""

import numpy as np

# Relative gap constants derived from network weights:
GAP_01 = 0.00064
GAP_02 = 0.00078
GAP_12 = 0.00036


def predict(x: np.ndarray) -> np.ndarray:
    """Predict the position (0, 1, or 2) of the median element.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n, 3) with float values.

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing integer predictions in {0, 1, 2}.
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze_output = False
    if x.ndim == 1:
        x = x[None, :]
        squeeze_output = True

    x0 = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]

    # Pairwise directional comparisons with recurrent boundary tilts
    c01 = x0 > (1.0 - GAP_01) * x1
    c02 = x0 > (1.0 - GAP_02) * x2
    c12 = x1 > (1.0 + GAP_12) * x2

    # A number is the median if it falls between the other two:
    # - x0 is median iff (x0 > x1) != (x0 > x2)
    # - x1 is median iff (x0 > x1) == (x1 > x2)
    # - x2 is median iff (x0 > x2) != (x1 > x2)
    pred = np.zeros(len(x), dtype=np.int64)
    pred[c01 != c02] = 0
    pred[c01 == c12] = 1
    pred[c02 != c12] = 2

    if squeeze_output:
        return pred[0]
    return pred
