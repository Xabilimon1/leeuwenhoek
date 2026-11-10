"""Hand-written twin of the 8-unit ReLU RNN trained to output the median position.

Rule (see EXPLANATION.md):
  1. Sort the three inputs. The answer is the median position, except that the
     decision between the median and its nearer neighbour is made with a
     slightly shifted boundary.
  2. For the pair of positions (i, j), i < j, that are the two values closest
     together around the median, with third position k, the network answers
     i when  x_i - x_j  >  T  and j otherwise... more precisely it answers the
     position the true median rule gives for d = x_i - x_j - T, where
         T = C[(i, j), case] * sqrt(s**2 + x_k**2),   s = (x_i + x_j) / 2,
     and case is "k_above" (x_k larger than the pair) or "k_below".
  The offsets are tiny (|C| < 1e-3), so this is "exact median, with a sliver of
  wrong answers right next to ties".
"""
import numpy as np

# Boundary offsets d*/r, measured by bisecting the model's decision boundary
# across the tie plane x_i = x_j (median over angles; see EXPLANATION.md).
C = {
    ((0, 1), "k_above"): 0.00034,
    ((0, 1), "k_below"): 0.00032,
    ((0, 2), "k_above"): 0.00064,
    ((0, 2), "k_below"): -0.00084,
    ((1, 2), "k_above"): 0.00069,
    ((1, 2), "k_below"): 0.00078,
}


def predict(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(x), dtype=int)
    for n, v in enumerate(x):
        order = np.argsort(v)
        med = order[1]
        # neighbour of the median that is closest in value
        lo, hi = order[0], order[2]
        nb = lo if v[med] - v[lo] < v[hi] - v[med] else hi
        i, j = sorted((int(med), int(nb)))
        k = 3 - i - j
        s = 0.5 * (v[i] + v[j])
        case = "k_above" if v[k] > s else "k_below"
        T = C[((i, j), case)] * np.hypot(s, v[k])
        d = v[i] - v[j] - T
        # median of the pair: if k is above, the median is the larger of i/j
        if case == "k_above":
            out[n] = i if d > 0 else j
        else:
            out[n] = j if d > 0 else i
    return out
