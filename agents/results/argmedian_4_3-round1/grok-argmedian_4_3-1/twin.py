"""Median-position network, rebuilt from the cones in EXPLANATION.md.

The recurrent net is positively homogeneous and piecewise linear.
After the first two steps the hidden state remembers (x1, x2) in one of
ten cones of the ratio r = x2/x1.  In each cone the four pre-activations
at the third step are a fixed linear form of (x1, x2, x3); ReLU keeps the
units that fire, and a fixed readout votes for the median position.

Every number below is a product of the trained weights, derived in
EXPLANATION.md.  This file does not load weights.
"""

import numpy as np

# Cuts of r = x2/x1.  Positive x1, then negative x1.
# Each cut is the zero of one second-step pre-activation.
_BP = (
    0.78820119380026787,
    0.80731132019282603,
    1.2391633332362961,
    1.268168624238154,
)
_BN = (
    0.78639009579660735,
    0.80862322931788466,
    1.2396062431323245,
    1.2654474174824597,
)

# Third-step pre-activation, rows = hidden units, columns = (x1, x2, x3).
# Five cones for x1 > 0, in increasing r, then five for x1 < 0.
_MZ_POS = (
    np.array([
        [0.48618635885475686, -4.9153646339339332, 3.9652628898620605],
        [-1.3476127096627541, -3.6956309466921766, 4.6994881629943848],
        [0.45078384512828656, 3.8137255539146793, -4.7163195610046387],
        [-1.5573225411201277, 5.0177541451279524, -3.9652016162872314],
    ]),
    np.array([
        [-9.3099202942575072, 7.5130695401896617, 3.9652628898620605],
        [-11.707129193578707, 9.4476078169642506, 4.6994881629943848],
        [9.498577361819029, -7.6653150614227741, -4.7163195610046387],
        [6.5883884554182117, -5.3168039101120428, -3.9652016162872314],
    ]),
    np.array([
        [-0.96484926975135166, -2.8237990454173598, 3.9652628898620605],
        [-2.4731414994017706, -1.9903436439063853, 4.6994881629943848],
        [-1.1459274658767458, 5.5198150878154877, -4.7163195610046387],
        [-3.4386292800357836, 7.1034576230212281, -3.9652016162872314],
    ]),
    np.array([
        [8.3450710245061561, -10.336868585607021, 3.9652628898620605],
        [9.2339876941769372, -11.437951460870636, 4.6994881629943848],
        [-10.644504827695775, 13.185130149238262, -4.7163195610046387],
        [-10.027017735453995, 12.420261533133271, -3.9652016162872314],
    ]),
    np.array([
        [1.613319234349027, -5.0286219837789758, 3.9652628898620605],
        [-0.45989513088327938, -3.7939499930975558, 4.6994881629943848],
        [1.3621624466942097, 3.7174087083274117, -4.7163195610046387],
        [-0.50721944384144591, 4.9135324517924346, -3.9652016162872314],
    ]),
)
_MZ_NEG = (
    np.array([
        [1.5486754495604549, -5.0286219837789758, 3.9652628898620605],
        [-0.4808601969331584, -3.7939499930975558, 4.6994881629943848],
        [1.3675908452406444, 3.7174087083274117, -4.7163195610046387],
        [-0.46178242212651055, 4.9135324517924346, -3.9652016162872314],
    ]),
    np.array([
        [-6.580135627711992, 5.3082466018280456, 3.9652628898620605],
        [-9.4755519419641629, 7.6440014677730801, 4.6994881629943848],
        [11.736246606390857, -9.4677214409108501, -4.7163195610046387],
        [9.3053882347330799, -7.5067290813408363, -3.9652016162872314],
    ]),
    np.array([
        [3.4697849495325865, -7.1201875722955492, 3.9652628898620605],
        [1.1523762307996996, -5.4992372958833471, 4.6994881629943848],
        [2.4540277145455272, 2.0113191744266032, -4.7163195610046387],
        [0.94862452653175666, 2.8278289738991589, -3.9652016162872314],
    ]),
    np.array([
        [10.049920577244579, -12.428434174123595, 3.9652628898620605],
        [10.627928172763863, -13.143238763656427, 4.6994881629943848],
        [-9.2822188918453303, 11.479040615337453, -4.7163195610046387],
        [-8.3567637082013224, 10.334558055239995, -3.9652016162872314],
    ]),
    np.array([
        [0.54252613024544072, -4.9153646339339332, 3.9652628898620605],
        [-1.3275227406006476, -3.6956309466921766, 4.6994881629943848],
        [0.41783425682152153, 3.8137255539146793, -4.7163195610046387],
        [-1.6286279308893938, 5.0177541451279524, -3.9652016162872314],
    ]),
)

# Logit rows.  Units 0 and 3 vote for position 2, units 1 and 2 for position 1.
# Position 0 wins only when a unit from each pair is on and their ratio is near 1.
_READ = np.array([
    [-1.6718447208404541, 1.5993161201477051, 1.6174601316452026, -1.6575409173965454],
    [-18.74772834777832, 15.999290466308594, 15.959170341491699, -18.724285125732422],
    [11.430537223815918, -15.74033260345459, -15.697331428527832, 11.476856231689453],
])


def _apply(logits, mask, x, MZ):
    if not np.any(mask):
        return
    hidden = np.maximum(x[mask] @ MZ.T, 0.0)
    logits[mask] = hidden @ _READ.T


def predict(x: np.ndarray) -> np.ndarray:
    """Predicted median positions.  x has shape (n, 3); result has shape (n,)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")
    n = x.shape[0]
    x1 = x[:, 0]
    x2 = x[:, 1]
    logits = np.zeros((n, 3), dtype=np.float64)

    pos = x1 > 0.0
    neg = x1 < 0.0
    zero = ~pos & ~neg

    if np.any(pos):
        r = np.zeros(n, dtype=np.float64)
        r[pos] = x2[pos] / x1[pos]
        b0, b1, b2, b3 = _BP
        _apply(logits, pos & (r < b0), x, _MZ_POS[0])
        _apply(logits, pos & (r >= b0) & (r < b1), x, _MZ_POS[1])
        _apply(logits, pos & (r >= b1) & (r < b2), x, _MZ_POS[2])
        _apply(logits, pos & (r >= b2) & (r < b3), x, _MZ_POS[3])
        _apply(logits, pos & (r >= b3), x, _MZ_POS[4])

    if np.any(neg):
        r = np.zeros(n, dtype=np.float64)
        r[neg] = x2[neg] / x1[neg]
        c0, c1, c2, c3 = _BN
        _apply(logits, neg & (r < c0), x, _MZ_NEG[0])
        _apply(logits, neg & (r >= c0) & (r < c1), x, _MZ_NEG[1])
        _apply(logits, neg & (r >= c1) & (r < c2), x, _MZ_NEG[2])
        _apply(logits, neg & (r >= c2) & (r < c3), x, _MZ_NEG[3])
        _apply(logits, neg & (r >= c3), x, _MZ_NEG[4])

    # x1 = 0.  The x1 columns drop out and both extreme cones reduce to
    # relu(scale * x2) then the third step, which is what the net does.
    _apply(logits, zero & (x2 >= 0.0), x, _MZ_POS[4])
    _apply(logits, zero & (x2 < 0.0), x, _MZ_POS[0])

    return np.argmax(logits, axis=1).astype(np.int64)
