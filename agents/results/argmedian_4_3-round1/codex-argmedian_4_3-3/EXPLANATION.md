# How the small recurrent classifier behaves

## What the weights say

At the first step, the four activations are just positive and negative
rectifiers of the current scalar. The input coefficients are
`(3.9653, 4.6995, -4.7163, -3.9652)`. Thus units 0–1 are the positive-side
pair, and units 2–3 are the negative-side pair. Within either pair the
slopes are almost matched in reverse order. There is no offset, so all four
units are zero at input zero. This is a signed magnitude code, not four
independent semantic features.

The recurrent matrix has a similar near-reversal symmetry: row 0 is close to
the reverse of row 3, and row 1 is close to the reverse of row 2. Its entries
are around 1.3–2.8 in magnitude. At steps 2 and 3 it mixes the previous
signed-magnitude code with the new input's signed-magnitude code, then clips
negative coordinates to zero. The two positive-side and two negative-side
units therefore act as a redundant, piecewise-linear state for comparing
the values seen so far. In the last state, the units collectively separate
which input position is the middle one; they do not appear to be a clean
one-unit-per-position representation.

The readout rows make that separation explicit. For hidden coordinates
0–3, the output coefficients are approximately

| output position | coefficients |
|---|---|
| 0 | `(-1.67, 1.60, 1.62, -1.66)` |
| 1 | `(-18.75, 16.00, 15.96, -18.72)` |
| 2 | `(11.43, -15.74, -15.70, 11.48)` |

Position 1 has the largest readout scale; position 2 strongly favors the
outer pair of coordinates and opposes the inner pair, while position 0 has
the opposite preference to position 2 at roughly one tenth its scale. The
argmax of these three scores is the answer. This explains the broad behavior
as a learned ordering / median computation, but I cannot give a faithful
symbolic interpretation of every hidden coordinate and every activation
boundary from these weights. In particular, the small asymmetries in the
recurrent matrix and readout matter at close comparisons.

## Errors and their limits

The common rule is to return the position whose value is between the other
two. The failures I could characterize are concentrated near ordering
boundaries: when two values are close, small differences between the nearly
reversed recurrent paths can change which readout score wins. The wrong
answer is often position 0, consistent with its comparatively small readout
scale and the asymmetry of the scores. There are also less frequent failures
when values are well separated, so “near tie” is a useful description of the
dominant failure mode, not a complete decision rule. I do not know a simple
closed-form condition for those residual failures; I have not hidden the
network's full piecewise-linear computation in the twin.

The coefficients above are rounded directly from `weights.npz` to two
decimals for the readout and four for the input coefficients. The behavioral
description is intentionally coarser: the recurrent clipping and mixed
coefficients do not reduce cleanly to an exact comparison algorithm.

The only numerical scale used in the accuracy estimate is a first-order
boundary-width heuristic, not a hard threshold implemented by the network.
The largest mirror mismatch in the position-1 readout is about
`|16.00 - 15.96| = 0.04`; dividing by the smallest input slope,
`3.9653`, gives `g ≈ 0.010`. This estimates how far apart two inputs need to
be before that small score imbalance is overcome. The recurrence can change
the effective slope, which is why this is only a rough width estimate.
