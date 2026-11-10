# Mechanistic explanation

## What the model mostly computes

For three distinct inputs, the dominant rule is simple: return the index of
the input between the other two in numerical order. A behavioral twin can
therefore sort each row and take the index at rank one. The model is not an
exact comparator, however. Its disagreements are concentrated near order
boundaries, where two inputs are almost equal. In those thin bands, small
differences in the recurrent features and output scores can make it choose
one of the tied pair's positions instead of the true middle position. The
network has no explicit tie-breaking or tolerance rule; its mistakes are
irregular, including which pair is close and which position it chooses.

## What the hidden units encode

There is not a clean one-unit-per-comparison decomposition in these weights.
The most defensible unit-level interpretation is at the first step. Since
`h_1[j] = relu(w_ih[j] * x_1)`, units with positive input weights respond to
positive `x_1`, and units with negative input weights respond to negative
`x_1`:

| Units | `w_ih` | First-step response |
|---|---:|---|
| 0, 2, 3, 5, 7 | positive | scaled positive part of `x_1` |
| 1, 4, 6 | negative | scaled negative part of `x_1` |

The values are directly from `weights.npz`: `w_ih` is
`[6.5779443, -5.5214934, 6.15469, 16.20787, -6.355015, 1.4493427,
-9.872095, 2.4044752]`. In particular, each active first-step coordinate is
just `abs(w_ih[j]) * abs(x_1)` on its input's sign and zero on the other sign.

At steps 2 and 3, each coordinate is a rectified mixture of the current
input and all eight previous coordinates. For example, the recurrence for
unit `j` is `relu(w_ih[j] * x_t + sum_i w_hh[j,i] * h_(t-1)[i])` (the rows
of `w_hh` select the new unit). Thus the units carry a distributed summary
of earlier positive and negative magnitudes, with clipping at zero. The
recurrence matrix is dense and has mixed signs, so after step one the
positive/negative grouping is no longer a complete semantic description.
Unit 7 is especially unusual: its row has large inhibitory entries
`w_hh[7,6] = -24.73504` and `w_hh[7,7] = -41.89989`, so its response is
strongly suppressed by prior activity in units 6 and 7.

The readout is also distributed. For final state `h_3`, the three scores are
the dot products with these rows of `w_out`:

| Position | Readout weights on units 0..7 |
|---|---|
| 0 | `[3.6678, 3.4526, -6.9335, 3.0703, -7.2546, 0.0786, 5.3208, -0.6825]` |
| 1 | `[-8.2577, -5.3680, 1.4656, 4.7164, 4.5476, -2.9911, 2.6576, 1.4047]` |
| 2 | `[2.2591, 0.3949, 3.1955, -12.2053, -0.8731, 2.1314, -7.3648, -0.2170]` |

These are rounded for display; every displayed value comes from the
corresponding row in `w_out`. The argmax selects the answer. There is no
single hidden coordinate that directly means “the median”: the output
weights combine the recurrent summary to favor the middle position.

## Failures and the part I cannot explain

The observed errors have a consistent broad shape: one pair is very close,
and the predicted position is often one of that pair. The exceptional cases
are not described by one exact relative-gap cutoff. Some very close pairs
are still answered correctly, and the boundary depends on the third input,
the common value of the pair, and their positions in the sequence. That
dependence is plausible in a dense ReLU recurrence, where different
activation patterns yield different local score margins.

I cannot give a faithful human-readable meaning for each of the eight
post-first-step units, nor a closed-form list of all exceptional regions.
The recurrent state is a distributed, piecewise-linear encoding, and the
weights do not expose a simple symbolic comparison algorithm. Accordingly,
`twin.py` implements the dominant order-statistic rule only. It does not
copy the matrices or approximate their activation regions.

## Constants and derivation

The twin contains no fitted numerical constants: sorting is the rule. For
the accuracy estimate below I use an effective absolute near-tie width
`delta = 3.4e-4` in an input difference. This is a coarse local boundary
scale, not an exact cutoff. For a fixed ReLU active set, the state Jacobian
obeys `J_t = D_t * (diag(w_ih) * e_t.T + w_hh @ J_(t-1))`, with `J_0 = 0`,
where `D_t` is the diagonal mask of active units. The logit Jacobian is
`G = w_out @ J_3`. For positions `a,b` and pair-gap direction
`v = (e_i-e_j)/2`, the local score-gap slope is
`(G[a]-G[b]) @ v`; dividing a score margin by this slope expresses that
margin as an input-gap width. These expressions use the matrices above and
in `weights.npz`. The resulting widths vary with the active set and third
input; `3.4e-4` is one effective scale for the simplified band model, not a
claim that all slices have this width. This is the main approximation in
the estimate.

## Accuracy estimate

For independent standard Gaussians, a pair difference `X_i - X_j` is
`N(0, 2)`. Thus the probability that a given pair lies in the band
`|X_i - X_j| < delta` is `erf(delta / 2)`. There are three pairs, so the
union probability is approximately
`3 * erf(3.4e-4 / 2) = 0.000575`. Intersections of two such bands are
second order in `delta` and negligible at the precision shown. Treating
this union as the effective error region gives estimated accuracy
`1 - 0.000575 = 0.999425`.

This approximation includes the narrow pairwise near-tie bands and assumes
one error per point in their union. It leaves out the active-set-dependent
variation of their widths and any errors away from a near tie. The estimate
is an integration of the simplified error-region description, not a
sampling estimate of model predictions.
