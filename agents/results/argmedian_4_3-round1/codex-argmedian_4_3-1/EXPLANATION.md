# How the small recurrent model behaves

## What the weights suggest

The four input weights are

```
[3.965263, 4.699488, -4.716320, -3.965202]
```

They come in opposite-sign pairs. Averaging the near-matched magnitudes gives
3.965232 for units 0/3 and 4.707904 for units 1/2. At the first step, a
positive input therefore activates units 0 and 1; a negative input activates
units 2 and 3. ReLU makes the two pairs act like separate positive-value and
negative-value channels. The small mismatch between the paired magnitudes is
one source of asymmetry.

The recurrent matrix also has a conspicuous sign pattern. Its entries range
from about -2.434 to 2.787, with positive and negative couplings between both
halves of the state. The recurrence mixes the earlier channels at each step;
it does not simply preserve a sign-pair activation unchanged. Thus the state
at step 3 is not a stored copy of the first or second input: it is a mixed
summary of the successive sign and magnitude contrasts. The
four coordinates do not correspond one-for-one to the three input positions.
I can describe their polarity grouping, but I cannot assign each coordinate a
precise human-readable rank or comparison at every step.

The readout helps explain why predictions are usually useful. Its rows are

```
class 0: [-1.672,  1.599,  1.617, -1.658]
class 1: [-18.748, 15.999, 15.959, -18.724]
class 2: [11.431, -15.740, -15.697, 11.477]
```

Class 1 and class 2 are dominated by almost opposite contrasts: class 1
rewards units 1/2 over 0/3, while class 2 rewards 0/3 over 1/2. Class 0 has
the same contrast direction as class 1 but much smaller coefficients, so it
acts as a residual choice when the other two readout scores are not favored.
The readout turns the recurrent comparison summary into a position label.

## Where the explanation stops

The recurrent matrix and readout are only approximately symmetric. Small
differences between paired coefficients can tip the logits when two inputs
are nearly equal, so the model occasionally selects an endpoint instead of
the middle-valued position. Such close comparisons are the main visible
failure mode. The model is homogeneous (scaling all three inputs by a
positive factor scales all hidden states and logits by that factor), so the
relevant closeness is relative to the overall input magnitude.

I cannot give an exact, simple boundary for every wrong answer from this
weight structure. In particular, the precise region depends on all three
recurrent updates and on small logit differences; it is not just “always
choose the wrong item when a particular gap is small.” There are also a few
rare errors away from the closest comparisons. The twin consequently uses
the dominant rule—return the median position—and does not imitate the
remaining finely placed decision boundaries.

## Hand-built rule and accuracy estimate

`twin.py` sorts each three-number row and returns the index in the middle.
This is the ordinary median-position rule. It is not a transcription of the
matrices or of their piecewise-linear regions.

For a rough accuracy estimate, model the errors as a thin neighborhood of the
three pair-equality planes `x_i = x_j`. The output-side asymmetry scale can be
read from the paired class-2 coefficients: `|11.476856 - 11.430537| / 11.453697
≈ 0.00404`. Taking a two-sided displacement gives an effective relative
thickness `δ = 0.00809` of the largest input magnitude. For one pair, write
`D = X_i-X_j` and `M = (X_i+X_j)/2`. For independent standard Gaussians, `D`
and `M` are independent, with standard deviations `sqrt(2)` and `1/sqrt(2)`.
Conditioning on `M` and the third input `Z`, the first-order probability of
`|D| < δ max(|M|,|Z|)` is
`δ/sqrt(pi) * E[max(|M|,|Z|)]`. The expectation is the one-dimensional
integral `integral_0^infinity (1 - erf(t) * erf(t/sqrt(2))) dt = 0.9772`.
Summing the three pair bands gives mass `3 * δ/sqrt(pi) * 0.9772 = 0.0134`;
intersections are second order in δ. This gives accuracy estimate `0.987`.
This is a tube approximation, not an exact integration of the network's
piecewise-linear regions. It includes the close-pair error neighborhood and
leaves out the rare errors farther from a tie and the portions of the
neighborhood where the model still predicts correctly. Those omissions and
the approximate conversion from readout asymmetry to boundary displacement
are the main sources of uncertainty in the estimate.
