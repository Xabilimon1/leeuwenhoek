# How the median network works

The network reads three scalars one at a time and returns the index of the middle one. It is a bias-free ReLU RNN,

```
h_0 = 0
h_t = relu(w_ih * x_t + w_hh @ h_{t-1})
logits = w_out @ h_3
```

with shapes `w_ih` (4,), `w_hh` (4, 4), `w_out` (3, 4). The map is positively homogeneous: scaling the whole input by a positive constant scales every pre-activation and every logit by the same constant, so the argmax depends only on direction. There are no bias terms, so every region below is a cone through the origin.

On standard Gaussian inputs it matches the median index with probability 0.9845. The misses are thin slabs along the planes where two inputs are equal. They are produced by the readout, not by a failure to store the inputs.

## Weights, and the sign symmetry

From `weights.npz`:

```
w_ih = [ 3.965263,  4.699488, -4.716320, -3.965202 ]

w_hh = [[ 1.338687, -2.199573,  2.635198, -1.894751],
        [ 1.927741, -2.433872,  2.786757, -2.382630],
        [-2.387666,  2.805652, -2.433898,  1.933146],
        [-1.893123,  2.642897, -2.191234,  1.340866]]

w_out = [[ -1.671845,   1.599316,   1.617460,  -1.657541],
         [-18.747728,  15.999290,  15.959170, -18.724285],
         [ 11.430537, -15.740333, -15.697331,  11.476856]]
```

Negating every input swaps hidden units `0↔3` and `1↔2`. Write `P` for that permutation matrix. Then `P w_ih + w_ih` is at most `0.017`, `P w_hh P - w_hh` at most `0.019`, and `w_out - w_out P` at most `0.046`, against entries of size 1 to 20. The median index is invariant under negation, and the net is the same circuit twice, once for each sign, up to that residual. Units 0 and 3 are one channel; units 1 and 2 are the other. The two channels inside a sign are not redundant: they are the same signal at two different gains, and the gain ratio is the whole computation.

## Step 1: write the sign of x1 on a pair of units

`h_0 = 0`, so `h_1 = relu(w_ih * x_1)`.

- If `x1 > 0`, units 0 and 1 turn on: `h1 = x1 * (3.965263, 4.699488, 0, 0)`.
- If `x1 < 0`, units 2 and 3 turn on: `h1 = |x1| * (0, 0, 4.716320, 3.965202)`.

Call the two gains `α ≈ 3.9653` (units 0 and 3) and `β ≈ 4.70` (units 1 and 2). At this step the two live units are copies of `x1`. Two copies are kept because the next step mixes them with different rows of `w_hh`, and that mix is what separates `x1` from `x2`.

## Step 2: ten cones, each a linear memory of (x1, x2)

`h2 = relu(w_ih * x2 + w_hh @ h1)`. Inside a sign of `x1`, `h1` is linear in `x1`, so each pre-activation is linear in `(x1, x2)` and its zero set is a ray `x2 = c * x1`.

For `x1 > 0` the four zeros are

| unit | on when `r = x2/x1` is | ray `r` |
| --- | --- | --- |
| 2 | below | 0.788201 |
| 1 | above | 0.807311 |
| 3 | below | 1.239163 |
| 0 | above | 1.268169 |

Those four rays cut the half-plane into five cones. The live units are:

| cone | `r = x2/x1` | live units |
| --- | --- | --- |
| x2 much smaller | `r < 0.788201` | 2, 3 |
| narrow | `0.788201 ≤ r < 0.807311` | 3 only |
| comparable | `0.807311 ≤ r < 1.239163` | 1 and 3 |
| narrow | `1.239163 ≤ r < 1.268169` | 1 only |
| x2 much larger | `r ≥ 1.268169` | 0, 1 |

For `x1 < 0` the same arithmetic with the other half of `h1` gives rays `0.786390, 0.808623, 1.239606, 1.265447` and live pairs `(0,1)`, `(0,)`, `(0,2)`, `(2,)`, `(2,3)`. The `x1 = 0` line is the shared boundary: `h1 = 0` and `h2 = relu(w_ih * x2)`, which is the limit of the extreme cones.

Each ray is `- (w_hh @ h1c)_i / w_ih_i`, where `h1c` is `(w_ih[0], w_ih[1], 0, 0)` or `(0, 0, w_ih[2], w_ih[3])`.

On every open cone the live coordinates of `h2` are an invertible linear image of `(x1, x2)`. Both numbers are still in the state. For the cone `x1 > 0`, `r ≥ 1.268169`,

```
h2_0 = -5.028622 x1 + 3.965263 x2
h2_1 = -3.793950 x1 + 4.699488 x2
```

with the other two coordinates zero. The coefficient `-5.028622` is row 0 of `w_hh` dotted with `(w_ih[0], w_ih[1], 0, 0)`; `-3.793950` is the same with row 1. The 2×2 map has determinant `-8.588`, so `(x1, x2)` can be read back.

The two narrow cones are the exception. Only one unit survives, the state drops from two degrees of freedom to one, and `(x1, x2)` is no longer recoverable. Each of those cones is about 0.7° of angle (a relative gap of about 0.02 in the ratio). They are where the two comparison thresholds, 0.79 and 1.25, fail to overlap.

The comparable cone is the overlap: one unit from each channel stays on, and again the pair determines `(x1, x2)`. Its angular width is about 12°.

## Step 3: compare x3 with the stored pair

`h3 = relu(w_ih * x3 + w_hh @ h2)`. On a cone, `h2` is linear in `(x1, x2)`, so each of the four pre-activations is a linear form `A_i(x1, x2) + w_ih_i * x3`. Unit `i` turns on one side of the plane

```
x3 = -A_i / w_ih_i
```

The third column of every map in `twin.py` is `w_ih` for that reason. The first two columns are `w_hh @ M2`, where `M2` is the `(x1, x2)` map of `h2` with dead units zeroed. That is the whole list of constants in the twin: one 4×3 map per cone, plus the readout.

For the cone `x1 > 0`, `x2 ≥ 1.268 x1` the four planes are

```
unit 0 on above  x3 = -0.406863 x1 + 1.268169 x2
unit 1 on above  x3 =  0.097861 x1 + 0.807311 x2
unit 2 on below  x3 =  0.288819 x1 + 0.788201 x2
unit 3 on below  x3 = -0.127918 x1 + 1.239163 x2
```

They sit near `x3 = x2` and `x3 = x1`, tilted. The same picture, with the roles of the two channels exchanged, holds in every other two-unit cone.

## Readout: a ratio test, and why the gains match it

`w_out` has a simple voting pattern. One hidden unit at a time:

| live unit | logit triple (class 0, 1, 2) | winner |
| --- | --- | --- |
| 0 | `h * (-1.672, -18.748, 11.431)` | 2 |
| 1 | `h * ( 1.599,  15.999, -15.740)` | 1 |
| 2 | `h * ( 1.617,  15.959, -15.697)` | 1 |
| 3 | `h * (-1.658, -18.724, 11.477)` | 2 |

Units 0 and 3 vote for "the median is at the last position". Units 1 and 2 vote for "the median is at the middle position". Nothing votes for position 0 by itself. Position 0 wins when one unit from each campaign is on and the two activities are comparable.

With only units 0 and 1 on, `r = h0/h1` and the logit ties fall at

```
L0 = L1 at r = 0.843293
L0 = L2 at r = 1.323397
L1 = L2 at r = 1.051738
```

so the argmax is class 1 for `r < 0.843293`, class 0 for `0.843293 ≤ r ≤ 1.323397`, and class 2 for `r > 1.323397`. With only units 3 and 2 on, `s = h3/h2` gives the same shape with thresholds `0.840331` and `1.318278`. (Ties in the logits go to the smaller index, which is what `argmax` does; on continuous inputs they have measure zero.)

The input gains are built to sit on the lower threshold:

```
w_ih[0] / w_ih[1] = 3.965263 / 4.699488 = 0.843765
|w_ih[3] / w_ih[2]| = 3.965202 / 4.716320 = 0.840741
```

against `0.843293` and `0.840331`. The gaps are `4.7e-4` and `4.1e-4`.

Consequence, on the cone where `x2` is well above `x1 > 0`. For `x3` very negative, units 2 and 3 are the ones on (their input weights are negative). Their ratio tends to `|w_ih[3]/w_ih[2]| = 0.840741`, which is just inside the class-0 band. The entire lower tail is therefore class 0, not class 1. As `x3` rises, unit 2 (gain 4.716) falls faster than unit 3 (gain 3.965), the ratio climbs, and it leaves the class-0 band at the upper threshold `1.318278`. Solving `h3 = 1.318278 h2` on this cone gives the plane that the twin implements; for the ray `(x1, x2) = (1, 2)` it sits at `x3 = 1.011`. The true change from "median at index 0" to "median at index 2" is the plane `x3 = x1`. The network's cut is parallel and a little past it.

Sweeping `x3` further, the live pair changes. Past `x2` the state is units 0 and 1, and their ratio comes down from above the class-2 threshold, through the class-0 band, and into class 1. On the ray `(1, 2)` that class-0 crossing occupies `1.961 < x3 < 2.031`. The true median index there is 2 and then 1. It is never 0, because `x1 = 1` is not between `x3 ≈ 2` and `x2 = 2`. The band is a false "median at the first position".

So along a generic ray with `x1 < x2` the prediction is:

1. class 0 for `x3` below a plane near `x3 = x1` (correct, apart from a shift of the cut);
2. class 2 between that plane and a plane just short of `x3 = x2` (correct);
3. class 0 on a thin slab around `x3 = x2` (wrong);
4. class 1 beyond that slab (correct).

The mirror, `x2 < x1`, puts the long class-0 ray on the high side of `x1` and the false slab around `x2`. The comparable cones do the same job with the mixed pairs `(1, 3)` or `(0, 2)`; the planes are more tilted, so the false slab and the shifted cut are fatter. The one-unit cones cannot place two cuts, and the decision there is a coarser function of a single linear form.

The false slab is structural. A single ratio can enter the middle class only by passing through an interval. The transition from "median is x3" to "median is x2" has to cross the class-0 band. Training put that band on top of the true class-0 region on one side (the long tail beyond `x1`, by parking the gain ratio on the lower threshold) and could not avoid a copy of it at the other transition.

There is a further, irrelevant crossing. Because `w_ih[0]/w_ih[1]` is a hair above `0.843293` rather than a hair below, a pure class-1 state at enormous `x3` eventually tips into class 0. On `(x1, x2) = (1, 2)` that tip is near `x3 ≈ 750`. Under a standard normal that tail is zero for every purpose here.

## Where the median is wrong

The true median index, for distinct inputs, is the middle order statistic:

- index 0 if `x1` is between `x2` and `x3`,
- index 1 if `x2` is between `x1` and `x3`,
- index 2 if `x3` is between `x1` and `x2`.

The network disagrees on two families of regions, both unions of slabs between planes through the origin:

1. The shifted cut near `x3 = x1`. On `(1, 2)` this is the interval `(1, 1.011)`, where the truth is index 2 and the network still says 0.
2. The false class-0 slab near `x3 = x2`. On `(1, 2)` this is `(1.961, 2.031)`, truth 2 then 1, network 0.

Both widths scale with the inputs, because every boundary is linear and homogeneous. Conditional on `(x1, x2)`, the probability that a standard normal `x3` lands in the slab is large when `x1` and `x2` are close to each other and close to 0, and small when they are well separated. The comparable cones (ratio within about 0.81 to 1.24, roughly 7% of directions) contribute about a fifth of the total error, around three times their share of directions. The one-unit cones contribute about 0.00024 of the error probability altogether.

Negating the input permutes the units and preserves the median index, so the negative cones repeat the same two slabs. The symmetry residual of a few hundredths moves the positive and negative cuts by slightly different amounts; that is already included in the separate cones.

## Accuracy

Conditional on `(x1, x2)`, the set of `x3` on which the argmax differs from the middle order statistic is a union of intervals. Its Gaussian measure is a sum of differences of `Φ`. The expectation of that measure under `(x1, x2) ~ N(0, I_2)` is the accuracy's complement.

The integral is taken in polar coordinates. The radial part, after `u = ρ²/2`, is a Gauss–Laguerre rule. The angular part is a Gauss–Legendre rule on each activation cone, with the cones split along `x1 = x2` so that the true-median formula does not change inside a panel. Product rules at angular/radial orders (8, 32), (12, 40) and an unsplit order-16 rule all give an error probability in `0.0154` to `0.0156`, i.e. accuracy `0.9844` to `0.9846`. The reported value is `0.9845`.

## What the twin is

`twin.predict` is this cone circuit, not a copy of the recurrence. For each row it picks the cone from `x2/x1`, applies that cone's 4×3 map, rectifies, and multiplies by the readout. Every entry of those maps is an entry of `w_hh @ M2` or of `w_ih`, with `M2` the step-2 map described above. Dead units are zeroed before the multiply, which is what ReLU has already done inside the cone.

Checked against `load_model.predict` on 80,000 standard normal triples plus the cone boundaries, the ratio rays, and the `x1 = 0` line: no disagreements.

## What is not explained

The weights are a trained point, not a closed form. The design they approximate is visible: two gains whose ratio equals the readout's lower threshold, two-unit memories of `(x1, x2)` on cones, and a three-way ratio test. The leftovers are small and numerical, and I cannot derive them from a simpler exact circuit:

- the `4e-4` miss between each gain ratio and its threshold,
- the tilt that puts the main cut at `x3 = 1.011 x1 - 0.006 x2` rather than at `x3 = x1`, on the `(1, 2)` ray,
- the breaking of the sign-swap symmetry, at most a few hundredths on the entries,
- a near-null direction of `w_hh` (smallest singular value about `0.001`) lined up roughly with `|w_ih|`.

The one-unit cones and the corners where three units of `h3` are on at once are included in the piecewise map, but they are boundary cases of the same arrangement rather than a separate mechanism. The enormous-`x3` tip from class 1 back to class 0 is a side effect of the gain ratio sitting just above the threshold; it does not affect the Gaussian accuracy.
