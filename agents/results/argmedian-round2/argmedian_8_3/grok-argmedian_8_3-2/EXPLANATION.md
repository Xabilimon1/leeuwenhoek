# How the network picks the median position

Inputs are written `x0, x1, x2` in time order. The prediction is a position in `{0, 1, 2}`. The median position is the index of the middle value: the unique `i` such that one of the other two inputs is smaller and the other is larger. For three distinct numbers this is the same as the middle index of an argsort.

The network gets this right except on thin bands along the three planes where two inputs are equal. The twin in `twin.py` is exactly that median rule.

## Setup and two facts that shape everything

There are no biases, so every pre-activation is linear in the inputs seen so far. For `λ > 0`,

```
h_t(λx) = λ h_t(x),    logits(λx) = λ logits(x).
```

The argmax is constant on open rays. It is a function of the direction of `x` only. The median position has the same positive-scale invariance (and also translation invariance, which this network does not have). Both decision sets are cones.

With `h_0 = 0`,

```
h_1 = relu(w_ih * x0).
```

Unit `i` turns on at step 1 exactly when `w_ih[i] * x0 > 0`, and its value is `w_ih[i] * x0`. The input weights split into a positive bank and a negative bank:

| unit | `w_ih` | on at step 1 when |
| --- | ---: | --- |
| 0 | 6.577944 | `x0 > 0` |
| 2 | 6.154690 | `x0 > 0` |
| 3 | 16.207870 | `x0 > 0` |
| 5 | 1.449343 | `x0 > 0` |
| 7 | 2.404475 | `x0 > 0` |
| 1 | −5.521493 | `x0 < 0` |
| 4 | −6.355015 | `x0 < 0` |
| 6 | −9.872095 | `x0 < 0` |

So `h_1` is a population code for the single number `x0`: five scaled copies of `x0` when it is positive, three scaled copies of `|x0|` when it is negative. The copies are redundant at step 1. They stop being copies at step 2, because each unit has its own row of `w_hh`.

## Step 2: a piecewise-linear code of the pair `(x0, x1)`

```
pre_2 = w_ih * x1 + w_hh @ h_1,    h_2 = relu(pre_2).
```

**Case `x0 > 0`.** Then `h_1 = x0 * relu(w_ih)`, so

```
pre_2,i = w_ih[i] * x1 + x0 * (w_hh @ relu(w_ih))_i.
```

Unit `i` is on when `x1` is on one side of a fixed multiple of `x0`. The multiple is

```
c_i = -(w_hh @ relu(w_ih))_i / w_ih[i].
```

| unit | on when `x0 > 0` |
| --- | --- |
| 0 | `x1 > 2.270708 x0` |
| 1 | `x1 < 2.258119 x0` |
| 2 | `x1 > −2.652732 x0` |
| 3 | `x1 > 0.714646 x0` |
| 4 | `x1 < −2.651429 x0` |
| 5 | `x1 > 6.774439 x0` |
| 6 | `x1 < 1.018727 x0` |
| 7 | `x1 > 47.270074 x0` |

**Case `x0 < 0`.** Then `h_1 = |x0| * relu(−w_ih)`, and the same calculation with `A = w_hh @ relu(−w_ih)` gives the threshold `c_i = A_i / w_ih[i]`:

| unit | on when `x0 < 0` |
| --- | --- |
| 0 | `x1 > 3.052169 x0` |
| 1 | `x1 < 5.731407 x0` |
| 2 | `x1 > −2.213352 x0` |
| 3 | `x1 > 0.317191 x0` |
| 4 | `x1 < −2.210157 x0` |
| 5 | `x1 > 5.852873 x0` |
| 6 | `x1 < 0.319129 x0` |
| 7 | `x1 > −95.605370 x0` |

(`x0` is negative in this table, so a threshold such as `5.73 x0` is a large negative number.)

The ratio `x1/x0` of two independent standard normals is Cauchy. Most of the mass sits in a few of these intervals. For `x0 > 0` the central pattern, units `{1, 2, 6}` on and the others off, is the interval

```
−2.651429 x0 < x1 < 0.714646 x0,
```

which has Cauchy probability `arctan(0.714646)/π − arctan(−2.651429)/π ≈ 0.583`. The extreme units 5 and 7 (`x1 > 6.77 x0`, `x1 > 47.3 x0`) cover only about 4.7% and 0.67% of ratios.

Several thresholds come in near-complementary pairs, so the bank almost partitions the ratio line:

- `x0 > 0`: units 1 and 0 meet near `2.26 x0`, with a gap `(2.258119, 2.270708) x0` where neither is on. Units 4 and 2 meet near `−2.65 x0`, with a hairline overlap `(−2.652732, −2.651429) x0` where both are on.
- `x0 < 0`: units 6 and 3 meet near `0.318 x0`, with a gap between `0.317191 x0` and `0.319129 x0`. Units 4 and 2 meet near `−2.21 x0`, with a gap between `−2.213352 x0` and `−2.210157 x0`.

Inside one of these ratio cones the active coordinates of `h_2` are linear forms `a x0 + b x1`. Two independent forms already determine `(x0, x1)`, so the hidden state still remembers both numbers; it has not yet thrown away anything the median needs. In the central positive cone the three active forms are exactly the threshold expressions above:

```
h_2[1] = 12.468191 x0 − 5.521493 x1 = −5.521493 (x1 − 2.258119 x0)
h_2[2] = 16.326740 x0 + 6.154690 x1 =  6.154690 (x1 + 2.652732 x0)
h_2[6] = 10.056974 x0 − 9.872095 x1 = −9.872095 (x1 − 1.018727 x0)
```

These are hinges at `2.26`, `−2.65` and `1.02`, which is a redundant linear encoding of `(x0, x1)`, not a test of `x1 ? x0`.

## Step 3 and the readout: one comparison per cone

Step 3 does the same thing with `x2` in place of `x1` and `h_2` in place of `h_1`:

```
h_3 = relu(w_ih * x2 + w_hh @ h_2),    logits = w_out @ h_3.
```

Fixing which units are on at all three steps fixes a cone, and on that cone every hidden unit and every logit is a linear form in `(x0, x1, x2)` with zero intercept. The predicted position is whichever of the three forms is largest. The boundary between two positions is a plane through the origin.

The median partition is also made of planes through the origin. The three tie planes `x0 = x1`, `x0 = x2`, `x1 = x2` all contain the axis `(1, 1, 1)`. Their normals `(1,−1,0)`, `(1,0,−1)`, `(0,1,−1)` have pairwise dots `±1/2`, so they sit 60° apart around that axis and cut direction-space into six equal sectors, one per ordering. Each ordering has Gaussian probability `1/6`. In the sector `x0 < x1 < x2` the median is position 1, and the only face that separates it from a different median is one of those tie planes.

What the network does, cone by cone, is line one logit-difference up with that face. The dominant cone (positive `x0`, step-2 units `{1,2,6}`, step-3 units `{0,4,5,6}`) is the one just described. Masking the recurrence to those units and multiplying by `w_out` produces

```
L0 = −52.664140 x0 +   2.533930 x1 +  17.816358 x2
L1 = −21.676456 x0 + 140.839230 x1 − 113.790358 x2
L2 = −21.585735 x0 −  69.387337 x1 +  96.204169 x2
```

and the comparison that this cone actually uses is

```
L1 − L2 = −0.090721 x0 + 210.226568 x1 − 209.994527 x2.
```

The angle between this normal and `(0, 1, −1)` is `0.03615°`. Setting `L1 = L2` and solving gives the shifted tie

```
x1 − x2 = 0.00043154 x0 − 0.00110376 x2.
```

A pure median test would have been `x1 − x2 = 0`. The learned test is that plane, tilted: the coefficients of `x1` and `x2` disagree by about one part in a thousand, and `x0` leaks in with coefficient `0.09` against `210`. The same pattern repeats on the other tie planes. In every cone that carries appreciable Gaussian mass and that meets a tie plane, one logit difference is within a fraction of a degree of the corresponding coordinate difference, and the third logit stays below both near that face. Crossing the face swaps which of the two positions is the median; the network swaps a fraction of a degree later, or earlier.

The readout weights that build these scores are large and class-specific. Column `j` of `w_out` is the vote of hidden unit `j` (rows are positions 0, 1, 2):

```
unit 0:  +3.67, −8.26, +2.26     pulls toward position 0, against 1
unit 1:  +3.45, −5.37, +0.39     pulls toward position 0, against 1
unit 2:  −6.93, +1.47, +3.20     pulls toward position 2, against 0
unit 3:  +3.07, +4.72, −12.21    pulls toward position 1, against 2
unit 4:  −7.25, +4.55, −0.87     pulls toward position 1, against 0
unit 5:  +0.08, −2.99, +2.13     pulls toward position 2, against 1
unit 6:  +5.32, +2.66, −7.36     pulls toward position 0, against 2
unit 7:  −0.68, +1.40, −0.22     weak pull toward position 1
```

Units 0, 1 and 6 are the position-0 voters, units 3 and 4 the position-1 voters, units 2 and 5 the position-2 voters. Which of them are on at step 3 selects the pair of positions still in the running; their magnitudes, which still depend linearly on all three inputs, decide the winner.

## Where the answer is wrong

The wrong set is a band along each tie plane, on one side. Between the true plane `x_i = x_j` and the network's nearby plane, the argmax still names the position that was the median on the other side of the tie.

The width is not a single angle. Parameterize each tie plane by the foot angle `θ` around its great circle (the orthonormal frame with pole `(1,1,1)/√3` and normal `(e_i − e_j)/√2`). At foot `θ`, walk along the normal geodesic and record the angle `W(θ)` at which the prediction returns to the true median. `W` is a few hundredths of a degree on most of the circle and peaks near half a degree:

| plane | mean width on the positive side | mean width on the negative side | largest width |
| --- | ---: | ---: | ---: |
| `x0 = x1` | 0.0095° | 0.0022° | 0.51° |
| `x0 = x2` | 0.0099° | 0.0165° | 0.44° |
| `x1 = x2` | 0.0209° | 0.0054° | 0.21° |

The positive normal is `(e_i − e_j)/√2`, so the positive side is the side `x_i > x_j`. About 85% of the `x0 = x1` circle is wrong on the positive side, about 56% of the `x0 = x2` circle on the negative side, about 77% of the `x1 = x2` circle on the positive side. A few percent of feet have no band wider than `10⁻⁵` rad, and a few tenths of a percent are wrong (very slightly) on both sides. The sign of the shift changes as the third input swings around the tied pair, which is why there is no single global rule of the form "prefer the larger value when they are close".

A concrete point on the `x1 ≈ x2` band: `x = (0.358, 0.721175, 0.721106)`. The order is `x0 < x2 < x1`, so the median is position 2, but only by `7·10⁻⁵`. The network's logits put position 1 ahead by a small margin and answer 1. Relative to the size of the inputs, the gap is well inside the `0.001`-scale shift of the dominant-cone formula above.

Unit 7 barely participates. Its recurrent row is dominated by `w_hh[7,7] = −41.90` and `w_hh[7,6] = −24.74`, so after step 1 it turns back on only if the new input is tens of times the stored scale (the thresholds `47 x0` and `−96 x0` above). Its readout column has entries of size about 1, against 5–12 for the other units. On a standard Gaussian draw it is on at step 3 only a few percent of the time.

## Accuracy from the measure of the bands

A standard normal vector in `R^3` has direction uniform on the sphere. For a fixed plane through the origin with unit normal `n`, the height `Z = u · n` of a uniform unit vector is uniform on `[−1, 1]` and independent of the direction of the projection of `u` onto the plane. The geodesic angle `α` off the plane satisfies `Z = sin α`.

If the bad set at foot `θ` is the geodesic interval of length `W_+(θ)` on the positive side and `W_−(θ)` on the negative side, the Gaussian probability of that plane's band is

```
(1 / 2π) ∫_0^{2π} [ sin(W_+(θ)) + sin(W_−(θ)) ] / 2  dθ.
```

`W` was computed from the weights at 720 equally spaced feet on each great circle, by bisection on the angle (probe `10⁻⁵` rad, then 40 bisections). The three integrals are

```
P(x0, x1) = 1.01855e-4
P(x0, x2) = 2.30585e-4
P(x1, x2) = 2.29805e-4
```

and their sum is `5.6225e-4`. The accuracy on i.i.d. standard normal inputs is therefore

```
1 − 5.6225e-4 = 0.999438.
```

Included: the whole circle of each tie plane, both sides, every ratio of the free coordinate. Left out, and negligible next to `10⁻⁴`: the overlap of two bands, which is a neighborhood of the axis `(1,1,1)` of width equal to the band angles and has probability of order `10⁻⁷`; and any error component that does not touch the three tie planes. Inside a cone the logits are linear, so the decision changes only across a logit-tie plane. Every high-mass cone that stays in one ordering sector predicts that sector's median the whole way across; the cones that meet a sector boundary contribute the bands above. I did not exhaust the very small cones, and the estimate does not add a term for them.

## What is not explained

The hinge ratios `0.318, 0.715, 1.019, 2.26, 2.65, 5.8, 6.77, 47, 96` are derived above from `w_hh` and `w_ih`, but they are not a simple sequence. They are wherever training parked the kinks so that each resulting cone would admit an almost exact linear comparison. I cannot point at a hand-designed circuit, with hinges at `x1 = x0` or at a short list of integers, that these weights implement exactly.

Individual hidden units are not `relu(x_a − x_b)`. Fitting the step-3 state to the six pairwise ReLUs leaves a few percent of variance, which is exactly the difference between hinges at ratio 1 and hinges at the ratios above. The population jointly carries `(x0, x1)` and then the comparison with `x2`; the labels live in the readout geometry, not in one unit each.

There is also no single closed form for `W(θ)` beyond "the angle from the tie plane to the zero of the active cone's logit difference, clipped to the cone". The dominant cone's difference is the explicit linear form `L1 − L2` above; other cones have their own forms, with the same shape (one coordinate difference, plus a relative residual of order `10⁻³` to `10⁻²`) and with worse alignment on the small cones. Sharing 96 weights across all of those cones is why the cancellation is close and not perfect, and why the band widens on some arcs of the circle.
