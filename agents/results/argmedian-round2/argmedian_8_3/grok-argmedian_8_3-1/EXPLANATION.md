# How the argmedian network works

The network returns the **position of the median** of `(x0, x1, x2)`. It is wrong only when two inputs are almost tied for that role. The mistake is a fraction of a degree of tilt in an otherwise exact coordinate comparison, left over after large opposing readout terms cancel.

There are no biases, so the whole map is positively homogeneous: replacing `x` by `k x` with `k > 0` multiplies every pre-activation and every logit by `k` and does not change the argmax. The median position is invariant under the same scaling, so both the prediction and the truth are constant on rays. Error sets are cones. Under the standard Gaussian their probability is solid angle divided by `4π`.

Notation matches `load_model.py`:

```
h0 = 0
h_t = relu(w_ih * x_{t-1} + w_hh @ h_{t-1})    t = 1, 2, 3
logits = w_out @ h3
```

`w_ih` has shape `(8,)`, `w_hh` is `(8, 8)`, `w_out` is `(3, 8)`. Indices of inputs and of logit classes are `0, 1, 2`.

## Step 1: a sign split of `x0`

`h0 = 0`, so `h1 = relu(w_ih * x0)`. The input weights change sign across the eight units:

| unit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `w_ih` | +6.577944 | −5.521493 | +6.154690 | +16.207870 | −6.355015 | +1.449343 | −9.872095 | +2.404475 |

- If `x0 > 0`, units `{0, 2, 3, 5, 7}` turn on and hold `w_ih[i] * x0`. Units `{1, 4, 6}` stay off.
- If `x0 < 0`, units `{1, 4, 6}` turn on and hold `w_ih[i] * x0` (a positive number). The other five stay off.
- If `x0 = 0`, `h1 = 0`. The later steps still know the first input was zero, because every memory of it is absent.

So step 1 does not mix features. It writes the signed value into one of two disjoint channels, each coordinate scaled by its own gain. Five units carry the positive part, three carry the negative part. That imbalance is real: the negative channel is only units 1, 4 and 6.

## Step 2: ratio gates on `x1 / x0`

`h1` is rank-one in `x0` inside each sign half-space, so each coordinate of the step-2 pre-activation is linear in `(x0, x1)`:

```
pre2 = w_ih * x1 + w_hh @ h1
```

**Positive half-space.** `h1 = relu(w_ih) * x0`, and

```
pre2_i = w_ih[i] * x1 + (w_hh @ relu(w_ih))_i * x0
```

Unit `i` turns on when `pre2_i > 0`, i.e. at a threshold on the ratio `r = x1 / x0`. The coefficient `(w_hh @ relu(w_ih))` is the vector

```
(-14.936592, 12.468191, 16.326740, -11.582883, -16.849869, -9.818484, 10.056974, -113.659721)
```

Dividing by `w_ih` (and flipping the inequality when `w_ih < 0`) gives the gates. A unit is on when `r` is on the indicated side of the threshold:

| unit | on when `x1/x0` … | threshold |
| --- | --- | --- |
| 2 | > | −2.652732 |
| 4 | < | −2.651429 |
| 3 | > | 0.714646 |
| 6 | < | 1.018727 |
| 1 | < | 2.258119 |
| 0 | > | 2.270708 |
| 5 | > | 6.774439 |
| 7 | > | 47.270074 |

**Negative half-space.** `h1 = relu(−w_ih) * |x0|`, so

```
pre2_i = w_ih[i] * x1 + (w_hh @ relu(−w_ih))_i * |x0|
```

with `(w_hh @ relu(−w_ih))` equal to

```
(20.076998, -31.645928, -13.622493, 5.140997, 14.045583, 8.482819, -3.150468, -229.880742)
```

Unit 2 is the template: `w_ih[2] > 0` and the coefficient of `|x0|` is −13.622493, so unit 2 is on when `x1/|x0| > 13.622493 / 6.154690 = 2.213352`. The same division, with the inequality flipped when `w_ih[i] < 0`, gives:

| unit | on when `x1/|x0|` … | threshold |
| --- | --- | --- |
| 1 | < | −5.731407 |
| 5 | > | −5.852873 |
| 0 | > | −3.052169 |
| 6 | < | −0.319129 |
| 3 | > | −0.317191 |
| 4 | < | 2.210157 |
| 2 | > | 2.213352 |
| 7 | > | 95.605370 |

These thresholds are not comparisons of `x1` with `x0`. They are comparisons of `x1` with a **scaled** memory of `x0`. The scale is wrong for a pure difference because `w_hh` does not invert `w_ih`. What the pattern of on-units does record is a coarse thermometer of the ratio: which of these eight cuts the ratio has passed.

Three pairs are worth naming.

- Units 0 and 1, for `x0 > 0`, are almost complements of each other at ratio ≈ 2.26. Both are off only on the gap `(2.258119, 2.270708)`. Under the Gaussian that gap has probability `(arctan(2.270708) − arctan(2.258119)) / (2π) ≈ 3.3×10⁻⁴`.
- Units 2 and 4, for `x0 > 0`, are almost complements at ratio ≈ −2.65, with a thin overlap `(−2.652732, −2.651429)` of probability ≈ `2.6×10⁻⁵` where both are on.
- Units 3 and 6, for `x0 > 0`, overlap on a **wide** band `0.715 < x1/x0 < 1.019` (probability ≈ 0.028). That band is “`x1` is the same order as `x0`”. Outside it, exactly one of the two is on, which is a coarse bit for which of `x0, x1` is larger. The same role on the negative side is played by units 3 and 6 again, now cutting `x1/|x0|` near −0.32, and by units 2 and 4 near +2.21.

Unit 7 is the extreme-ratio detector. Its row of `w_hh` is dominated by the self-weight `w_hh[7,7] = −41.89989` and by `w_hh[7,6] = −24.73504`. For `x0 > 0` the big negative coefficient −113.66 of `x0` in `pre2_7` is mostly that self-weight times the step-1 value `h1_7 = 2.404475 x0`:

```
−41.89989 * 2.404475 x0 ≈ −100.7 x0
```

plus smaller cross terms from units 0, 2, 3 and 5. The input term `2.404475 x1` beats this only when `x1/x0 > 47.27`. The Gaussian probability of that tail with `x0 > 0` is `1/(2π) * (π/2 − arctan(47.27)) ≈ 0.0034`. The readout column of unit 7 is the smallest of the eight (`‖w_out[:,7]‖ ≈ 1.58`, against 3.7–13.4 for the others), so once the ratio is merely large rather than extreme, unit 7 is off and stays off. It does not carry the ordinary median decision.

## Step 3: the same gates, now on `x2` against the code of `(x0, x1)`

`h2` is constant in which units are on inside each ratio strip above, and on that strip it is linear in `(x0, x1)`. Therefore

```
pre3 = w_ih * x2 + w_hh @ h2
```

is linear in `(x0, x1, x2)` inside the strip, and each unit of `h3` turns on across a plane through the origin. The on-pattern of `h3` is a code for where `x2` sits relative to the first two inputs. Counting cells of this arrangement gives 120 masks with `x0 > 0` and 110 with `x0 < 0`. Their cones partition `R³`: the solid angles sum to `4π` to numerical error `< 10⁻⁹` in the fraction.

The masks that actually hold most of the Gaussian mass are few. On 30 000 standard normal inputs the two leading step-2 patterns were

- `x0 > 0` and `−2.65 < x1/x0 < 0.715`: units `{1, 2, 6}` on (about 29%),
- `x0 < 0` and `−2.21 < x1/x0 < 0.317`: units `{0, 3, 4, 5}` on (about 24%).

Inside those, the leading step-3 patterns are the ones in which the median contest is between a single pair of positions (the third input is the unique min or the unique max).

## Readout: votes that cancel down to `x_i − x_j`

`w_out` columns, and the class each unit votes for if it were the only one on:

| unit | column (class 0, 1, 2) | votes for |
| --- | --- | --- |
| 0 | (3.668, −8.258, 2.259) | 0 |
| 1 | (3.453, −5.368, 0.395) | 0 |
| 6 | (5.321, 2.658, −7.365) | 0 |
| 3 | (3.070, 4.716, −12.205) | 1 |
| 4 | (−7.255, 4.548, −0.873) | 1 |
| 7 | (−0.683, 1.405, −0.217) | 1 |
| 2 | (−6.933, 1.466, 3.195) | 2 |
| 5 | (0.079, −2.991, 2.131) | 2 |

Units `{0, 1, 6}` support position 0, units `{3, 4, 7}` support position 1, units `{2, 5}` support position 2. The support alone does not decide the answer. Several units are on at once, their votes are large, and the class is the argmax of the **sum**. Inside one activation cone the sum is a linear form `L x`, with `L` a `3×3` matrix built as

```
pre3_coeff[i, :]  from  w_ih and  w_hh @ (masked step-2 map)
L = w_out @ (pre3_coeff with inactive rows zeroed)
```

The decision `class p beats class q` is the plane `(L[p] − L[q]) · x = 0`.

In every cone that touches a median boundary, that plane is within a fraction of a degree of the coordinate plane `x_p = x_q`. The large pieces cancel, and the leftover is a small multiple of `x_p − x_q` plus a smaller contamination from the third coordinate.

### Worked cancellation

Take the most common positive cell: `x0 > 0`, step-2 units `{1, 2, 6}` on, step-3 units `{0, 4, 5, 6}` on. This is where the median contest is between positions 1 and 2 (position 0 is the extreme). The difference `logit2 − logit1` gets one term per active unit, ` (w_out[2,i] − w_out[1,i]) * pre3_coeff[i] `:

| unit | vote gap `w_out[2]−w_out[1]` | contribution to `(coeff of x0, x1, x2)` |
| --- | --- | --- |
| 0 | +10.516782 | (+130.081, −172.067, +69.179) |
| 4 | −5.420781 | (−99.582, +39.247, +34.449) |
| 5 | +5.122498 | (+34.958, −56.889, +7.424) |
| 6 | −10.022424 | (−65.367, −20.518, +98.942) |
| sum | | **(+0.090721, −210.226568, +209.994527)** |

Each unit contributes tens or hundreds. The sum is

```
logit2 − logit1 = 0.090721 x0 − 210.226568 x1 + 209.994527 x2
```

which is `210.110547 (x2 − x1)` plus a residual `(0.090721, −0.116020, −0.116020)` in the three coefficients. Class 2 beats class 1 when

```
x2 − x1 > −4.3178×10⁻⁴ x0 + 5.5219×10⁻⁴ x1 + 5.5219×10⁻⁴ x2
```

The true median boundary, in the cone where position 0 is the extreme and positions 1 and 2 are the two closest to the middle, is the plane `x2 = x1`. The network’s plane is tilted off that by `0.036°` (the angle between the normals `(0.090721, −210.226568, 209.994527)` and `(0, −1, 1)`).

The same pattern repeats in the other cells: four or so large contributions of order `10²` cancel to a pure difference of the two competing coordinates, with a relative mismatch of order `10⁻³` and a leak of order `10⁻⁴` from the third coordinate. The angle between the logit normal and `e_i − e_j` in the cells that carry the errors is between about `0.01°` and `0.06°`.

So the algorithm the weights implement is:

1. Store `x0` in a sign-split gain code.
2. Quantize the ratio `x1/x0` with the eight gates above. That code is coarse: the cuts sit near `±0.3, ±2.2, 0.7, 1.0, 2.3, 6.8, 47`, not only at `1`.
3. Quantize `x2` against that code the same way.
4. Let the linear readout finish the job. Inside the cell, cancellation moves the decision surface from those coarse cuts back to within `10⁻³` (relative) of the true tie `x_i = x_j`.

The position that wins is the median, because for three numbers the median is exactly the winner of “which of these two is closer to the third, given which one is the extreme”. Each ratio-code cell is one case of that contest (which input is the extreme, and which pair remains), and the surviving logit difference is that contest.

## Where it is wrong

It is wrong on the wedge between `x_i = x_j` and the slightly tilted logit plane, and only on the side where the tilt and the truth disagree, and only inside the cell where `i` and `j` are the two median candidates. Crossing `x_i = x_j` when the third input lies **between** them does not change the median (the third input is the median on both sides), and the network does not err there either.

The band is thin. On fresh standard normal inputs the relative gap `‖x‖⁻¹` times the gap between the true median value and the nearest other value, restricted to mistakes, has median about `3×10⁻⁴` and maximum about `2×10⁻³`. The predicted index on a mistake is always the other member of that close pair, never the far extreme.

The largest wedges, with the network plane written as the logit difference `n · x = 0` (`n · x > 0` means the predicted class beats the true class) and the Gaussian fraction of each wedge:

| fraction | true class | predicted | `n = L[pred] − L[true]` | cell (sign of `x0`, step-2 on-units, step-3 on-units) |
| --- | --- | --- | --- | --- |
| 7.995×10⁻⁵ | 2 | 1 | (−0.275, −295.333, +295.439) | `x0<0`, `{0,3,4,5}`, `{1,2,3}` |
| 6.460×10⁻⁵ | 0 | 2 | (+294.661, −0.289, −294.432) | `x0>0`, `{1,2,6}`, `{0,3,4,5}` |
| 6.237×10⁻⁵ | 1 | 2 | (+0.091, −210.227, +209.995) | `x0>0`, `{1,2,6}`, `{0,4,5,6}` |
| 4.128×10⁻⁵ | 0 | 2 | (−204.641, +0.027, +204.457) | `x0<0`, `{0,3,4,5}`, `{1,2,6}` |
| 2.825×10⁻⁵ | 0 | 1 | (−422.537, +422.307, −0.072) | `x0<0`, `{0,4,5,6}`, `{0,2,3}` |
| 2.522×10⁻⁵ | 0 | 2 | (+294.727, −0.228, −294.432) | `x0>0`, `{1,4,6}`, `{0,3,4,5}` |
| 2.382×10⁻⁵ | 1 | 0 | (−740.442, +740.732, +0.009) | `x0>0`, `{1,2,3,6}`, `{1,4,6}` |
| 1.848×10⁻⁵ | 1 | 2 | (+0.089, +294.870, −295.439) | `x0<0`, `{0,4,5,6}`, `{1,2,3}` |

These eight are 63% of the error measure. The same tilt, with a different leftover `n`, appears in the other ratio-code cells. There are 167 such wedges with positive solid angle. The two biggest families are “positions 0 and 2 are almost tied and the network picks 2” and “positions 1 and 2 are almost tied”.

A representative boundary, the third row, is the worked example above: the network treats position 2 as beating position 1 when `x2 − x1` exceeds a linear form whose coefficients are about `5×10⁻⁴`.

The direction of the tilt is **not** the same in every cell. Sometimes the network prefers the later index, sometimes the earlier one, by an amount proportional to a linear form in all three inputs. A single global rule of the form “if the relative gap is below `g`, answer index `k`” does not reproduce the mistakes, because `g` changes sign across cells. `twin.py` therefore implements the untilted rule, the exact median position, and disagrees with the network exactly on these wedges.

## Accuracy

The error probability is the Gaussian measure of the union of those wedges.

Each wedge is a convex polyhedral cone: the activation inequalities of one ratio-code cell, the two logit inequalities that select the wrong class, and the two ordering inequalities that make the other class the true median (`x_a ≥ x_b ≥ x_c` written as two linear inequalities). The Gaussian probability of a cone is its solid angle over `4π`. Solid angles were computed from the cone’s extreme rays (intersections of the bounding planes that satisfy every inequality), ordered around the cone axis, and summed as spherical triangles via

```
Ω(a,b,c) = 2 atan2( a · (b × c) , 1 + a·b + b·c + c·a )
```

on the unit vectors of the rays. Two-faced cones, which are lunes, use the dihedral angle `π − arccos(n1·n2)` and solid angle twice that. The orthant, the half-space, and each of the six median-ordering cones (each of measure `1/6`) were used as checks.

Sum of the wedge solid angles: `0.006849891`. Divided by `4π`:

```
0.000545097
```

Accuracy:

```
1 − 0.000545097 = 0.999455
```

Included: every wedge in every activation cone. The cones themselves partition space (solid angles sum to `4π`). Left out: nothing of positive measure. Ties on `x_i = x_j` are a lower-dimensional subset of the wedge boundary and do not contribute.

This number is the measure of the bands the cancellation failure creates. It is not a hit rate from sampling the network. A separate draw of `10⁵` standard normal inputs, used only as a check, had 57 mismatches with the true median (rate `0.00057`), against `54.5` expected from `0.000545097`.

## What is not explained

- There is no global re-expression of the eight units as eight fixed features such as `relu(x_i − x_j)` that holds in every orthant. Away from a single ratio cell, a unit’s on-condition is a different linear form. The population code above is the description that does hold globally; a shorter one was not found.
- The cuts sit at awkward ratios (`0.715`, `2.26`, `6.77`, …) rather than at `1`. The readout cancellation then moves the **decision** back to ratio `1`. I can derive each cut from `w_hh @ relu(±w_ih)`, but not a reason those particular cuts, rather than a sharper code, were the ones gradient descent kept.
- Unit 7’s self-weight explains its extreme threshold. It does not explain why that unit is allocated a readout column at all, given how rarely the threshold is crossed and how small the column is.
- The tiny gap and overlap between the near-complement pairs (units 0/1 and 2/4) are visible in the thresholds and have measure well below `10⁻³`. They are extra linear pieces, not a separate algorithm, and they are already included in the wedge integral.
- Why training left a `10⁻³` relative residual instead of cancelling exactly is an optimization fact, not a further computational step. The residual’s effect is the wedge, and that is fully determined by the weights.
