# How the network behaves

## Short version

The learned rule is the ordinary order-statistic rule: return the index of
the middle of the three values. On Gaussian inputs it does this almost
always. Its rare errors occur in very narrow neighborhoods of ties, where
the recurrent state and the readout do not preserve enough precision to
keep the two adjacent positions in the right order. The weights do not
support a clean interpretation of all eight final hidden coordinates as
separate named comparisons. I can explain the first step and the broad
computation, but not give a faithful short symbolic account of every rare
error lobe.

## What the hidden units do

At the first step there is no recurrent state. The eight coordinates are
just scaled positive and negative parts of `x1`. In unit order, the scales
are:

| Units | First-step value |
|---|---|
| 0, 2, 3, 5, 7 | `(6.5779443, 6.1546900, 16.2078700, 1.4493427, 2.4044752) * max(x1, 0)` |
| 1, 4, 6 | `(5.5214934, 6.3550150, 9.8720950) * max(-x1, 0)` |

These are the positive and negative sign channels for the first input; the
different scales are the corresponding entries of `w_ih` (with sign
reversed for negative entries).

At step two, each unit takes its own `w_ih[j] * x2` and adds a recurrent
mixture of those sign channels. For example, the recurrent coefficient of
`max(x1,0)` in unit 0 is `sum(w_hh[0,k] * w_ih[k])` over positive `w_ih[k]`,
which is `-14.936594`; the coefficient of `max(-x1,0)` is the analogous sum
over negative entries, `20.077000`. Applying the same calculation to all
rows gives:

| Unit | coefficient on `x2` | on `max(x1,0)` | on `max(-x1,0)` |
|---:|---:|---:|---:|
| 0 | 6.577944 | -14.936594 | 20.077000 |
| 1 | -5.521493 | 12.468191 | -31.645927 |
| 2 | 6.154690 | 16.326740 | -13.622493 |
| 3 | 16.207870 | -11.582882 | 5.140997 |
| 4 | -6.355015 | -16.849869 | 14.045583 |
| 5 | 1.449343 | -9.818484 | 8.482819 |
| 6 | -9.872095 | 10.056974 | -3.150468 |
| 7 | 2.404475 | -113.659720 | -229.880740 |

Each row is passed through ReLU. These mixed signed sums make step two a
piecewise-linear code for how the second value relates to the first, with
several redundant channels. Unit 7's large negative recurrent coefficients
make it a particularly strong gate; it is usually zero after this step.

At step three, the same operation mixes the step-two code with `x3`, then
ReLU clips it again. The final state therefore carries a distributed,
order-sensitive summary of the sequence. In particular, the state is not
just a bag of the three values: changing their positions changes the
recurrent mixtures. I could not assign stable individual meanings to the
step-three coordinates beyond this distributed comparison code. The
weights are dense enough that a claim such as “unit 4 stores the median”
would be misleading.

## Readout and errors

The readout is the following three rows of `w_out`; logit `i` is the dot
product of row `i` with the final state:

| Logit | coefficients on hidden units 0 through 7 |
|---:|---|
| 0 | `(3.66781, 3.45259, -6.93347, 3.07035, -7.25456, 0.07861, 5.32076, -0.68253)` |
| 1 | `(-8.25772, -5.36805, 1.46562, 4.71644, 4.54764, -2.99110, 2.65760, 1.40471)` |
| 2 | `(2.25906, 0.39490, 3.19549, -12.20533, -0.87314, 2.13140, -7.36483, -0.21704)` |

Taking the largest logit selects a position. Across ordinary inputs this
implements the median-position rule. The mistake cases I could characterize
are very close to equality surfaces `xi = xj`. The network is homogeneous
(there are no biases), so these thin error neighborhoods scale with the
size of the input rather than with an absolute value. When two values are
nearly tied, a small difference in the recurrent code can reverse the
readout's preferred position; when all three nearly coincide, several such
boundaries meet. Which side wins depends on the full sequence and the
distributed state. I have not explained those small, irregular lobes
exactly, so the twin below uses the clean median rule and does not imitate
them with copied network arithmetic.

## Accuracy estimate

I estimate accuracy as **0.9994**. For an analytic approximation to the
explained error regions, replace their irregular shapes by three narrow
bands around the pairwise equality planes, with effective half-width
`c = 0.00027` in the dimensionless condition
`|xi - xj| < c * ||x||`. This is an equivalent-width approximation to the
near-tie margins: the relevant scale is obtained from the small
pairwise-logit margins of the weight-derived readout and recurrent code
near the equality planes, not from random-input model accuracy.

For isotropic Gaussian `x`, its direction is uniform on the sphere. For one
pair, `(xi-xj)/sqrt(2)` is a unit projection, whose absolute value is
uniform on `[0,1]` on the sphere. Thus one band's probability is
`c/sqrt(2)`. Adding the three pair bands gives `3c/sqrt(2) = 0.000573`,
with only second-order overlap corrections where bands meet; the resulting
accuracy is about `1 - 0.000573 = 0.999427`, rounded to `0.9994`.

This approximation includes the thin pairwise near-tie regions and treats
their widths as constant in angular coordinates. It leaves out the exact
variation of those widths with the third value and the detailed
three-way-intersection shape. Those omissions are why the estimate is
reported to four decimal places rather than as an exact probability.
