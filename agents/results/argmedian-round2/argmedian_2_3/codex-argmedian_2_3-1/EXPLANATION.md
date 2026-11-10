# Mechanistic explanation

## What the readout can say

Write the final hidden state as `(u,v)`, with both entries nonnegative. The
three logits from `w_out` are

```
L0 =  2.8454947 u - 2.4923484 v
L1 = -3.7529578 u + 3.4452689 v
L2 =  0.8350099 u - 0.92129105 v.
```

For class 2 to beat class 0 requires `u/v <= 0.7814`; to beat class 1
requires `u/v >= 0.9517` (when `v>0`). These conditions cannot both hold.
The same conclusion holds on the axes by direct substitution. Thus class 2
is never the argmax. If `u=v=0`, all logits tie and NumPy's argmax picks
class 0. Otherwise the class 0 / class 1 boundary is `u/v = r`, where

```
r = (3.4452689 - (-2.4923484)) / (2.8454947 - (-3.7529578))
  = 0.89984997.
```

The model therefore uses its final state as a binary switch: it says 1 when
there is positive `v` and `u/v < r`, and 0 otherwise. It has no way to answer
position 2.

## The approximate hidden-state story

Both input weights are positive, so the first input makes a pulse only when
`x1>0`:

```
u1 = 2.9500926 max(x1,0)
v1 = 3.152859  max(x1,0).
```

The large negative `u -> u` recurrent weight (-76.4917) usually suppresses
unit 1 at the next step. In the ordinary regime where that unit is off at
step 2, unit 2 carries a residual from the first input while responding to
the second. Substituting the step-1 state into its recurrence gives

```
v2 = max(0, 3.152859 x2 - 2.739... max(x1,0))
   = 3.152859 p,
p  = max(0, x2 - gamma max(x1,0)),
gamma = (1.9860576*3.152859 - 1.1925036*2.9500926)/3.152859
      = 0.87024619.
```

Here `p` is a simple proxy for the unit-2 carry entering the last step. At
step 3, temporarily take `u2=0` and use `v2=3.152859 p`. Then

```
u3 = max(0, 2.9500926 x3 - 2.5048876*3.152859 p)
v3 = max(0, 3.152859 x3 - 1.9860576*3.152859 p).
```

For `p>0`, unit 2 turns on when `x3 > 1.9860576 p`. Unit 1 turns on at
`x3 > 2.6770541 p`. Once both are on, `u3/v3=r` at

```
x3 = beta p,
beta = 3.152859*(2.5048876-r*1.9860576)/(2.9500926-r*3.152859)
     = 20.027107.
```

Consequently this simplified mechanism predicts class 1 in a band
`1.9860576 p < x3 < 20.027107 p`, and class 0 outside it. When `p=0`, it
predicts class 0, including the all-zero/tied case. The small difference
between the two input scales, combined with the readout boundary, makes the
upper edge of the band large.

## What goes wrong, and limits of this account

Class 2 predictions are structurally absent, so every input whose median is
at position 2 is wrong. The network also answers 0 throughout most of the
input space, so it misses many cases with median at position 1. Its class 1
band depends on positive earlier evidence and a sufficiently large positive
last input; that band does not in general mean `x2` is the median, so some
class 1 answers are wrong too. In the simplified rule, the exact correctness
conditions are: outside the band, `x1` must be the median for class 0 to be
right; inside the band, `x2` must be the median for class 1 to be right.
Every case with `x3` as median is wrong, regardless of the band.

This is an intentionally compact explanation, not a complete algebraic
description of every activation region. It treats unit 1 as off at steps 2
and 3. The first assumption fails for unusual large `x2` relative to a small
positive `x1`; the second can fail when `x3` is large enough to overcome the
unit-2 carry. It also replaces the full hidden state with the scalar `p`.
These omitted cases can move either edge or change the output. In particular,
the stated band and the probability calculation below approximate the
network's behavior; they are not claimed to be an exact characterization.

## Constants and sources

All constants above are rounded values derived from the arrays in
`weights.npz`: `gamma` and `beta` are shown by substitution, the lower band
edge is the magnitude of the unit-2 self-recurrence, and `r` is the equality
of the class-0 and class-1 readout logits. The readout comparisons use the
printed `w_out` values. No fitted thresholds or outside sources are used.
