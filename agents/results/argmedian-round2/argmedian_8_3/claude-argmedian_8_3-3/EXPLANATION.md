# How the 8-unit ReLU RNN computes the position of the median

## Summary
The network computes the argmedian essentially exactly. Because it has **no biases**, it is
positively homogeneous (`logits(c·x) = c·logits(x)` for c > 0), so every decision boundary is a
cone through the origin. Each "is x_i above x_j?" boundary is the plane `x_i = x_j`, tilted very
slightly: the switch happens at `x_i - x_j = d_ij`, with `d_ij = A·m + B·x_k`, where m is the mean
of the pair and x_k is the third input. The tilt is about 0.03–0.1% of the input scale. So the
network is wrong **only on near-ties**, where the median and its nearest neighbour differ by less
than about 0.001×scale. Measured accuracy on 200k Gaussian samples is 0.99944. In my
quadrature estimate (estimate.txt), the error comes out at about 4.3e-4.

## Step by step
**t = 1 (sign split of x1).** `h1 = relu(w_ih·x1)`. Units {0,2,3,5,7} (w_ih > 0) store
`w_ih·x1⁺`, and units {1,4,6} (w_ih < 0) store `|w_ih|·x1⁻`. So h1 holds x1 as a positive part
and a negative part. That keeps the value of x1 while letting later ReLUs gate on its sign.
(I checked by regression: R² = 1.000.)

**t = 2 (comparison units).** The pre-activation of unit u is `w_ih[u]·x2 + P_u·x1⁺ + N_u·x1⁻`,
where `P = w_hh @ max(w_ih,0)` and `N = w_hh @ max(-w_ih,0)` are computed from the weights:
`P = [-14.9, 12.5, 16.3, -11.6, -16.9, -9.8, 10.1, -113.7]`,
`N = [20.1, -31.6, -13.6, 5.1, 14.0, 8.5, -3.2, -229.9]`.
So each unit is a ReLU of a signed comparison between x2 and a multiple of x1. For example:
- unit 6: `relu(-9.87·x2 + 10.06·x1⁺ - 3.15·x1⁻)`. For x1 > 0 it fires when x2 < 1.019·x1,
  which is almost exactly the comparison "x2 < x1".
- unit 3: `relu(16.2·x2 - 11.6·x1⁺ + 5.1·x1⁻)` fires when x2 > 0.715·x1 (x1 > 0) or
  x2 > -0.317·|x1| (x1 < 0).
- unit 0 fires when x2 > 2.27·x1 (x1 > 0) or x2 > -3.05·|x1| (x1 < 0). Units 1, 2, 4 and 5 are
  similar mixes. Unit 7 is almost dead after t = 1 (it is active on 0.4% of inputs at t = 2 and
  3.3% at t = 3) because of its huge negative recurrent weights (-113.7, -229.9). Its main job
  seems to be its large weights back into the other units at t = 1.
After t = 2 the hidden state encodes, piecewise linearly, both the order of (x1, x2) and their
values (the regression R² of each unit on x1, x2 and their positive parts is 0.88–0.98).

**t = 3 and readout.** At t = 3 the units again compare x3 with the stored order of (x1, x2), and
`w_out` turns the result into the logit for "position of the median". I did **not** finish a
unit-by-unit description of t = 3 or of the readout. Instead I described the function they
compute by measuring its decision boundaries directly (next section).

## Where it is wrong, and the constants
Every error in the sample is a near-tie between the median and one of its neighbours (relative
gap < 0.003). I found the boundaries by bisection. For each pair (i,j) I set
`x_i = m + δ/2, x_j = m − δ/2`, with the third input x_k fixed, and bisected on δ for the point
where the prediction switches between i and j (135k model calls; 355k model inputs in total, all logged). The switch
point `δ*` is linear in (m, x_k) within each sign region, to a residual of ~1e-4 or less. That is
expected: the network is piecewise linear and homogeneous. A pooled least-squares fit per pair and
per side of the third input gives the 12 constants in `twin.py`:

| pair | k above the pair (A, B) | k below the pair (A, B) |
|---|---|---|
| (0,1) | (-0.00046, 0.00002) | (0.00034, -0.00011) |
| (0,2) | (-0.00092, 0.00014) | (-0.00086, 0.00079) |
| (1,2) | (-0.00059, 0.00063) | (0.00046, -0.00075) |

Rule: if x_k is above the pair, the median is the larger of the pair, and the network picks i
when `x_i − x_j > A·m + B·x_k`. If x_k is below, the median is the smaller, and the network picks
i when `x_i − x_j < A·m + B·x_k`. These constants are **measured from the model's behaviour**.
I did not derive them in closed form from the weights. The tilts are tiny, nonzero leftovers of
imperfect training: the ideal boundary is A = B = 0.

Errors happen when the gap between the median and its closer neighbour falls between 0 and
`A·m + B·x_k`. The Gaussian measure of these bands (see estimate.txt) is about 4.3e-4.

## What I could not explain
- I did not decompose the t = 3 units or the readout logit by logit, and I did not derive the
  tilt constants analytically from the weights.
- The shift coefficients actually vary by sign region (in some regions by up to ~0.003×scale,
  e.g. pair (1,2) with all inputs negative). The twin and the estimate use one pooled fit per
  pair and side, so they slightly underestimate the error. Measured error is 5.6e-4 against 4.3e-4
  estimated. I also did not model triple near-ties, where all three inputs are close and the
  boundaries bend.
- The twin matches the model on 99.976% of the 200k sampled inputs.
