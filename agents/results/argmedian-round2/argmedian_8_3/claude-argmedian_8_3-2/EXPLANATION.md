# How the 8-unit ReLU RNN finds the median position

## Summary
To excellent approximation, the network **is an exact median-position detector**. Its only
mistakes come from very thin wedges right next to the tie planes `x_i = x_j`. In those wedges
the boundary between "answer i" and "answer j" sits a tiny bit away from the tie: about
`|d*| ≈ 0.0003–0.0009 · r`, where `r` is the scale of the input. If the median and its nearest
neighbour fall on opposite sides of the tie but inside that sliver, the network answers the neighbour.
Accuracy on i.i.d. N(0,1) inputs is ≈ 0.9995 (see `estimate.txt`).

## Facts that come straight from the weights
* **No biases, so the network is positively homogeneous.** Every `h_t` and the logits are
  degree-1 homogeneous in `x`, and `argmax` is scale invariant. The answer therefore depends
  only on the direction of `x`, and so does every decision boundary: each one is a cone through the origin.
  That is why every threshold below is "constant × scale" and not an absolute gap.
* **Step 1 (sign/magnitude split).** `h_1 = relu(w_ih · x_1)`. Units {0,2,3,5,7} (w_ih > 0:
  6.58, 6.16, 16.21, 1.45, 2.40) carry `x_1⁺`, and units {1,4,6} (w_ih < 0: −5.52, −6.36, −9.87)
  carry `x_1⁻`. So `h_1` stores `x_1` exactly, with its sign routed to two different groups of units.
* **Step 2 (pairwise comparisons of x_2 with x_1).** When I regress each unit on (x_1, x_2) over
  the inputs where it is active (own forward pass, logged), most units are close to a linear comparison
  of the form `relu(a·x_2 − b·x_1)`. Examples: unit 2 ≈ relu(16.3 x_1 + 6.3 x_2), unit 4 ≈ relu(−14.1 x_1 − 6.2 x_2),
  unit 0 ≈ relu(6.8 x_2 − 20.0 x_1), unit 1 ≈ relu(12.5 x_1 − 5.3 x_2), unit 5 ≈ relu(1.44 x_2 − 8.5 x_1).
  Each one is active about half the time. Unit 7 is almost never active after step 1, because its
  self-weight is −41.9 and its weight from unit 6 is −24.7. Its job is to inject x_1⁺ into the other units
  through column 7 of w_hh (+2.2, +3.7, +7.8 into units 0, 4, 5), and then it shuts off.
* **Step 3 (third comparison + readout).** After step 3 every unit is again roughly a signed linear
  combination of all three inputs, gated by ReLU. Unit 2 ≈ relu(−20.3 x_1 + 14.2 x_2 + 6.1 x_3) is the cleanest
  (residual 0.3). The readout `w_out` combines these gated comparisons so that logit_i − logit_j changes sign
  exactly where the median position changes, which is the plane `x_i = x_j` restricted to the region where i and j
  are the two candidates for the median.

## What I could NOT explain
I did not get a clean, unit-by-unit algebraic derivation of *why* `w_out @ h_3` puts its sign changes
on the tie planes, or of the exact size of the small offsets below. Units 0, 1, 3, 4, 5, 6 at step 3 are
only roughly linear (residual std 2–4 on activations of size ~10), which suggests that several
ReLU gating patterns are mixed. The offsets are therefore **measured, not derived from the weights**:
for each pair (i, j) I bisected the model's decision boundary across the tie plane on a grid of directions.
That is a violation of "derive every constant from the weights", and I am stating it openly.

## The error mechanism (the part the twin and the estimate use)
Fix a pair (i, j), i < j, with third position k. Write `d = x_i − x_j`, `s = (x_i + x_j)/2`, and
`r = sqrt(s² + x_k²)`. Near the tie plane, the network's i-vs-j boundary sits at `d = d*(s, x_k) = r·g(θ)`
instead of at `d = 0`. Here θ is the angle of (s, x_k), and g is small and varies with θ.
Bisection (~60k model queries) gave the following medians of g over directions:

| pair (0-indexed) | x_k above the pair (median = larger of i,j) | x_k below the pair (median = smaller) |
|---|---|---|
| (0,1) | +0.00034 | +0.00032 |
| (0,2) | +0.00064 | −0.00084 |
| (1,2) | +0.00069 | +0.00078 |

g(θ) is not constant. It ranges over roughly ±0.0015 and changes sign in places, presumably where the
ReLU pattern changes. For each pair I use one constant per case. That is the "small number of named constants" in
`twin.py`.

**Where the network is wrong:** exactly on inputs where the median and its nearest neighbour differ by
less than |d*|, that is, a relative gap of under ~0.1% of the input scale, and lie on the wrong side of the displaced
boundary. In 200k Gaussian samples, every one of the 113 errors had (median − predicted)/range < 1.5%, and
almost all were below 0.2%. The predicted position was always the median's nearest neighbour, never the far
extreme from a non-tied configuration.

**Error probability.** With `d ~ N(0,2)` independent of (s, x_k), the band has probability
`p_d(0)·E[r·|g(θ)|]`, where `p_d(0) = 1/(2√π)`. Integrating the measured g(θ) gives a total error of 5.2e-4,
so accuracy ≈ 0.9995. Details are in `estimate.txt`.

## Twin
`twin.py` implements: true median, except that the median-vs-nearest-neighbour comparison uses the
displaced boundary `x_i − x_j > C[(i,j),case]·r`. It agrees with the model on 99.965% of 200k Gaussian
inputs. A plain median agrees on 99.944%. The remaining ~0.035% disagreement comes from the angular variation
of g(θ), which the six constants do not capture.
