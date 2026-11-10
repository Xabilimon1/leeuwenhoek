# How the 8-unit ReLU RNN finds the median position

## Summary
The network is, to within very thin bands, an **exact median finder**. On i.i.d. Gaussian inputs
it is right ~99.94% of the time; every error I found is a **near-tie between the median and one of
the extremes** (|x_i - x_j| ~ 1e-3 x scale), where the network swaps those two positions. I could
describe *where* its pairwise decision boundaries sit (measured, then summarised by a small formula),
but I could **not** give a clean unit-by-unit algorithm for how the 8 units implement the comparisons.

## Structural facts derived from the weights
1. **Positive homogeneity.** No biases + ReLU => logits(c x) = c logits(x) for c > 0. So the prediction
   depends only on the direction of x, and every boundary is a cone through the origin; any
   "gap" constant is relative (proportional to |x|).
2. **Step 1 is a sign split.** h_1 = relu(w_ih x_1). w_ih = [6.58, -5.52, 6.16, 16.21, -6.36, 1.45, -9.87, 2.40]:
   units {0,2,3,5,7} carry x_1 when x_1 > 0, units {1,4,6} carry |x_1| when x_1 < 0.
3. **Step 2 units are ray thresholds comparing x_2 to x_1.** Pre-activation of unit u at step 2 is
   w_ih[u] x_2 + a_u x_1 with a = w_hh @ relu(w_ih) if x_1>0, a = w_hh @ min(w_ih,0) if x_1<0, so unit u
   fires on one side of the line x_2 = rho_u x_1 with rho = -a/w_ih:
   - x_1>0: rho = [2.27, 2.26, -2.65, 0.72, -2.65, 6.77, 1.02, 47.3]
   - x_1<0: rho = [3.05, 5.73, -2.21, 0.32, -2.21, 5.85, 0.32, -95.6]
   e.g. unit 6 (x_1>0) is active iff x_2 < 1.02 x_1 — essentially the comparison x_2 < x_1 and it carries
   the amount (x_1 - x_2)-like; unit 3 (w_ih=16.2) mostly tracks x_2 itself. Unit 7 is almost dead after
   step 1 (active in ~0.5% of cases at step 2, ~3% at step 3) because its self-weight is -41.9 and its
   input from unit 6 is -24.7.
   So after step 2 the state is a piecewise-linear code of (x_1, x_2) split by several rays near x_2 = x_1
   (plus rays at other slopes that encode magnitude/sign information).
4. **Step 3 and readout.** Step 3 adds x_3 with the same w_ih; the logits w_out @ h_3 are piecewise linear.
   Empirically the argmax equals the true median except near the three tie planes x_i = x_j. I did not
   manage, in the time budget, to decompose the step-3 / readout computation into named comparisons.

## Where the decision boundaries are (the rule the twin uses)
For each pair (i, j) with third index k, write v = (x_i+x_j)/2, w = x_k, u = x_i - x_j. Away from ties the
network's choice is the median. Near u = 0 the boundary between answering i and answering j sits at
u = D_ij(v, w), not at u = 0. By homogeneity D_ij(v,w) = r d_ij(theta) with (v,w) = r(cos theta, sin theta).
I measured d_ij(theta) by bisection on the logit difference L_i - L_j at 720 angles (this is a
measurement of the boundary, ~90k model calls). Findings:
- |d| is ~2e-4 to 1.5e-3, i.e. the network treats x_i and x_j as "equal" within a relative gap of
  ~0.03–0.15% of |x|. Pair (0,1) (decided earliest, at step 2) has the smallest and smoothest offset,
  ~0.6e-3|v| - 0.2e-3 v when they are the upper two; pairs involving x_3 have larger, more irregular offsets
  (~1e-3), consistent with x_3 being compared only once, through the mixed step-3 units.
- Summary formula (least-squares fit per pair, separately for "pair is upper two" w<v and "lower two"
  w>v): D = a v + b w + c|v| + d|w|; coefficients (x1e-4) are in `twin.py` (TIE_OFFSET). Fit residuals are
  tiny for pair 01-upper, larger for others (the true d(theta) has kinks I did not model).
- The twin then computes "i beats j iff x_i - x_j > D_ij" for each pair and answers the position that
  beats exactly one other. Agreement with the network on 100k fresh Gaussian inputs: **99.981%**
  (plain true median: 99.937%).

## When the network is wrong, and why
Errors occur exactly when sign(x_i - x_j) differs from sign(x_i - x_j - D_ij), i.e. x_i - x_j lies in
the thin band between 0 and D_ij, for a pair (median, extreme). The network then reports the extreme
instead of the median (examples: [-1.524, 0.193, 0.192] -> 1 instead of 2; [0.518, 0.519, 0.852] -> 0
instead of 1). The cause is that the finite-precision piecewise-linear comparisons built from the ray
units do not put their switching line exactly on x_i = x_j. Because of homogeneity the band widens
proportionally to |x|. Near triple ties (all three nearly equal) the bands overlap and behaviour is
messier; I did not characterise that region (it carries negligible probability).

## Accuracy
P(error for pair ij) = phi_u(0) E|D_ij(v,w)| with u ~ N(0,2) independent of v ~ N(0,1/2), w ~ N(0,1),
phi_u(0) = 1/(2 sqrt pi). Quadrature over the measured offsets gives total error 0.00054,
accuracy ~ **0.99946** (see estimate.txt). For reference only (not used for the estimate), sampled
accuracy was ~0.9994.

## What I could not explain
- A clean functional role for each hidden unit at step 3, and how w_out turns h_3 into exactly median
  comparisons. The units are distributed; I only characterised steps 1–2 analytically.
- Derivation of the tie offsets D_ij from the weights in closed form: they were measured by bisection on
  the model's logits and summarised by a fit, not derived symbolically.
- The precise behaviour near triple near-ties.

## Notes on process
Model evaluations: ~790k inputs via load_model (exploration, bisection, twin check) plus one 20k-input
own forward pass (logged). I copied twin.py once into /tmp to verify it runs from a directory with only
.py files.
