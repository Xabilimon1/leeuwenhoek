# How the 4-unit ReLU RNN computes argmedian of 3 inputs

## Weights and their symmetry
```
w_ih = [ 3.97  4.70 -4.72 -3.97]
w_hh = [[ 1.34 -2.20  2.64 -1.89]
        [ 1.93 -2.43  2.79 -2.38]
        [-2.39  2.81 -2.43  1.93]
        [-1.89  2.64 -2.19  1.34]]
w_out = [[ -1.67   1.60   1.62  -1.66]
         [-18.75  16.00  15.96 -18.72]
         [ 11.43 -15.74 -15.70  11.48]]
```
* Units come in mirror pairs (0,3) and (1,2): w_ih flips sign, and w_hh is (to ~1%)
  invariant under the permutation P = (0 3)(1 2) applied to rows and columns. Hence
  h_t(-x) ≈ P h_t(x): unit 0 is the "positive" copy and unit 3 the "negative" copy of the
  same signed quantity, likewise units 1/2. Each pair is a rectified (+,−) encoding of a
  signed number (relu(z)−relu(−z) = z, relu(z)+relu(−z) = |z|).
* w_out is (to ~1%) P-invariant: each logit only reads the **even** parts
  S = h0+h3 and R = h1+h2, i.e. magnitudes |·| of the two signed quantities. So the output is
  (almost) invariant under x → −x, as the median position must be.
* No biases ⇒ the network is positively homogeneous of degree 1: scaling x scales all
  logits, the argmax is scale-invariant, and every decision boundary is a cone. All band
  widths below are therefore constants times |a| or |c−a|.

## Step by step
* t=1: h1 = relu(w_ih x1): for x1>0 units 0,1 hold 3.97·x1, 4.70·x1; for x1<0 units 2,3
  hold |x1|·(4.72, 3.97). The pair stores x1 with its sign.
* t=2: w_hh @ h1 = x1·(−5.03, −3.80, 3.72, 4.91) for x1>0 (=3.97·col0 + 4.70·col1), so the
  pre-activations are 3.97(x2−1.27x1), 4.70(x2−0.81x1), −4.72(x2−0.79x1), −3.97(x2−1.24x1)
  (mirror-image for x1<0). The units now hold rectified **comparisons of x2 with x1**
  (x2−x1 up to a ±20% slop of x1): which side x2 is of x1, and by how much.
* t=3: the same recurrent matrix compares x3 against the stored comparison state, so the
  two signed quantities at the end (pair 0/3 → S, pair 1/2 → R) are piecewise-linear
  signed combinations of the pairwise differences x3−x2, x3−x1, x2−x1.
* Readout: logit differences are
  l1−l0 ≈ −17.1 S + 14.4 R, l2−l0 ≈ 13.1 S − 17.3 R, l1−l2 ≈ −30.2 S + 31.7 R
  (differences of w_out rows, pairing columns 0/3 and 1/2). So the answer is a threshold
  on the ratio R/S: **1 if R/S > 17.1/14.4 = 1.19, 2 if R/S < 13.1/17.3 = 0.76, else 0**.
  Answer 0 is the "middle" band of the ratio.

Functionally this implements "return the median position". I did not complete the full
algebraic derivation of R and S per sign region in the time available (30 min); the band
constants below were measured by scanning the network along lines through tie surfaces
(x_i = a, x_j = a + d, third input c, d scanned with step 2e-4, ~50k model calls), and
then fitting the edges of the wrong region as linear functions of |a| and |c−a| (which
homogeneity and piecewise linearity guarantee are exact within a linear region).

## Where it is wrong: three tie bands
Because the comparisons carry slop (x2 is compared with 0.8·x1 / 1.25·x1, not x1) and the
ratio thresholds are finite, when two inputs are almost equal the network cannot tell which
of them is the median and falls back to a fixed answer. Notation: a = tied value, d = gap
(sign-normalised: multiplied by sign(a)), c = third input, g = |c−a|, t = c/a.

* **Band A: x2 ≈ x3 → answers 0** (true answer 1 or 2). Dominant error (≈1.27%).
  - c on the zero side or opposite sign (t ≤ 0.5): wrong for
    −(α_lo|a| + β_lo g) < d < α_hi|a| + β_hi g with (a>0) α_lo=0.0072, β_lo=0.0170,
    α_hi=0.0164, β_hi=0.0070; (a<0, in −x coordinates) 0.0128, 0.0108, 0.0107, 0.0135.
    Total width ≈ 0.024(|a| + |c−a|) in both signs; the asymmetry comes from the ~1%
    breaking of the mirror symmetry in the weights.
  - t > 0.5 (third input on the same side, further out): symmetric half-width |a|·h(t), h
    measured at t = 0.5, 0.7, 1.0, 1.3, 1.6, 2, 3: 0.018, 0.036, ~0.10, 0.074, 0.041, 0, 0.0125;
    the band closes at c = 2a and reopens slowly beyond (≈0.0125(t−2)).
* **Band B: x1 ≈ x3 → answers 2** (true answer 0), when x2 is on the zero side of the tie:
  x3 just beyond x1 (away from x2), width 0.0108|c−a| − 0.0039|a|. (≈0.19%)
* **Band C: x1 ≈ x2 → answers 1** (true answer 0): x2 just beyond x1 (away from zero) with
  width 0.0055|a| if x3 is on the zero side; x2 just inside x1 with width 0.0195|a| if x3 is
  beyond. (≈0.14%)

Outside these bands the network returns the true median. twin.py implements exactly
this rule (median + three band overrides); it agrees with the model on 99.6% of fresh
Gaussian inputs. Estimated accuracy (estimate.txt, from band measures): 0.984.

## What I could not explain
* I did not derive the band constants symbolically from the weights; they were measured
  from the network along tie lines and fitted (the ratio thresholds 1.19 / 0.76 and the
  step-2 slopes 1.27, 0.81, 0.79, 1.24 are derived from the weights as shown).
* Band A for t between 0.5 and 2 (near-triple ties) is only described by a few measured
  knots, interpolated linearly.
* Band B when x2 lies beyond the tie (very small, sometimes answers 0) is omitted.
* The residual 0.4% disagreement of the twin comes from these approximations.
