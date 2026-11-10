# How the 2-unit ReLU RNN "finds the median"

Short answer: it mostly doesn't. It **never answers position 2**, answers
**position 1** only in two thin cone-shaped regions (mainly "x1 < 0 < x2 and x3
not much smaller than x2"), and answers **position 0** everywhere else.
Accuracy on i.i.d. N(0,1) inputs ≈ **0.41** (analytic 0.4102; sampled 0.4097).

Notation: `a = w_ih = (2.950, 3.153)`, `w = w_hh = [[-76.49, -2.505], [1.193, -1.986]]`,
`o = w_out`. All constants below are computed in `estimate_regions.py`'s header /
by the formulas given (I computed them with a scratch script from the weights).

## 1. Readout: only "unit 2 vs unit 1"

Every row of `w_out` is roughly proportional to (1, -1), so the logits depend
mainly on the balance of the two units at t=3.
- logit0 > logit1  ⇔ `(o00-o10) h1 > (o11-o01) h2` ⇔ **h2 < r·h1**, with
  `r = (o00-o10)/(o11-o01) = 1.1113`.
- logit2 beats logit0 only if h2/h1 > (o20-o00)/(o01-o21) = 1.280, and beats
  logit1 only if h2/h1 < (o10-o20)/(o21-o11) = 1.051. Both cannot hold, so
  **position 2 is never predicted** (h ≥ 0 always, after the ReLU).
- If h_3 = 0 all logits tie at 0 and argmax returns 0.

So: **predict 1 iff h2_3 > r·h1_3 (and h2_3 > 0); otherwise predict 0.**

## 2. What the hidden units do

- **Unit 1** has huge self-inhibition (w11 = -76.5). It fires on a positive
  input only if it was silent at the previous step; once it has fired, the next
  step it is crushed unless the new input is ~79× bigger. It is a
  "positive input after silence" detector, and it also pushes unit 2 up
  (w21 = +1.19) and is pushed down by unit 2 (w12 = -2.5).
- **Unit 2** carries a "rise" signal: current input minus a damped copy of
  the previous state (self-weight w22 = -1.99).

Step 1: `h_1 = relu(x1)·a`, i.e. both units hold x1⁺ (ratio a2/a1 = 1.069 < r).

Step 2:
- If x1 ≤ 0: `h_2 = relu(x2)·a`.
- If x1 > 0: unit 1 gets `a1 x2 + (w11 a1 + w12 a2) x1`, positive only if
  `x2 > G x1`, **G = -w11 - w12 a2/a1 = 79.17**; so normally unit 1 = 0.
  Unit 2 gets `a2 x2 + (w21 a1 + w22 a2) x1 = a2 (x2 - K x1)` with
  **K = -w22 - w21 a1/a2 = 0.8702**. Define the rise **U = x2 - K x1**.

Step 3 and readout give the regions (each is a cone, the rule is scale-free):

**Region A** (x1 ≤ 0, x2 > 0; both units hold a·x2). Unit 2 at t=3 is
`a2 (x3 - K x2)` (the same K: unit 1's excitation cancels part of unit 2's
self-inhibition), unit 1 is `a1 (x3 - G x2)`. Predict 1 iff
**x3 > K·x2** and **x3 < D·x2**, where the upper cut (unit 1 revives and wins the
readout) is `D = (r a1 G - a2 K)/(r a1 - a2) = 2045` – practically never active.

**Region A'** (0 < x1 < x2/G, a rare sliver, ~0.2% of mass): as A, but the
small positive x1 adds inhibition: predict 1 iff `x3 > K x2 + M x1` (and
x3 < D x2), where `M = [w21 (w11 a1 + w12 a2) + w22 (w21 a1 + w22 a2)]·(-1)/a2 = 86.6`.

**Region B** (x1 > 0, U > 0, x2 < G x1; h_2 = (0, a2 U)). At t=3 unit 2 is
`a2 (x3 - C_lo U)` with **C_lo = -w22 = 1.986**; unit 1 is
`relu(a1 x3 + w12 a2 U)`, which switches on for x3 > 2.68 U. Comparing with
the readout ratio r: predict 1 iff **C_lo·U < x3 < C_hi·U**, with
`C_hi = a2 (r(-w12) - (-w22)) / (r a1 - a2) = 20.03`.

Everything else (x2 ≤ 0 with x1 ≤ 0; x2 ≤ K x1 with x1 > 0; x3 outside the bands)
leaves unit 2 dead or dominated → **predict 0**.

## 3. Where it is right and wrong

- Median at position 2 (prob 1/3): always wrong (never predicts 2).
- Median at position 0 (1/3): right unless it falls in region B
  (P = 0.0030; e.g. x2 < x1 < x3 with x3 in the band). ~99% right.
- Median at position 1 (1/3): right only inside A/A'/B, mostly region A
  (x1 < 0 < x2 < x3, P = 0.0625) plus B (0.0172) and A' (0.0002); total
  0.080 of 0.333, i.e. ~24% right.
- Accuracy = 1/3 - 0.0030 + 0.0800 = **0.4102** (see estimate.txt).

So the trained network is a degenerate solution: a default "answer 0" plus a
detector for "the sequence rises from below 0 through x2 to x3", which is a
cheap partial cue for "x2 is the middle".

## 4. Twin and checks

`twin.py` implements exactly the rule above with the named constants
K, C_LO, C_HI, G, M, D. It matched the model on 299,999 of 300,000 fresh
Gaussian inputs (the one miss is within 1e-5 of the x3 = K x2 boundary, a
constant-rounding effect).

## 5. What I could not / did not explain

- Why training landed on this degenerate solution (never using position 2);
  I only describe the trained mechanism.
- In region A' the D cut ignores x1 (negligible); ties on boundaries (measure
  zero) are not treated specially.
