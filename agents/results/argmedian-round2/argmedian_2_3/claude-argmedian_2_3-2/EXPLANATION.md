# How the 2-unit ReLU RNN computes its answer

Weights (rounded): `w_ih = (2.950, 3.153)`,
`w_hh = [[-76.49, -2.505], [1.1925, -1.986]]`,
`w_out = [[2.845, -2.492], [-3.753, 3.445], [0.835, -0.921]]`.
Call the hidden units **a** (row 0) and **b** (row 1).

## Summary

The network **never predicts position 2** and is essentially a detector for
"x2 rose above x1, and x3 then rose further (but not absurdly far)". When that
fires it says 1, otherwise it says 0. Accuracy on Gaussian inputs is about
**0.41**. All inputs whose median is at position 2 (1/3 of the mass) are wrong
by construction.

## 1. Readout: only the direction of h3 matters, and only answers 0 and 1 occur

The logits are linear in `h3 = (a, b) >= 0`, so what matters is the ratio `b/a`.
- logit1 > logit0 iff `b/a > rho`, with
  `rho = (w_out[0,0]-w_out[1,0]) / (w_out[1,1]-w_out[0,1]) = 1.1113`.
- logit2 > logit0 needs `b/a > (w_out[0,0]-w_out[2,0])/(w_out[2,1]-w_out[0,1]) = 1.280`.
- logit2 > logit1 needs `b/a < (w_out[2,0]-w_out[1,0])/(w_out[1,1]-w_out[2,1]) = 1.051`.

Both conditions for 2 can't hold at once, so **class 2 is never the argmax**.
When `h3 = 0` (about 60% of inputs) every logit is 0 and argmax returns 0.
So the decision is: **answer 1 iff `b3 > rho * a3` (strictly, which needs `b3 > 0`), else 0.**

Since `w_ih[1]/w_ih[0] = 1.069 < rho`, a state proportional to `w_ih`
(meaning "only the current input, nothing inhibited") reads out as 0.

## 2. Step 1: both units hold relu(x1)

`h1 = w_ih * relu(x1)`, so a1 = 2.95·p1 and b1 = 3.15·p1, with p1 = relu(x1).

## 3. Step 2: unit a is a "x1 was not positive" gate, unit b is the rise of x2 over x1

Feeding h1 through `w_hh`:
- `a2 = w_ih[0] * relu(x2 - GATE*p1)`, with
  `GATE = -(w_hh[0]·w_ih)/w_ih[0] = (76.49*2.950 + 2.505*3.153)/2.950 = 79.17`.
  When x1 > 0, a is wiped out (unless x2 is more than 79 times x1). When x1 <= 0, a2 = 2.95·relu(x2).
- `b2 = w_ih[1] * relu(x2 - K*p1)`, with
  `K = -(w_hh[1]·w_ih)/w_ih[1] = (1.986*3.153 - 1.1925*2.950)/3.153 = 0.8702`.
  Call `rise = relu(x2 - K*p1)`. It is about "how much x2 exceeds x1" (with a slope of 0.87, not 1).

That gives two regimes:
- **Reset branch** (x1 <= 0, or x2 > 79.17·x1): h2 = w_ih·relu(x2), the same form as
  h1 with x2 in place of x1. In effect the network forgets x1.
- **Normal branch** (x1 > 0): h2 = (0, 3.153·rise). Only b carries information.

## 4. Step 3 and the decision

**Normal branch** (a2 = 0, b2 = 3.153·rise):
- `b3 = 3.153 * relu(x3 - LO*rise)` with `LO = -w_hh[1,1] = 1.986`
- `a3 = 2.950 * relu(x3 - 2.677*rise)` (2.677 = -w_hh[0,1]·w_ih[1]/w_ih[0])
- If rise = 0, h3 ∝ w_ih and the answer is 0. If rise > 0, the answer is 1 iff
  `LO*rise < x3 < HI*rise`, where
  `HI = (rho*w_hh[0,1] - w_hh[1,1]) * w_ih[1] / (rho*w_ih[0] - w_ih[1])`
  `= (−2.784 + 1.986)·3.153 / (3.278 − 3.153) = 20.03`.
  (Between 1.986 and 2.677 times the rise, a3 = 0 and b3 > 0, so the answer is 1. Above
  2.677, a turns on but b/a stays above rho until x3 = 20.03·rise.)

**Reset branch** (h2 = w_ih·relu(x2)): this is the step-2 calculation again,
with x2 in the role of x1.
- If x2 <= 0, the answer is 0.
- If x2 > 0, the answer is 1 iff `LO0*x2 < x3 < HI0*x2`, with `LO0 = K = 0.8702` and
  `HI0 = (rho*(w_hh[0]·w_ih) - (w_hh[1]·w_ih)) / (w_ih[1] - rho*w_ih[0]) = 2045`
  (in practice no upper limit).

**Rule (twin.py):** answer 1 when x3 falls in the window above (normal branch:
`1.986·rise < x3 < 20.03·rise`, with `rise = x2 − 0.87·relu(x1)`; reset branch:
`0.87·x2 < x3 < 2045·x2`, with x2 > 0). Otherwise answer 0.

## 5. Where it gets the median wrong, and why

- **Median at position 2 (mass 1/3): always wrong.** The readout can't produce 2.
- **Median at position 1 (x2 in the middle) but answered 0.** This happens in
  three cases:
  - x2 is the middle value with x1 > x2 > x3 (descending order). The network only
    detects rising patterns.
  - The order is rising, but x3 isn't far enough above the rise
    (`x3 < 1.986·rise`, normal branch).
  - x3 is above x2 but x2 is not sufficiently above 0.87·x1.
  Of the 1/3 mass where x2 is the median, only 0.080 is answered 1.
- **Median at position 0 but answered 1 (mass 0.003).** Normal branch only.
  The "rise" detector uses slope K = 0.87 instead of 1, so it also fires when
  `0.87·x1 < x2 < x1` (x2 slightly *below* x1). If x3 then lands in the window
  above x1, the order is x2 < x1 < x3, so x1 is the median but the network
  answers 1. (The case x3 < x1 < x2 inside the window adds a little more.)
- **Median at 0 and answered 0: mostly correct**, since 0 is the default answer.

Accuracy = 1/3 − 0.00297 + 0.06279 (reset branch) + 0.01715 (normal branch) = **0.4103**.
See estimate.txt and integrate_rule.py. A sample of 200k model calls gave 0.4097,
which is consistent.

## 6. What I did not fully explain

- The sub-branch x1 > 0 with x2 > 79.17·x1: both units survive step 2 with
  slightly different offsets. I treat it as the reset branch. The twin
  disagrees with the network on about 0.015% of inputs, almost all in this
  sliver (x1 tiny and positive).
- *Why* training produced a network that never outputs 2 (presumably a poor
  local optimum for 2 units with no biases). I can describe it but not justify it.
