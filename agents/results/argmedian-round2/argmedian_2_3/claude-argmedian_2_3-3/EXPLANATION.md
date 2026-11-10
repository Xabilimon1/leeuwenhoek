# How the 2-unit ReLU RNN "computes the median position"

Short answer: it doesn't, really. It answers **0 by default** and answers **1 only
when the sequence is rising** (x2 clearly above max(x1,0) and x3 clearly above x2).
It **never answers 2**. Its accuracy is ~0.41 (estimate.txt).

Weights (rounded): `w_ih = (2.950, 3.153)`,
`w_hh = [[-76.49, -2.505], [1.1925, -1.986]]`,
`w_out = [[2.845, -2.492], [-3.753, 3.445], [0.835, -0.921]]`.
Call the hidden units **a** (unit 0) and **b** (unit 1).

## 1. Readout: only the angle of (a,b) matters, and class 2 can't win

Hidden states are non-negative. Each logit row is roughly proportional to
`a - c*b` with different signs, so the answer depends on the ratio b/a:

- 0 vs 1: logit0 - logit1 = 6.598 a - 5.938 b, so 1 beats 0 iff
  **b > k a, k = (w_out[0,0]-w_out[1,0])/(w_out[1,1]-w_out[0,1]) = 1.1113**.
- 2 beats 0 needs b/a > 1.280; 2 beats 1 needs b/a < 1.051. Both at once is
  impossible, so with a,b >= 0 **class 2 is never predicted**.
- a = b = 0 gives all-zero logits; argmax ties go to 0.

So: **predict 1 iff b3 > k·a3 (and b3 > 0), else 0.**

## 2. The recurrent step: a "fresh positive input" detector plus a "rise" unit

When the state is zero, a positive input v writes (a,b) = (2.950 v, 3.153 v);
ratio b/a = w_ih[1]/w_ih[0] = 1.069 < k, so a lone positive input reads out as 0.
A non-positive input onto a zero state leaves it at zero.

From state (2.950 v, 3.153 v), the next input y gives
- a' = relu(2.950 y - 233.6 v): a is killed unless y > **R = 79.17 v**
  (R = -(w_hh[0,0] w_ih[0] + w_hh[0,1] w_ih[1]) / w_ih[0]). Unit a has huge
  self-inhibition (-76.5): it only survives if the new input dwarfs the old one.
- b' = relu(3.153 (y - r v)) with **r = -(w_hh[1,0] w_ih[0] + w_hh[1,1] w_ih[1]) / w_ih[1] = 0.8702**.
  b measures **how much y rose above 0.87 × the previous value** ("rise").

So b is a "rise" detector and a is a "big fresh jump" detector.

## 3. Full three-step rule

Let p = max(x1, 0) (what step 1 stored; x1 <= 0 stores nothing) and
rise = x2 - r p.

1. **rise <= 0** (x2 not above 0.87·x1, or x2 <= 0): state after step 2 is 0,
   step 3 can only write the "lone input" state (ratio 1.069) -> **answer 0**.
2. **0.87 x1 < x2 < 79.17 x1** (x1 > 0; a dead, b = 3.153·rise): at step 3
   b3 = 3.153 x3 - 1.986·(3.153 rise), a3 = relu(2.950 x3 - 2.505·(3.153 rise)).
   - b3 > 0 iff **x3 > C_LO·rise, C_LO = -w_hh[1,1] = 1.986**.
   - When x3 is very large, a3 grows faster than b3 (2.950 k = 3.278 > 3.153)
     and the readout flips back to 0: b3 > k a3 iff
     x3 < **C_HI·rise, C_HI = w_ih[1](w_hh[1,1] - k w_hh[0,1]) / (k w_ih[0] - w_ih[1]) = 20.03**.
   - **Answer 1 iff 1.986·rise < x3 < 20.03·rise.**
3. **x2 > 79.17·p** (includes every x1 <= 0 with x2 > 0; a survives step 2,
   A = 2.950(x2 - R p), B = 3.153(x2 - r p)): a3 is crushed by -76.5 A, so
   b3 > 0 decides: x3 > (-w_hh[1,1] B - w_hh[1,0] A)/w_ih[1] =
   **0.8702 x2 + M·p, M = (w_hh[1,1] w_ih[1] r + w_hh[1,0] w_ih[0] R)/w_ih[1] = 86.61**
   (the x2 coefficient is exactly r again). An astronomically large x3 would
   again let a win: cap x3 < CAP_X2·x2 - CAP_X1·p with
   CAP_X2 = ((w_hh[1,0]-k w_hh[0,0]) w_ih[0] + (w_hh[1,1]-k w_hh[0,1]) w_ih[1])/(k w_ih[0]-w_ih[1]) = 2045,
   CAP_X1 = ((w_hh[1,0]-k w_hh[0,0]) w_ih[0] R + (w_hh[1,1]-k w_hh[0,1]) w_ih[1] r)/(k w_ih[0]-w_ih[1]) = 1.6e5
   (matters only when x2 is tiny, ~1e-4).
   The common case is x1 <= 0: **answer 1 iff x3 > 0.87·x2 (x2 > 0)**.

The two branches agree on the shared boundary x2 = 79.17 x1 (both give
x3 > 155.5 x1), so the decision surface is continuous.

## 4. Interpretation and errors

In words: "answer 1 if x1 <= 0 < x2 and x3 roughly exceeds x2, or if
x1 > 0 and x2 rose above 0.87·x1 and x3 rose about 2x further than that;
otherwise answer 0." Answering 1 for increasing triples x1 < x2 < x3 is
correct (median is x2). Everything else defaults to 0.

Errors (Gaussian measure, from the integration in estimate.txt; sampled
counts agree):
- **Median at position 2 (1/3 of inputs): always wrong** (class 2 never output;
  mostly answered 0, ~2% answered 1).
- **Median at 1 (1/3): mostly wrong.** Decreasing triples x1 > x2 > x3 always get
  0; increasing triples are caught only if x3 passes the thresholds above
  (and for x1 > 0 only when x3 > 1.986 (x2 - 0.87 x1), below 20x that). About
  0.079 of all inputs are med=1 answered 1.
- **Median at 0 (1/3): mostly right** (default answer). It is wrong only when
  the rising rule fires with x1 as median. For x1 <= 0 that is impossible
  (x1 median would need x3 < x1 <= 0, but the rule needs x3 > 0.87 x2 > 0).
  For x1 > 0 it happens when x2 < x1 < x3 with x2 inside the 13% slack
  (0.87 x1 < x2 < x1) and x3 > 1.986 (x2 - 0.87 x1): the factor r = 0.87 < 1
  makes the "rise" test slightly too lenient. About 0.3% of inputs.
- The same slack causes the med=2 answered-1 errors: x1 <= 0 < x2 and
  0.87 x2 < x3 < x2 (x3 is the median but the rule fires). About 0.7%.
- Net: accuracy = 1/3 + P(1 & med=1) - P(1 & med=0) = 1/3 + 0.0768 = 0.410.

## 5. Twin and what I could not explain

`twin.py` implements exactly the rule of section 3 with the named constants
R_RISE, R_GATE, C_LO, C_HI, M_X1, CAP_X2, CAP_X1 above. Agreement with the
model on 600k fresh Gaussian inputs: 99.997-99.999% (the residual
disagreements are floating-point ties on decision boundaries, e.g. x3 = 0.8702 x2
to 4 digits, because the twin's constants are rounded).

Not explained: *why* training landed on this degenerate solution (a 2-unit
bias-free RNN seemingly found a local optimum that solves "median at 0 or
increasing triple" and gave up on position 2). The far cap (CAP_X2/CAP_X1)
and the x2 > 79 x1 branch are mechanistically derived but carry negligible
probability (~0.2%) and have no meaningful "median" interpretation.
