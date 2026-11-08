# How the 4-unit ReLU RNN finds the median position

## 1. Structure in the weights

```
w_ih  = [ 3.97  4.70 -4.72 -3.97]
w_hh  = [[ 1.34 -2.20  2.64 -1.89]
         [ 1.93 -2.43  2.79 -2.38]
         [-2.39  2.81 -2.43  1.93]
         [-1.89  2.64 -2.19  1.34]]
w_out = [[ -1.67   1.60   1.62  -1.66]
         [-18.75  16.00  15.96 -18.72]
         [ 11.43 -15.74 -15.70  11.48]]
```

* **Sign symmetry.** Swapping units 0<->3 and 1<->2 maps `w_hh` to itself (to ~0.01), flips the sign of
  `w_ih`, and leaves every row of `w_out` unchanged. So the network computes the same logits for `x` and `-x`
  (median position is also invariant under negation). Units 0,1 are "positive" (P) units, 2,3 the "negative" (N)
  mirror copies.
* **No biases** => the network is positively homogeneous: logits(c x) = c logits(x) for c>0, so the answer
  depends only on the direction of x. Together with the sign symmetry, all thresholds below are *ratios*.
* **Readout uses only two sums.** Rows of `w_out` weight (h1+h2) and (h0+h3) almost equally within each pair:
  with S = h1+h2, T = h0+h3,
  `l1-l0 ≈ 14.4 S - 17.1 T`, `l2-l0 ≈ -17.3 S + 13.1 T`. Hence (S,T >= 0)
  answer 1 if S/T > 17.1/14.4 ≈ 1.19, answer 2 if S/T < 13.1/17.3 ≈ 0.76, answer 0 in between
  (and 0 also when all units are off: all logits are 0, argmax picks index 0).

## 2. What the hidden state represents

* **t=1:** if x1>0 only the P pair is on, h = x1·(3.97, 4.70, 0, 0); if x1<0 only the N pair, mirrored.
  The state is a signed copy of x1. Note h1/h0 = 4.70/3.97 = 1.18, i.e. it sits right at the 0/1 boundary.
* **t=2:** feeding back the P state gives recurrent input x1·(-5.03, -3.79, 3.71, 4.90)
  (= `w_hh[:, :2] @ (3.97, 4.70)`), so pre-activations are ≈ 4·(x2 - x1) on the P units and ≈ 4·(x1 - x2) on
  the N units (per-unit slopes 1.27, 0.81, 0.79, 1.23 instead of exactly 1). The state after step 2 is a signed
  encoding of the comparison x2 - x1: the P pair is active if x2 > x1, the N pair if x2 < x1
  (measured: >92% of inputs have exactly one pair active after step 2).
* **t=3:** the same subtraction is applied again with x3, and the ratio S/T of the final state encodes on which
  side of / between the earlier values x3 falls. The three readout zones of S/T correspond to the three median
  positions; on generic inputs the network answers the true median (≈98.5% of Gaussian inputs).

I did not derive the exact closed-form map from (x1,x2,x3) to S/T through the step-3 ReLU pattern; the
error-band widths below are therefore measured (fitted from model outputs), not derived from the weights.

## 3. Where it is wrong (and why)

All errors are on (near) ties, where the comparison signals that the network subtracts become tiny and the
imperfect (not exactly 1) slopes of the units dominate. Write, for a close pair (x_i, x_j) whose third value is
x_k, the relative gap r = sign(x_i - x_k)·(x_j - x_i)/|x_i - x_k|.

1. **x2 ≈ x3 (the dominant error, ~0.94%).** When x2 and x3 are closer to each other than to x1 and
   r ∈ [-0.018, 0.016] (gap under ~2% of their distance to x1), S/T lands in the middle zone and the network
   answers **0**, although the median is position 1 or 2. Mechanism: the final state is then dominated by the
   x2-x1 comparison carried from step 2, which (like the t=1 state) has S/T ≈ 1.2, in the "0" zone, instead of
   by the tiny x3-x2 comparison which should decide 1 vs 2.
2. **x1 ≈ x3, x2 extreme (~0.17%).** If x1 is just inside (median is x1) by a relative gap r ∈ [0, 0.0063],
   the network answers **2** – the tie is broken slightly in favour of the later input.
3. **x1 ≈ x2, x3 extreme (~0.05%).** Median is x1 by relative gap r ∈ [0, 0.0018]: answers **1**.
4. **Dead state (~0.16%).** For same-sign sequences that shrink slowly toward 0 (empirically
   0.53 < x2/x1 < 1 and 0.88 < x3/x2 < 1.08, e.g. (1.61, 1.32, 1.27)), every unit's pre-activation at t=3 is
   negative, all logits are 0 and argmax gives 0 (the median is 1 there). My box for this region is too wide;
   the true dead region is roughly half of it and I did not characterise its exact shape.

The band edges are soft (error rate goes 0.2 -> 0.9 over ~0.01 in r), presumably because the exact edge depends
on the second shape coordinate (where x_k sits relative to 0); I did not model that.

## 4. Twin and estimate

`twin.py` implements: true median, overridden by rules 1-4 with the constants above. Agreement with the model on
100k fresh Gaussian inputs: 99.22%.

`estimate.txt`: for each band, (a, e) = (x_i - x_k, x_j - x_i) is bivariate normal with var 2, cov -1, so
r = -1/2 + (√3/2)·Cauchy and P(r∈[l,u]) = [atan((u+½)/(√3/2)) - atan((l+½)/(√3/2))]/π. Dead region: integrate
the density of (s,t)=(x2/x1, x3/x2), f = |s|/(2π(1+s²+s²t²)^{3/2}), over the box. Total ≈ 0.0152 error,
accuracy ≈ 0.985.

## What I could not explain
* Exact derivation of band edges from the weights (they are fitted).
* The precise shape of the dead region and the soft edges / scattered errors just outside the bands
  (~0.8% of inputs where twin and model disagree).
