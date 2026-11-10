# How the network picks a position

The network answers the median position only in a narrow situation, and otherwise answers 0. On i.i.d. standard normal inputs its accuracy is about **0.410**. It **never answers position 2**.

It keeps a one-sided memory of positive numbers. A negative input writes nothing and tends to erase whatever was stored. From that memory it answers **1** when the newest input lands in a multiplicative band above the stored anchor, and answers **0** in every other case. The true median test ("is x2 between x1 and x3?") is translation invariant. This network is only invariant to scaling all three inputs by the same positive constant, so any ordering that lives on the non-positive side is invisible to it.

Notation. Inputs are `x1, x2, x3`. Hidden state `h = (h0, h1)` starts at 0 and stays nonnegative because of ReLU. Weights below are the float32 values in `weights.npz`, read as real numbers.

```
w_ih = (a, b) = (2.9500925540924072, 3.1528589725494385)

W = w_hh = [[ -76.49169921875,    -2.504887580871582 ],
            [  1.192503571510315,  -1.9860576391220093]]

O = w_out = [[  2.8454947471618652, -2.4923484325408936],
             [ -3.752957820892334,   3.4452688694000244],
             [  0.8350098729133606, -0.9212910532951355]]
```

Update: `h ← relu(w_ih * x_t + W h)`. Logits: `O h`. Prediction: argmax, with the smallest index winning a tie.

## Readout: class 1 iff h1 > R\* h0, and class 2 is impossible

For `h ≥ 0` the three logits are linear. Class 0 beats class 1 when

```
(O00 - O10) h0 + (O01 - O11) h1 > 0
```

with `O00 - O10 = 6.598452568054199` and `O11 - O01 = 5.937617301940918`. The boundary is the ray

```
R* = (O00 - O10) / (O11 - O01) = 1.1112963723507852.
```

So class 1 wins exactly when `h1 > R* h0`. The zero state ties all three logits at 0 and the argmax returns 0.

Class 2 never wins on `h ≥ 0`. Its two boundaries ask for incompatible ratios:

- logit 2 > logit 0 requires `h1/h0 > (O00 - O20)/(O21 - O01) = 1.2797017478850514`
- logit 2 > logit 1 requires `h1/h0 < (O20 - O10)/(O11 - O21) = 1.0507053092205993`

No nonnegative `h` is on both sides. The same check at the axes (`h0 = 0`, `h1 > 0` gives class 1; `h1 = 0`, `h0 > 0` gives class 0) covers the whole quarter-plane. **Position 2 is not in the range of this readout.** Every input whose median sits at index 2 is an error. That event has probability 1/3.

## What the two units store

Both entries of `w_ih` are positive, so from the zero state a negative input stays at zero and a positive input `u` writes the **balanced code**

```
h = u * (a, b),    ratio b/a = 1.068732222714759 < R*.
```

A sequence that ends on a balanced code answers 0. Meaning: a positive value was stored, and nothing later has confirmed it.

Let `d = W w_ih = (-233.555149599689, -2.743763740464999)`. Feeding a balanced code of value `u` into the next step contributes `u * d` to the preactivation. The next input `x` turns the units back on only above

```
K0 = -d0 / a = 79.16875329070548     (unit 0)
K1 = -d1 / b =  0.8702462635829092   (unit 1)
```

Three modes follow.

1. **Erased.** `x ≤ K1 u`. Both units hit 0. The anchor is gone.
2. **Residual.** `K1 u < x ≤ K0 u`. Unit 0 stays off (it would need `x` about 79 times `u`). Unit 1 stays on with activation

   ```
   v = b * (x - K1 u).
   ```

   The stored number is the surplus of `x` over 87% of the anchor, scaled by `b`.
3. **Overwritten.** `x > K0 u`. Both units turn on again. The new code is dominated by `x` but its ratio is still above `b/a` (infinite just above `K0`, falling toward `b/a` as `x/u → ∞`).

Unit 0 is a one-shot latch: its self-weight `W00 = -76.49169921875` is so negative that a balanced code suppresses unit 0 unless the next input is ~79 times larger. Unit 1 is the soft comparator: self-weight `W11 = -1.9860576391220093` lets it survive a modest step up, and `W10 > 0` lets unit 0 excite it.

### Next step from a residual `h = (0, v)`

Unit 1 stays on for `x > T1 v` and unit 0 turns on for `x > T0 v`, with

```
T1 = -W11 / b = 0.6299227642002839
T0 = -W01 / a = 0.8490877946852105
```

`T1 < T0`, so there is a range where only unit 1 is on (ratio `+∞`, answer 1), then a range where both are on. The ratio of that refreshed code equals `R*` at

```
ZS = (R* W01 - W11) / (b - R* a) = 6.352045851510871,
```

and for larger `x` the ratio falls below `R*` (limit `b/a`), so the answer goes back to 0. Therefore a residual answers 1 exactly on

```
T1 v < x < ZS v.
```

Substituting `v = b * gap` with `gap = x - K1 u` turns the band into a multiple of the surplus:

```
G_LO = -W11 = 1.9860576391220093
G_HI = ZS * b = 20.027104756981487
G_LO * gap < x < G_HI * gap.
```

### Next step from a balanced code of value `u`

The same ratio calculation, now with preactivation `u * d` instead of `(0, v)`, crosses `R*` at

```
ZC = (R* d0 - d1) / (b - R* a) = 2045.1457119732577.
```

Combined with the on-threshold `K1`, a balanced anchor answers 1 exactly when

```
K1 u < x < ZC u.
```

(`K0` sits inside this interval. Crossing `K0` changes which units are on, but the answer stays 1 until `ZC`, because the ratio is still above `R*`.)

### Next step from an overwritten code

If the previous anchor was `x1 > 0` and the new input `x2 > K0 x1`, the state is the linear mix

```
h = x2 * w_ih + x1 * d.
```

The unit-1 threshold and the `R*` threshold are the same formulas as above applied to this `h`. Collecting coefficients gives the band

```
K1 x2 + C_LO_X1 x1  <  x3  <  ZC x2 + C_HI_X1 x1
```

with

```
C_LO_X1 = -(W_row1 · d) / b           = 86.6090362693115
C_HI_X1 = (R* (W_row0 · d) - W_row1 · d) / (b - R* a)
                                    = -160343.54391220762.
```

The coefficient of `x2` on the lower edge is `K1`, and on the upper edge is `ZC`, because those are what the formulas return on the `w_ih` part of `h`. The `x1` terms are the same formulas on the leftover `d`. As `x1/x2 → 0` the band collapses to `(K1 x2, ZC x2)`, the balanced-anchor rule. At `x2 = K0 x1` the two edges agree with `G_LO * gap` and `G_HI * gap`, so the residual case and the overwritten case meet.

## The three-step rule

`h` starts at 0, so step 1 only creates a balanced code of `x1` when `x1 > 0`. Step 2 then does one of the following, and step 3 reads it out.

**Anchor case.** `x1 ≤ 0` and `x2 > 0`. Step 1 wrote nothing, step 2 stored a balanced code of `x2`. Answer 1 iff

```
K1 * x2 < x3 < ZC * x2.
```

**Residual case.** `x1 > 0` and `K1 x1 < x2 ≤ K0 x1`. Step 2 stored the surplus `gap = x2 - K1 x1`. Answer 1 iff

```
G_LO * gap < x3 < G_HI * gap.
```

**Reset case.** `x1 > 0` and `x2 > K0 x1`. Both units re-fired. Answer 1 iff `x3` lies strictly between the two linear edges above.

**Everything else answers 0.** That includes: `x2` below `K1` times a positive `x1` (memory erased, and step 3 then writes at most a fresh balanced code, whose ratio `b/a` is below `R*`); both of the first two inputs non-positive; `x3` below the lower edge or above the upper edge.

`twin.py` is this rule with the seven constants above. It multiplies no weight matrix.

Worked checks.

- `(1, 2, 3)`. Positive anchor `u = x1 = 1`, `gap = 2 - K1 = 1.12975`, band `(2.244, 22.63)`. `3` sits inside, answer 1. The median is at index 1.
- `(1, 2, 100)`. Same residual, but `100` is above `22.63`, answer 0. The median is still at index 1. A huge last input refreshes both units into a near-balanced code and the readout falls back to 0.
- `(2, 1, 0)`. `1 < K1 * 2`, memory erased, `0` writes nothing, answer 0. The median is at index 1. A decrease through positive numbers looks like "anchor lost".
- `(0, 1, 2)`. No anchor at step 1, `x2 = 1 > 0`, `2` is inside `(K1, ZC)`, answer 1. Correct.
- `(-1, -0.5, 0.2)`. Nothing positive is stored before the last step. The last step writes a balanced code and answers 0. The median is at index 1.

## Where this differs from the median

The true index-1 event is "`x2` lies between `x1` and `x3`". The network instead asks for a chain of **positive, mostly increasing** ratios.

**Caught, anchor case.** If `x1 ≤ 0 < x2 ≤ x3` and `x3 < ZC x2`, the ordering really does put the median at index 1, and the anchor band contains it (`K1 < 1`, so `x3 ≥ x2` implies `x3 > K1 x2`). Probability of the cone `x1 ≤ 0 < x2 ≤ x3` is `1/16`. The cap `x3 ≥ ZC x2` removes only `3.89e-5`.

**False alarm, anchor case.** `x1 ≤ 0 < x2` and `K1 x2 < x3 < x2`. Then `x2` is the largest input. Since `x3 > K1 x2 > 0 ≥ x1`, one also has `x1 ≤ x3 ≤ x2`, so the median is at index 2. The network answers 1 and is wrong. Probability `0.005512`. There is no anchor-case input whose median is at index 0: that would need `x3 ≤ x1 ≤ 0` and `x3 > K1 x2 > 0` at once.

**Missed, non-positive middle.** Answer 1 requires `x2 > 0` in every case (the anchor is `x2`, or `x2 > K1 x1 > 0`, or `x2 > K0 x1 > 0`). Half of all index-1 medians have `x2 ≤ 0`, probability exactly `1/6`, and all of them are missed. This is the largest single miss. Adding a constant to a correctly ordered triple can push `x2` through zero and flip a correct 1 into a 0. Scaling by a positive constant cannot.

**Residual case, partial overlap.** For `x1 > 0` the acceptance window is the surplus band `G_LO * gap < x3 < G_HI * gap`, with `gap = x2 - K1 x1`. It is not a band around `x2`. The edges cross `x1` and `x2` at four ratios `x2/x1`:

- `G_HI * gap = x2` at `0.915983451384`
- `G_HI * gap = x1` at `0.920178593400`
- `G_LO * gap = x1` at `1.373756323060`
- `G_LO * gap = x2` at `1.752797373230`

What that does to the median:

- `K1 < x2/x1 < 0.91598`. The whole window lies below `x2`, and `x2 < x1`, so every accepted `x3` satisfies `x3 < x2 < x1`. The median is at index 1.
- `0.91598 < x2/x1 < 1`. Still `x2 < x1`, but the window now extends above `x2` and, past `0.92018`, above `x1`. Then `x3 < x2` is a true index-1 median, `x2 < x3 < x1` puts the median at index 2, and `x3 > x1` puts it at index 0.
- `1 < x2/x1 < 1.75280`. Now `x2 > x1` and the lower edge is still below `x2`. Then `x3 > x2` is a true index-1 median (`x1 < x2 < x3`), `x1 < x3 < x2` puts the median at index 2, and `x3 < x1` (possible while the lower edge stays below `x1`, i.e. up to ratio `1.37376`) puts it at index 0.
- `1.75280 < x2/x1 ≤ K0`. The whole window lies above `x2 > x1`, so `x1 < x2 < x3` and every answer 1 is an index-1 median.

Index-0 false alarms inside this band have total probability `0.002970`. The anchor band and the reset band contribute none: in both of those, answer 1 forces the median off index 0.

Index-1 inputs with `x1 > 0` and `x2 ≤ K1 x1` erase the anchor and are missed. A typical one is a decrease: `x1 > x2 > x3 > 0` with `x2` less than 87% of `x1` (for example `(2, 1, 0)`).

**Reset case.** The cone `x1 > 0`, `x2 > K0 x1` has probability `0.002010`. The answer-1 sub-band inside it has probability `0.000198`, of which `0.000195` is a true index-1 median. The lower edge stays above `x1` on this whole cone, so a median at index 0 does not occur here. Leaving the case out of the twin would move the agreement with the network by `0.000198` and the accuracy by `0.000195`. It is the same gate as the anchor band, evaluated on the mixed code `x2 * w_ih + x1 * d`.

**Position 0 is the default and is usually right.** False alarms onto position 0 have probability `0.002970`, so the network is correct on about `0.3304` of the `1/3` of inputs whose median is at index 0.

**Position 2 is always wrong.** Probability `1/3`. Answering 1 on a subset of them (the anchor false alarm, plus a thin residual slice) does not repair them.

## Accuracy from these regions

For continuous i.i.d. inputs, `P(median at k) = 1/3`. Correct predictions are "answer 0 and median at 0" plus "answer 1 and median at 1":

```
accuracy = 1/3 + P(answer 1, median at 1) - P(answer 1, median at 0).
```

The cones are orthants of trivariate centered normals after taking the three linear forms that cut each piece. With correlations `ρij`,

```
P(all three forms positive) = 1/8 + (arcsin ρ12 + arcsin ρ13 + arcsin ρ23) / (4π).
```

A band is a difference of two orthants. Splits and numbers are in `estimate.txt`. The result is

```
P(answer 1, median at 1) = 0.079808331378
P(answer 1, median at 0) = 0.002970402985
accuracy                  = 0.410171261727
```

Regions included: anchor, residual, and reset, each intersected with median-at-1 and median-at-0 as above. The whole median-at-2 event is the `1/3` that this formula treats as wrong. Equality cases (tied inputs, `x3` exactly on a band edge) have probability 0. No positive-probability piece of the rule was left out of the integral.

## What is fully determined, and what is not

The forward map is accounted for. Hidden state after three steps is always one of: zero, a balanced code, a unit-1 residual, or an overwritten two-unit code, and the readout on each of those is the band test above. The seven constants in `twin.py` are the ones derived in this note.

Two things are facts about these weights rather than consequences I can push any further.

- The numerical size of `K0 ≈ 79` and `ZC ≈ 2045`. Both are fixed by `W00 ≈ -76.5` and by where the ratio `h1/h0` crosses `R*`. On Gaussian inputs they almost never bind: the anchor cap removes `4e-5` of probability, and the reset cone is a few thousandths. They are real gates, and they are not where the accuracy is won or lost. I do not have an account of why training drove `W00` that far.
- Why the learned readout makes class 2 unreachable. The geometry above shows that it is unreachable. A different `w_out` could put class 2 on top of some cone of `h ≥ 0`. This one does not, so one third of all inputs are errors before any comparison is done. I am not explaining the training dynamics that settled there.

The comparison the network does implement is fully described by the band rule.
