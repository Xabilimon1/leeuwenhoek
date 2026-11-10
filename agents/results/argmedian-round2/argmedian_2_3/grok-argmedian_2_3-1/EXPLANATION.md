# How the 2-unit ReLU RNN finds (some) medians

The network maps `x = (x0, x1, x2)` to 3 logits and takes argmax.
It **never predicts position 2** (except measure-zero logit ties).
It predicts position 1 on two conical “ascent” regions described below,
and predicts position 0 everywhere else.

On i.i.d. standard Gaussian inputs this is a weak median-position
detector: it catches many of the cases where `x1` is the median of an
upward trend, defaults to 0 otherwise, and has no channel for answering 2.

Notation: hidden state `h_t = (u_t, v_t)`, `relu(z) = max(z, 0)`.
Weights, as loaded from `weights.npz`:

```
w_ih = (a, b) = (2.9500925540924072, 3.1528589725494385)
w_hh = [[A, B],
        [C, D]]
      = [[-76.49169921875,    -2.504887580871582 ],
         [  1.192503571510315, -1.9860576391220093]]
w_out rows: class 0 (2.84549475, -2.49234843)
            class 1 (-3.75295782,  3.44526887)
            class 2 ( 0.83500987, -0.92129105)
```

The recurrence is `h_t = relu(w_ih * x_{t-1} + w_hh @ h_{t-1})`
with `h_0 = 0` (0-based inputs `x0,x1,x2`).

---

## Constants derived from the weights

Every threshold used later is a function of these weights.

**Comparison ratio `k`** (when the previous hidden state is the
“fresh positive” vector `w_ih * x_prev`):

```
k = -(w_hh[1] · w_ih) / w_ih[1]
  = -(C a + D b) / b
  = 0.8702462636
```

Then `v_t > 0` iff `x_t > k x_{prev}` (given `x_{prev} > 0` and both
units on at the previous step). Equivalently
`k = -C (a/b) - D`.

**Unit-0 carry-over ratio `K_u`** (almost never met for Gaussians):

```
K_u = -(w_hh[0] · w_ih) / w_ih[0]
    = -(A a + B b) / a
    = 79.16875329
```

`A ≈ −76.5` is so negative that if `u` fired on the previous step,
`u` is crushed to 0 unless the new input is ~79× larger. For
N(0,1) inputs this is a thin wedge of probability
`(π/2 − arctan(K_u))/(2π) ≈ 0.002` when the previous input is
positive, and we ignore it in the twin.

**v-self-inhibition scale `d`:**

```
d = −D = −w_hh[1,1] = 1.9860576391
```

When `u_{t-1} = 0` and `v_{t-1} = b s` (`s` an “excess” defined
below), `v_t > 0` iff `x_t > d s`.

**Readout ratio `r★`** at which logit 0 equals logit 1:

```
r★ = (W11 − W01) / (W00 − W10) = 0.8998499634
```

If `v3 > 0` and `u3 / v3 < r★`, class 1 wins; otherwise class 0
wins. Class 2 never wins (see readout).

**Upper band factor `U`** (class-1 / class-0 crossing when only `v`
is carrying state):

With `u2 = 0`, `v2 = b s`, `s = x1 − k x0`,

```
u3 = relu(a x2 + B v2),   v3 = relu(b x2 + D v2)
```

`u3/v3 < r★` (and `v3 > 0`) rearranges, because `a − r★ b > 0`, to
`x2 < U s` with

```
U = b (r★ D − B) / (a − r★ b) = 20.02710476
```

The matching lower edge is `x2 > d s` (`v3` just turning on).
Between `d s` and `g s` with `g = −B b / a = 2.677`, only `v3` is
on (so class 1 wins automatically). Between `g s` and `U s` both
units are on with ratio below `r★` (class 1 still wins). Above
`U s` the ratio exceeds `r★` and class 0 wins.

**Fresh-state upper factor `U_A`** (same crossing when the previous
state is `w_ih * x1`, i.e. both units on at step 2):

```
U_A = (r★ (C a + D b) − (A a + B b)) / (a − r★ b) = 2045.1457
```

So if `x0 ≤ 0` and `x1 > 0`, class 1 wins for `k x1 < x2 < U_A x1`.
`U_A ~ 10^3` is so large that for Gaussians the upper cut is
irrelevant; the twin omits it.

---

## What the hidden units do

Both input weights `a, b` are positive and similar (`a/b = 0.9357`).
A lone positive input, starting from `h = 0`, turns **both** units
on in that fixed ratio. A lone non-positive input leaves `h = 0`.

### Unit `u` (hidden 0): a non-repeatable “fresh positive” pulse

`A = w_hh[0,0] ≈ −76.5` forbids `u` from firing two steps in a row
unless the new input is huge. So `u_t` is on only when the current
input is positive **and** `u` was off at the previous step
(practically: the previous input was not substantially positive).

- After `x0`: `u1 = a relu(x0)`. On iff `x0 > 0`.
- After `x1`: if `x0 ≤ 0`, `u2 = a relu(x1)` (on iff `x1 > 0`).
  If `x0 > 0`, `u2 = 0` unless `x1 > K_u x0` (ignored).
- After `x2`: same logic applied to `(u2, v2)`.

The class-0 readout has **positive** weight on `u` and negative
weight on `v`. A fresh positive at the last step, with `u3/v3 = a/b
= 0.936 > r★`, votes for 0.

### Unit `v` (hidden 1): “current exceeds `k` times the remembered positive”

`D = w_hh[1,1] ≈ −1.99` is mild self-inhibition; `C = w_hh[1,0] ≈
+1.19` lets a previous `u` (a remembered positive) **raise** the
bar for the next `v`.

- From rest (`h = 0`): `v_t = b relu(x_t)`, same gate as `u`.
- From a fresh positive `h = w_ih x_prev` (`x_prev > 0`):
  `v_t > 0` iff `x_t > k x_prev`.
- From a v-only state `u = 0`, `v = b s`: `v_t > 0` iff `x_t > d s`.

So `v` is a one-sided running comparison: it stays on only if the
new input is large enough relative to what has already been seen.

### Unrolled cases that matter for Gaussians

Write `s = x1 − k x0`.

**Step 1.** `h1 = 0` if `x0 ≤ 0`, else `h1 = w_ih x0`.

**Step 2.**

- If `x0 ≤ 0`: `h2 = 0` if `x1 ≤ 0`, else `h2 = w_ih x1`
  (both on, “fresh”).
- If `x0 > 0` and `x1 ≤ k x0`: `h2 = 0` (comparison failed).
- If `x0 > 0` and `k x0 < x1 < K_u x0`: `u2 = 0`,
  `v2 = b s > 0` (v-only memory of the excess).

**Step 3 and the prediction.** If `h2 = 0`, then either `h3 = 0`
(all logits 0 → argmax 0) or `h3 = w_ih relu(x2)` with ratio
`a/b > r★` → class 0. So **any history that zeros `h2` is
answered 0**, regardless of `x2`.

The only ways to answer 1 are the two regions where `v3` is on and
`u3/v3 < r★`:

**Region A** (`h2` is a fresh positive):
`x0 ≤ 0`, `x1 > 0`, `x2 > k x1`
(and, strictly, `x2 < U_A x1`, omitted).

**Region B** (`h2` is v-only):
`x0 > 0`, `x1 > k x0`, `d s < x2 < U s`.

Everywhere else, including every input whose median is at
position 2, the network answers 0.

---

## How the readout chooses 0 vs 1 (and never 2)

Logits are `W @ (u3, v3)`. If both hidden units are off, all three
logits are 0 and argmax returns 0 (the default).

If only `u3 > 0`: class 0 has the unique positive logit
(`W00 > 0 > W10`, and `W00 > W20`). Predict 0.

If only `v3 > 0`: class 1 has the unique positive logit
(`W11 > 0 > W01, W21`). Predict 1.

If both are on, the decision depends only on `r = u3/v3`:

```
L0 > L1  ⇔  r > r★ = 0.89985
L0 > L2  ⇔  r > 0.78143 = (W21 − W01)/(W00 − W20)
L1 > L2  ⇔  r < 0.95174 = (W21 − W11)/(W10 − W20)
```

These three cuts are inconsistent for class 2: `L2` would need
`r < 0.781` to beat class 0 and `r > 0.952` to beat class 1.
The class-2 row `(0.835, −0.921)` sits in the same `(+u, −v)`
quadrant as class 0 but with smaller scale, so it is strictly
dominated. **No activation pattern of `(u3, v3)` makes class 2
the unique argmax.**

Hence the network is a 0-vs-1 classifier, with class 1 exactly
when `v3 > 0` and `u3 < r★ v3`.

---

## The I/O rule (what `twin.py` implements)

Named constants: `k = 0.8702462636`, `d = 1.9860576391`,
`U = 20.02710476`.

```
predict(x) = 1  if  (x0 ≤ 0  and  x1 > 0  and  x2 > k x1)          # region A
              or (x0 > 0  and  x1 > k x0  and  d(x1 − k x0) < x2
                                          and  x2 < U (x1 − k x0)) # region B
           = 0  otherwise
```

This is the comparison rule above, not a copy of the matrix
recurrence. It ignores (i) unit-`u` carry-over `x1 > K_u x0`,
(ii) the `U_A ≈ 2045` cap on region A, (iii) the exact ReLU
kink at `x2 = g s` inside region B (prediction is 1 on both
sides of that kink).

---

## Where the median is wrong, and why

True median position of three distinct numbers:

- 0 iff `x0` lies between `x1` and `x2`,
- 1 iff `x1` lies between `x0` and `x2`,
- 2 iff `x2` lies between `x0` and `x1`.

Each occurs with probability `1/3` for i.i.d. continuous inputs.

**Always wrong when the true median is 2.** There is no output
channel for it. That is a third of all Gaussian inputs, and it is
the dominant error.

**Region A (`x0 ≤ 0 < x1`, `x2 > k x1`).** Here `x2 > 0 ≥ x0`, so
`x0` is never the median.

- True median is 1 iff `x2 > x1`. The network agrees.
  Probability: `P(x0 < 0, x1 > 0, x2 > x1) = 1/16`
  (trivariate orthant with correlations `(0,0,−1/√2)`).
- True median is 2 iff `k x1 < x2 < x1`. The network answers 1
  because `k < 1`: `v` turns on a little before `x2` actually
  overtakes `x1`. Probability `P(A) − 1/16`, with
  `P(A) = 1/8 + arcsin(−k/√(1+k²))/(4π) ≈ 0.06801`, so this
  band has measure ≈ `0.00551`. Already counted in the “true = 2”
  errors.

**Region B (`x0 > 0`, `x1 > k x0`, `x2` in `(d s, U s)`).** A
thinner cone (measure ≈ `0.0215`). Roughly, `x1` overshot `x0`
and `x2` then landed in a wide multiple of that overshoot.

- Often the true median really is 1 (an ascent `x0 < x1 < x2`,
  plus a small `x2 < x1 < x0` sliver). Those are extra correct
  answers, measure ≈ `0.01726`.
- Sometimes `x0` is still the median (`x1 < x0 < x2` or
  `x2 < x0 < x1` inside the band). The network answers 1, which
  is wrong. Measure ≈ `0.00297`.
- The rest have true median 2 (already counted).

**Missed true-1 cases (false 0).** The network has no detector
for the *descending* true-1 pattern `x2 < x1 < x0` except the
tiny part of it that falls in B. It also misses true-1 patterns
that fail the `k`-ascent tests (e.g. `x0 < 0 < x1` but `x2` not
quite `k x1`, or `x0 > 0` with `x1` not above `k x0`). All of
those are answered 0.

Qualitatively: the network implements “answer 1 if the sequence
looks like a one-sided ascent through position 1; otherwise
answer 0.” That is a strict subset of “position 1 is the
median,” plus a thin `k < x2/x1 < 1` over-trigger and the
region-B band.

---

## Accuracy, from the regions above

Let `R = A ∪ B` be the predicted-1 set of the rule (U-cap on B
included; `K_u` / `U_A` omitted). Then

```
accuracy = P(true = 0 and not R) + P(true = 1 and R)
         = 1/3 − P(B ∩ {true = 0}) + P(A ∩ {true = 1}) + P(B ∩ {true = 1})
         = 1/3 + 1/16 + 0.017259 − 0.002970
         = 0.41012
```

The two B-probabilities are 3-dimensional Gaussian measures of
polyhedral cones. Each cone is a Boolean combination of three
half-spaces through the origin, evaluated with the trivariate
orthant formula

```
P(n1·x > 0, n2·x > 0, n3·x > 0) = 1/8 + (∑_{i<j} arcsin ρ_{ij}) / (4π),
ρ_{ij} = n_i·n_j / (|n_i| |n_j|)
```

after splitting B along the median planes `x1 = x0`, `x2 = x1`,
`x2 = x0` and the auxiliary planes `x1 = m x0` with
`m = d k/(d−1)` (where `d s = x1`) and `x1 = (k + 1/d) x0`
(where `d s = x0`). The upper cut `x2 = U s` is subtracted as a
second cone with `d` replaced by `U`. Details of the splits are
in the derivation notes below.

Included in the 0.410: all true-2 errors; the region-A true-1
hits (`1/16`); region-B true-1 hits and true-0 false alarms,
with the `U` cap. Left out: `u`-carry-over (`K_u ≈ 79`) and the
`U_A ≈ 2045` tail of A, together a wedge of probability ~0.002
whose effect on accuracy is well below 0.001; ties of logits or
of input values (probability 0).

### Orthant splits used for B

Let `s = x1 − k x0`, `m = dk/(d−1) ≈ 1.7528`,
`p = k + 1/d ≈ 1.3738`. Write `⟨n1,n2,n3⟩` for the orthant
probability of those three normals.

**B without the U cap,** `B0 = {x0 > 0, s > 0, x2 > d s}`:
`⟨(1,0,0), (−k,1,0), (dk, −d, 1)⟩ ≈ 0.024477`.
U-tail `⟨(1,0,0), (−k,1,0), (Uk, −U, 1)⟩ ≈ 0.002947`.
`P(B) = P(B0) − P(U-tail) ≈ 0.021531`.

**`B0 ∩ {true = 1}`** = C1a + C1b + part2:

- C1a (`x0 < x1 < m x0`, `x2 > x1`) =
  `P(0 < x0 < x1 < x2) − ⟨(1,0,0),(−m,1,0),(0,−1,1)⟩`
  = `1/48 − 0.012769 = 0.008064`
- C1b (`x1 > m x0`, `x2 > d s`) =
  `⟨(1,0,0),(−m,1,0),(dk,−d,1)⟩ = 0.007802`
- part2 (`k x0 < x1 < x0`, `d s < x2 < x1`) =
  `(P(B0) − ⟨(1,0,0),(−1,1,0),(dk,−d,1)⟩)
   − (⟨(1,0,0),(−k,1,0),(0,−1,1)⟩ − 1/48) = 0.002607`

Sum ≈ `0.018473`. Subtracting the U-tail’s true-1 piece
(≈ `0.001214`) leaves `P(B ∩ {true = 1}) ≈ 0.017259`.

**`B0 ∩ {true = 0}`** = D1 + D2:

- D1 (`k x0 < x1 < x0`, `x2 > x0`) =
  `⟨(1,0,0),(−k,1,0),(−1,0,1)⟩ − ⟨(1,0,0),(−1,1,0),(−1,0,1)⟩`
  = `0.043926 − 1/24 = 0.002260`
- D2 (`x0 < x1 < p x0`, `d s < x2 < x0`) =
  `(⟨(1,0,0),(−1,1,0),(dk,−d,1)⟩ − ⟨(1,0,0),(−p,1,0),(dk,−d,1)⟩)
   − (1/24 − ⟨(1,0,0),(−p,1,0),(−1,0,1)⟩) = 0.002353`

Sum ≈ `0.004612`. Subtracting the U-tail’s true-0 piece
(≈ `0.001642`) leaves `P(B ∩ {true = 0}) ≈ 0.002970`.

---

## What this does not explain

- Why training produced *these* numbers (`a ≠ b`, `k ≈ 0.87`
  rather than 1, a live but powerless class-2 row). The
  mechanics above take the weights as given.
- The exact prediction on the `K_u` wedge `x0 > 0`, `x1 > 79 x0`
  (and the symmetric `x0 ≈ 0` strip where `u2` turns on). The
  network then behaves more like region A than region B. We did
  not put that exception into the twin.
- Behaviour far off the Gaussian scale (inputs of size ≫ 1, or
  ratios ~`K_u`, ~`U_A`), where the huge `A` and the `U_A` cap
  become visible.
- Softmax / logit *margins*; only argmax is explained.
