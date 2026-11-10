# How the network picks a position

The network answers **0 or 1, and never 2**. It keeps a one-step memory of a positive **anchor** and answers 1 only when a later input leaves a comparison residual in a specific band. An empty state and a fresh anchor both read out as 0, so 0 is the default.

Checked against the trained network on 1e5 standard normal inputs and on several thousand inputs in the rare wedge `x2 > K x1`: the rule below matched every argmax.

## Weights

`w_ih = (a_u, a_v)`, `w_hh = [[w_uu, w_uv], [w_vu, w_vv]]`, and `w_out` has rows `o_0, o_1, o_2` for the three logits. Hidden unit 0 is called **u** (the anchor marker) and unit 1 is called **v** (the comparison residual). There are no biases, and `h_0 = 0`, so

```
h_t = relu(a * x_t + w_hh @ h_{t-1}),    logits = w_out @ h_3.
```

The float32 weights, written as exact float64 values, are

```
a_u =  2.9500925540924072     a_v =  3.1528589725494385
w_uu = -76.49169921875        w_uv = -2.504887580871582
w_vu =  1.192503571510315     w_vv = -1.9860576391220093
o_0 = [ 2.8454947471618652, -2.4923484325408936]
o_1 = [-3.752957820892334,   3.4452688694000244]
o_2 = [ 0.8350098729133606, -0.9212910532951355]
```

## Readout: class 2 is unreachable

Both hidden units are nonnegative. Logit `i` beats logit `j` when `h · (o_i - o_j) > 0`.

- Logit 1 beats logit 0 when `u / v < r*`, with
  `r* = -(o_1 - o_0)_v / (o_1 - o_0)_u = 0.8998499634122318`,
  provided `v > 0`. If `v = 0` and `u > 0`, logit 0 is the largest. If `h = 0`, all three logits are 0 and argmax returns 0.
- Logit 2 beats logit 0 only when `u / v < 0.7814320810709907`.
- Logit 2 beats logit 1 only when `u / v > 0.9517416455635768`.

Those two half-planes do not meet anywhere in `u ≥ 0, v ≥ 0`. **Class 2 is never the argmax.** Whenever the median really sits at position 2 (probability 1/3 for continuous i.i.d. inputs), the network is wrong. I can see this in the weights; I cannot say why training left `o_2` inside that dead cone.

A pure anchor, derived next, has ratio `α = a_u / a_v = 0.9356880785907554`. Since `α > r*`, a fresh anchor reads out as **0**. A v-only state (`u = 0, v > 0`) reads out as **1**.

## What the hidden state stores

**Step t = 1.** `h = relu(a x_1)`.

- `x_1 ≤ 0`: state is empty, `(0, 0)`.
- `x_1 > 0`: both units turn on, `h = a x_1`. This is an **anchor** of value `s = x_1`. The two units carry the same number; the ratio is `α`.

**Leaving an anchor `s`.** The pre-activation is `a x + (w_hh @ a) s`.

```
w_hh @ a = [-233.555149599689, -2.743763740465]
```

Unit v stays positive when `x > R s`, and unit u stays positive when `x > K s`, with

```
R = 2.743763740465 / a_v = 0.8702462635829092
K = 233.555149599689 / a_u = 79.16875329070548
```

Equivalently `R = -w_vu α - w_vv` and `K = -w_uu - w_uv / α`. So, from an anchor `s`:

| new input `x` | next state |
|---|---|
| `x ≤ R s` | empty. The anchor is erased. |
| `R s < x ≤ K s` | **v-only**. `u = 0` and `v = a_v (x - R s)`. The margin `m = x - R s` is what v remembers. |
| `x > K s` | both units stay on: `u = a_u (x - K s)`, `v = a_v (x - R s)`. This is rare under a standard normal (see below). |

`K ≈ 79` because `w_uu ≈ -76.5`: the anchor marker inhibits itself so hard that it dies on the next step unless the new input is enormous. `R ≈ 0.87` is a soft threshold, not a true `≥`.

**Leaving a v-only state with margin `m`.** `v_prev = a_v m`, so

```
pre_v = a_v x + w_vv a_v m = a_v (x - λ m),    λ = -w_vv = 1.9860576391220093
pre_u = a_u x + w_uv a_v m
```

v survives when `x > λ m`. The readout ratio `u / v = r*` is crossed at

```
ρ = a_v (r* w_vv - w_uv) / (a_u - r* a_v) = 20.02710475698141
```

and `ρ > λ`, so the final state reads as **1** exactly on the band `λ m < x < ρ m`. Below the band the state is empty (answer 0). Above it, u has grown past `r* v` and the answer goes back to 0.

**Leaving a fresh anchor, the large-input end.** For `x > K s` the ratio `u / v` climbs from 0 at `x = K s` toward `α`. It hits `r*` at

```
T = (α K - r* R) / (α - r*) = 2045.145711973249
```

So a fresh anchor `s` followed by `x` reads as 1 for `R s < x < T s`, and as 0 outside that. `T` is large enough that `x > T s` almost never happens for standard normals; it is kept because it is part of the same ratio test.

## The rule, unfolded over three steps

Answer **1** in exactly these cases, and **0** otherwise.

1. **Fresh anchor at step 2.** `x_1 ≤ 0`, `x_2 > 0`, and `R x_2 < x_3 < T x_2`.
   The first input left the state empty, `x_2` was stored as the anchor, and `x_3` landed in the band where v is on and u is not yet past `r*`.

2. **V-only margin.** `x_1 > 0`, `R x_1 < x_2 ≤ K x_1`, and `λ m < x_3 < ρ m` with `m = x_2 - R x_1`.
   `x_2` beat the anchor but did not revive u. `x_3` then fell in the margin band.

3. **Rare distorted anchor.** `x_1 > 0`, `x_2 > K x_1`, and
   `R x_2 + C_LO x_1 < x_3 < T x_2 + C_HI x_1`, where
   ```
   C_LO = w_vu α K - λ R = 86.60903626931152
   C_HI = (k_u K + k_m R) / (a_u - r* a_v) = -160343.54391220692
   k_u = a_u (w_uu - r* w_vu),   k_m = a_v (w_uv - r* w_vv)
   ```
   This is the same `v > 0` and `u < r* v` test, starting from the both-on state of step 2. The wedge `x_1 > 0, x_2 > K x_1` has probability `arctan(1/K) / (2π) ≈ 0.00201`.

Ties on any of these hyperplanes have probability 0 under a non-degenerate Gaussian. On a tie the network's argmax returns the smaller index, which agrees with using strict inequalities for "answer 1".

### Two walks

`x = (1, 2, 3)`. Anchor `s = 1`. `x_2 / s = 2` sits between `R` and `K`, so the state goes v-only with `m = 2 - R = 1.12975`. The band is `(λ m, ρ m) ≈ (2.244, 22.63)`. `x_3 = 3` is inside, so the answer is 1. The median is 2, at position 1. Correct. The hidden state at the end is `(u, v) = (0, 2.384)`, and the logits are about `(-5.94, 8.21, -2.20)`.

`x = (1, 2, 2.1)`. Same margin, but `2.1 < 2.244`, so v dies, the state is empty, and the answer is 0. The median is still 2, at position 1. Wrong: the lower edge of the band sits above `x_2` itself (`λ m > x_2` once `x_2 / x_1` is large).

`x = (-1, 0.5, 0.2)`. Anchor `s = 0.5` after step 2. `R s ≈ 0.435 > 0.2`, so step 3 erases the state and the answer is 0. Order is `-1 < 0.2 < 0.5`, median at position 2. Wrong, and class 2 was never available.

## Where this differs from the median

For three distinct numbers the median position is 1 when `x_2` lies between `x_1` and `x_3`, and 0 when `x_1` lies between the other two. The network only approximates the first test, and only from one side.

**It is right on most inputs whose median is at 0.** Answering 0 is correct exactly then, and the answer-1 bands rarely cover those inputs. The only false alarms are a thin set of positive triples with `τ x_1 < x_2 < η x_1` (about `0.920 x_1` to `1.374 x_1`; the constants `τ, η` are defined below) whose `x_3` falls inside the margin band while `x_1` is still the middle value. That set has probability **0.00297040**.

**It catches the median at position 1 only in fragments.**

- The clean fragment is `x_1 < 0 < x_2 < x_3`. Then case 1 fires (`x_3 > x_2 > R x_2`) and `x_2` really is the median. Probability of this cone is exactly `1/16`, minus the cap `x_3 > T x_2` of mass about `3.9 × 10^{-5}`. This single cone is most of the network's correct 1's (measure `0.06246109` out of `0.07980833`).
- The rest of the correct 1's (measure `0.01734724`) come from the margin band and the rare wedge, on inputs where `x_2` happens to lie between `x_1` and `x_3`. The band is `λ (x_2 - R x_1) < x_3 < ρ (x_2 - R x_1)`, which is not the half-line `x_3 > x_2` or `x_3 < x_2`. It starts too high when `x_2` is several times `x_1` (the `(1, 2, 2.1)` miss) and it also fires for some `x_3` that sit between `x_1` and `x_2`, where the median is at position 2 or 0.

**Patterns it systematically misses.**

- Any input whose median is at position 2. Includes the whole open slab `min(x_1, x_2) < x_3 < max(x_1, x_2)`.
- Both of the first two inputs nonpositive. The state is still empty at step 2, step 3 can only build a fresh anchor, and a fresh anchor reads as 0. Correct only when `x_1` is the median.
- `x_1 > 0` and `x_2 ≤ R x_1`. The anchor is erased, and the answer is 0 regardless of `x_3`. Every decreasing triple `x_3 ≤ x_2 ≤ x_1` in this wedge has its median at position 1, so the answer is wrong. (The other side, `x_2 ≤ x_1 ≤ x_3`, has its median at position 0, and answering 0 is correct.)
- `x_1 ≤ 0 < x_2` but `x_3 ≤ R x_2`. The anchor is erased at step 3. If `x_3 ≤ x_1` the median is at 0 and the answer is right; if `x_1 < x_3 ≤ R x_2` the median is at 2 and the answer is wrong.

## Accuracy on standard normal inputs

For i.i.d. continuous inputs the median position is uniform on `{0, 1, 2}`, so `P(median at 0) = 1/3`. Answering 0 is correct on that event and wrong on the other two. Each time the network instead answers 1:

- it **gains** a correct answer if the median is at 1,
- it **loses** a correct answer if the median is at 0,
- it stays wrong if the median is at 2 (both answers are wrong).

So

```
accuracy = 1/3 + P(answer 1 and median at 1) - P(answer 1 and median at 0).
```

Both probabilities are Gaussian measures of cones cut out by the rule above. A standard normal vector in the `(x_1, x_2)` plane has angle `θ` uniform on `[0, 2π)` and an independent Rayleigh radius `s`. Every threshold in the rule is linear through the origin, so conditionally on `θ` the success set in `x_3` is a union of intervals whose endpoints scale with `s`. The radius integral is closed form:

```
H(β) = ∫_0^∞ s exp(-s^2/2) Φ(β s) ds = (1/2) (1 + β / sqrt(1+β^2)),
```

with `H(+∞) = 1` and `H(-∞) = 0`. The accuracy contribution of an angle is a sum of values of `H` at the scaled endpoints, and the probability is that quantity averaged over `θ`.

The integrand is smooth between the rays where the combinatorial shape of the band changes. Those rays are the sign axes, the line `x_2 = x_1` in the third quadrant, and, for `x_1 > 0`, the ratios

```
R   = 0.870246263583     anchor erased below this
ν   = ρ R / (ρ - 1) = 0.915983451384    upper edge U equals x_2
τ   = R + 1/ρ     = 0.920178593400    upper edge U equals x_1
η   = R + 1/λ     = 1.373756323060    lower edge L equals x_1
μ   = R λ / (λ - 1) = 1.752797373230  lower edge L equals x_2
K   = 79.168753290705    u fails to die
```

(`L = λ (x_2 - R x_1)`, `U = ρ (x_2 - R x_1)`.) Adaptive quadrature on each sector, absolute tolerance `1e-13`, puts the two probabilities at

```
P(answer 1, median at 1) = 0.079808331378
P(answer 1, median at 0) = 0.002970402985
```

and therefore

```
accuracy = 1/3 + 0.079808331378 - 0.002970402985 = 0.41017126.
```

Equivalently, as a sum of error regions (this is the number in `estimate.txt`):

| error region | probability |
|---|---|
| median at position 2 (class 2 dead) | `1/3` exactly |
| median at position 1, answer 0 | `1/3 - 0.079808331378 = 0.253525001955` |
| median at position 0, answer 1 | `0.002970402985` |
| **total error** | **0.58982874** |
| **accuracy** | **0.41017126** |

Nothing in the rule was dropped. Quadrature error on the angular integrals is below `1e-11`.

Pieces with a closed form, used as checks on those integrals:

- `x_2 < x_1 ≤ 0`. Answer 0, correct iff `x_3 > x_1`. With `u = Φ(x_1)`, the probability is `∫_0^{1/2} u(1-u) du = 1/12`.
- `x_1 < x_2 ≤ 0`. Answer 0, correct iff `x_3 < x_1`. Probability `∫_0^{1/2} u(1/2 - u) du = 1/48`.
- `x_2 ≤ 0 < x_1`. The anchor is erased and the answer is 0, correct iff `x_3 > x_1`. Probability `(1/2) ∫_{1/2}^1 (1-u) du = 1/16`.
- `x_1 < 0 < x_2 < x_3`. Answer 1 and the median is at 1. Probability `(1/2) ∫_0^∞ φ(x)(1-Φ(x)) dx = 1/16`, minus `P(x_1 < 0, x_2 > 0, x_3 > T x_2) ≈ 3.89 × 10^{-5}`.

The remaining mass, about `0.11854`, is the both-positive part of the plane (including the rare wedge, which contributes `0.00120652` to the accuracy). Its sector contributions, in the ratio order above, are `0.01857382, 0.00136624, 0.00016255, 0.00221749, 0.02123563, 0.01608082, 0.05770045, 0.00120652`.

## What is fully accounted for, and what is not

The map from `(x_1, x_2, x_3)` to the argmax is accounted for: every ReLU mode of the two units collapses to empty, fresh anchor, v-only, or the rare both-on state, and the readout on those states is the band test above. Every constant in `twin.py` is one of `R, K, T, λ, ρ, r*, α, C_LO, C_HI`, with the formula given in the section it comes from.

What the weights do **not** reveal is why training stopped at this partial comparator. A true median needs to be able to say 2, and it needs a threshold of 1 rather than `R ≈ 0.87` and a band `(λ m, ρ m)` rather than a half-line. The dead class-2 row and the soft threshold are facts about this particular weight vector. They are not forced by the architecture: the same recurrence could represent other rules. I have no account of the loss landscape that selected this one, beyond the observation that always answering 0 already scores `1/3` and the comparison band adds about `0.077` by catching part of "median at 1".
