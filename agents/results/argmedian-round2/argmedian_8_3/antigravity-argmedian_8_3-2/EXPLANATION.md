# Mechanistic Explanation of the Argmedian Recurrent Neural Network

## 1. Executive Summary & Core Invariants

The model is a 3-step, 8-hidden-unit Elman-style recurrent neural network with ReLU activations and no biases:
$$h_0 = 0$$
$$h_t = \text{relu}(w_{ih} x_t + w_{hh} h_{t-1}) \quad \text{for } t \in \{1, 2, 3\}$$
$$\text{logits} = w_{out} h_3$$

The task is to identify the index ($0, 1$, or $2$) of the median of three real inputs $(x_0, x_1, x_2)$ sampled i.i.d. from $\mathcal{N}(0, 1)$.

Two fundamental mathematical invariants govern the network's operation:

1. **Positive Scale Invariance (Homogeneity):**
   Because there are zero bias vectors in all layers, scaling the input by any positive scalar $c > 0$ scales every hidden state and the output logits by $c$:
   $$h_t(c x) = c \cdot h_t(x), \quad \text{logits}(c x) = c \cdot \text{logits}(x)$$
   Therefore, $\text{argmax}(\text{logits}(c x)) = \text{argmax}(\text{logits}(x))$. The decision regions are scale-invariant cones (or wedges) originating at the origin $(0, 0, 0)$ in $\mathbb{R}^3$.

2. **Shift Cancellation (Approximate Translation Invariance):**
   The true median position is invariant under shifting all inputs by a constant $c \cdot (1, 1, 1)$. Remarkably, the trained network implements this by mapping the constant direction $(1, 1, 1)$ into the approximate null space of the end-to-end network:
   $$\|h_3(c, c, c)\| \approx 0 \implies \text{logits}(c, c, c) \approx 0$$
   On a canonical constant input $x = (1, 1, 1)$, $h_3 = [0.04, 0.00, 0.03, 0.00, 0.00, 0.00, 0.18, 0.00]^T$ and logits are $[0.89, 0.18, -1.13]^T$, which is negligible compared to typical logit magnitudes ($\sim 100$).
   Consequently, the network computes almost entirely in the 2D subspace orthogonal to $(1, 1, 1)$, where the logits reflect pairwise differences $x_i - x_j$.

---

## 2. Hidden Unit Representations Across Time Steps

### Step 1 ($t = 1$, Input $x_0$)
At $t = 1$, $h_0 = 0$, so $h_1 = \text{relu}(w_{ih} x_0)$.
The input weight vector $w_{ih}$ is:
```
h0: +6.5779    h1: -5.5215    h2: +6.1547    h3: +16.2079
h4: -6.3550    h5: +1.4493    h6: -9.8721    h7: +2.4045
```
Because $w_{ih}$ contains both positive and negative entries, the network splits $x_0$ into positive and negative rectifier channels:
- **Positive channel ($x_0 > 0$):** Units **{0, 2, 3, 5, 7}** activate with values proportional to $x_0$.
- **Negative channel ($x_0 < 0$):** Units **{1, 4, 6}** activate with values proportional to $|x_0| = -x_0$.

Units $\{2, 4\}$ have almost equal and opposite weights ($+6.15$ vs $-6.36$). Similarly, $\{0, 1\}$ have weights $+6.58$ vs $-5.52$, and $\{3, 6\}$ have large magnitude weights $+16.21$ vs $-9.87$.

---

### Step 2 ($t = 2$, Input $x_1$)
At $t = 2$, the pre-activation vector is $z_2 = w_{ih} x_1 + w_{hh} h_1(x_0)$.
Since $h_1(x_0)$ is piecewise linear with a single knot at $x_0 = 0$, $w_{hh} h_1(x_0) = a_{\text{pos}} x_0$ when $x_0 > 0$, and $w_{hh} h_1(x_0) = a_{\text{neg}} x_0$ when $x_0 < 0$.

Computing $z_2$:
- **Pair $\{3, 6\}$ — The Primary Difference Detector $\pm(x_1 - x_0)$:**
  - For $x_0 > 0$:
    $$z_2[3] = 16.208 (x_1 - 0.715 x_0)$$
    $$z_2[6] = -9.872 (x_1 - 1.019 x_0)$$
  - For $x_0 < 0$:
    $$z_2[3] = 16.208 (x_1 - 0.317 x_0)$$
    $$z_2[6] = -9.872 (x_1 - 0.319 x_0)$$
  *Mechanistic role:* When $x_1 > x_0$, Unit 3 is strongly active and Unit 6 is zero. When $x_1 < x_0$, Unit 6 is strongly active (correlation $0.98$ with $\text{relu}(x_0 - x_1)$) and Unit 3 is zero. They form a complementary rectified pair encoding $\text{sign}(x_1 - x_0)$ and $|x_1 - x_0|$.

- **Pair $\{2, 4\}$ — The Sum/Anti-Difference Channel:**
  - For $x_0 > 0$: $z_2[2] = 6.155 (x_1 + 2.653 x_0)$ and $z_2[4] = -6.355 (x_1 + 2.651 x_0)$.
  - For $x_0 < 0$: $z_2[2] = 6.155 (x_1 + 2.213 x_0)$ and $z_2[4] = -6.355 (x_1 + 2.210 x_0)$.
  *Mechanistic role:* Unit 2 and Unit 4 have exactly matching slope ratios ($2.65$ for $x_0>0$ and $2.21$ for $x_0<0$). They encode a full linear feature $x_1 + 2.65 x_0$ across both positive and negative half-spaces.

- **Pair $\{0, 1\}$ — Scaled Asymmetric Difference:**
  - For $x_0 > 0$: $z_2[0] = 6.578 (x_1 - 2.271 x_0)$ and $z_2[1] = -5.522 (x_1 - 2.258 x_0)$.
  *Mechanistic role:* Activates when $x_1$ deviates substantially from $x_0$, providing non-linear slope adjustments for extreme values.

- **Unit 7:**
  Row 7 of $w_{hh}$ has massive negative entries on the active units from step 1 ($w_{hh}[7, 6] = -24.74$, $w_{hh}[7, 7] = -41.90$). As a result, $z_2[7]$ is strongly negative ($\le -113$). Unit 7 is shut down for steps $t \ge 2$ (active probability $< 0.004$).

---

### Step 3 ($t = 3$, Input $x_2$)
At $t = 3$, the network receives $x_2$ and recurrent input $w_{hh} h_2$. The pre-activation is $z_3 = w_{ih} x_2 + w_{hh} h_2$.
By regressing $z_3$ onto $(x_0, x_1, x_2)$, we uncover the high-level semantic features:

- **Pair $\{3, 6\}$ — Midpoint Comparison $\pm\left(x_2 - \frac{x_0 + x_1}{2}\right)$:**
  - $z_3[3] \approx 16.24 \left(x_2 - 0.56 x_0 - 0.48 x_1\right)$ (R² = 0.974)
  - $z_3[6] \approx -9.90 \left(x_2 - 0.48 x_0 - 0.55 x_1\right)$ (R² = 0.965)
  *Mechanistic role:* Unit 3 fires when $x_2$ is above the midpoint $\frac{x_0 + x_1}{2}$, while Unit 6 fires when $x_2$ is below the midpoint. Comparing $x_2$ to the midpoint of the earlier two inputs is the optimal continuous relaxation for sorting $x_2$ relative to the interval $[x_0, x_1]$.

- **Pair $\{2, 4\}$ — Extremum Detectors:**
  - $z_3[2] \approx 6.15 x_2 + 14.33 x_1 - 21.69 x_0$
  - $z_3[4] \approx -6.26 x_2 - 19.89 x_1 + 26.03 x_0$
  *Mechanistic role:* Signals whether $x_0$ is the extreme outlier relative to $x_1$ and $x_2$.

---

## 3. Readout and Decision Boundaries

The readout matrix $w_{out} \in \mathbb{R}^{3 \times 8}$ is:
```
Class 0: [+3.67, +3.45, -6.93,  +3.07, -7.25, +0.08, +5.32, -0.68]
Class 1: [-8.26, -5.37, +1.47,  +4.72, +4.55, -2.99, +2.66, +1.40]
Class 2: [+2.26, +0.39, +3.20, -12.21, -0.87, +2.13, -7.36, -0.22]
```

### Logit Formulation
The final prediction is $\text{argmax}_{c \in \{0, 1, 2\}} \text{logits}[c]$.
In each geometric sector:
1. **Median = 1 ($x_0 < x_1 < x_2$ or $x_2 < x_1 < x_0$):**
   Unit 3 and Unit 4 dominate. In $w_{out}$, Class 1 has positive weights on Unit 3 ($+4.72$) and Unit 4 ($+4.55$), whereas Class 0 and Class 2 have negative weights (e.g. $w_{out}[2, 3] = -12.21$). This pushes Logit 1 to $\sim +90$ and suppresses the others to negative values.
2. **Median = 2 ($x_0 < x_2 < x_1$ or $x_1 < x_2 < x_0$):**
   Unit 2 dominates ($w_{out}[2, 2] = +3.20$ vs $w_{out}[0, 2] = -6.93$). Logit 2 wins with $\sim +60$ to $+100$.
3. **Median = 0 ($1 < 0 < 2$ or $2 < 0 < 1$):**
   Unit 0 and Unit 6 dominate ($w_{out}[0, 6] = +5.32$ vs $w_{out}[2, 6] = -7.36$). Logit 0 wins with $\sim +95$ to $+125$.

---

## 4. Error Analysis: Where and Why the Model Fails

The true median decision boundaries in $\mathbb{R}^3$ are the pairwise equality hyperplanes:
$$x_0 = x_1, \quad x_1 = x_2, \quad x_2 = x_0$$

Inside each active linear region of the network, the map from input $x$ to logits is linear:
$$\text{logits} = M x = (w_{out} m_3 w_{hh} m_2 w_{hh} m_1 w_{ih}) x$$
where $m_1, m_2, m_3$ are diagonal binary matrices indicating active ReLUs.

The network's decision boundary between class $a$ and class $b$ is given by $(M[a] - M[b]) x = 0$.
Because training used finite gradient descent with finite data, the normal vector $v = M[a] - M[b]$ is **slightly tilted** relative to the true boundary normal:

$$v = c_i x_i + c_j x_j + c_k x_k = 0 \quad \text{where } c_i \approx -c_j \text{ and } c_k \ll c_i$$

### Systematic Error Modes
The errors occur when the true median is tied with another coordinate within a tiny boundary layer ($\Delta < 0.005$):
1. **Ordering $(1, 0, 2)$ ($x_1 < x_0 < x_2$, true median $0$):**
   Readout boundary normal between Logit 0 and Logit 2:
   $$L_0 - L_2 = -294.6611 x_0 + 0.2893 x_1 + 294.4317 x_2 = 0$$
   Normalizing gives:
   $$(x_2 - x_0) < \frac{0.2893}{294.4317} (x_0 - x_1) \approx 0.000982 (x_0 - x_1)$$
   When $x_2$ is above $x_0$ by less than $0.000982 (x_0 - x_1)$, the model prematurely flips to predicting **$2$ instead of $0$**.

2. **Ordering $(0, 2, 1)$ ($x_0 < x_2 < x_1$, true median $2$):**
   Readout boundary normal between Logit 2 and Logit 1:
   $$L_2 - L_1 = 0.2749 x_0 + 295.3334 x_1 - 295.4390 x_2 = 0$$
   Normalizing gives:
   $$(x_1 - x_2) < \frac{0.2749}{295.4390} (x_2 - x_0) \approx 0.000930 (x_2 - x_0)$$
   When $x_1$ is above $x_2$ by less than $0.000930 (x_2 - x_0)$, the model predicts **$1$ instead of $2$**.

3. **Ordering $(2, 1, 0)$ ($x_2 < x_1 < x_0$, true median $1$):**
   Readout boundary normal between Logit 1 and Logit 2:
   $$L_1 - L_2 = -0.0907 x_0 + 210.2266 x_1 - 209.9945 x_2 = 0$$
   Normalizing gives:
   $$(x_1 - x_2) < \frac{0.0907}{210.2266} (x_0 - x_1) \approx 0.000431 (x_0 - x_1)$$
   The model predicts **$2$ instead of $1$**.

4. **Ordering $(2, 0, 1)$ ($x_2 < x_0 < x_1$, true median $0$):**
   Readout boundary normal between Logit 0 and Logit 2:
   $$L_0 - L_2 = 204.6413 x_0 - 0.0269 x_1 - 204.4570 x_2 = 0$$
   Normalizing gives:
   $$(x_0 - x_2) < \frac{0.0269}{204.6413} (x_1 - x_0) \approx 0.000131 (x_1 - x_0)$$
   The model predicts **$2$ instead of $0$**.

---

## 5. Derivation of Constants

Each named constant in `twin.py` is derived analytically from the weight matrices $w_{ih}, w_{hh}, w_{out}$ and the ReLU activation pattern across each boundary:

For a given active path $(m_1, m_2, m_3)$:
$$M = w_{out} m_3 w_{hh} m_2 w_{hh} m_1 w_{ih}$$
The boundary hyperplane normal vector between class $a$ and class $b$ is $v = M[a] - M[b]$.
The relative gap threshold is the ratio of the competing coordinate's tilt coefficient to the main boundary coefficient:
$$\text{GAP} = \frac{|v_k|}{|v_j|}$$

Summary of derived constants:
| Constant | Target Ordering | Competing Classes | Analytical Ratio | Value |
| :--- | :--- | :--- | :--- | :--- |
| `GAP_02` | $(1, 0, 2)$ | Class 0 vs Class 2 | $0.2893 / 294.4317$ | **$0.000982$** |
| `GAP_21` | $(0, 2, 1)$ | Class 2 vs Class 1 | $0.2749 / 295.4390$ | **$0.000930$** |
| `GAP_12` | $(2, 1, 0)$ | Class 1 vs Class 2 | $0.0907 / 210.2266$ | **$0.000431$** |
| `GAP_02_REV` | $(2, 0, 1)$ | Class 0 vs Class 2 | $0.0269 / 204.6413$ | **$0.000131$** |

---

## 6. Derivation of Accuracy Estimate

Because inputs are standard Gaussian $X \sim \mathcal{N}(0, I_3)$ and all decision hyperplanes are translation-invariant along $(1, 1, 1)$ ($\sum_i v_i \approx 0$), the problem reduces to an isotropic 2D Gaussian in the plane orthogonal to $(1, 1, 1)$, spanned by orthonormal vectors:
$$v_1 = \frac{1}{\sqrt{2}}(1, -1, 0), \quad v_2 = \frac{1}{\sqrt{6}}(1, 1, -2)$$

In this plane, standard Gaussian measure is spherically symmetric ($\phi \sim \text{Uniform}[0, 2\pi]$).
Any angular error wedge with opening angle $\Delta \theta$ has Gaussian measure:
$$P(\text{wedge}) = \frac{|\Delta \theta|}{2\pi}$$

Projecting each boundary normal $v$ onto $(v_1, v_2)$ and calculating its angular tilt relative to the true 2D normal yields:
- Sector $(1, 0, 2)$ boundary: $\Delta \theta_{102} = 0.000792\text{ rad} \implies P = 0.000126$
- Sector $(0, 2, 1)$ boundary: $\Delta \theta_{021} = 0.000641\text{ rad} \implies P = 0.000102$
- Sector $(2, 1, 0)$ boundary: $\Delta \theta_{210} = 0.000568\text{ rad} \implies P = 0.000090$
- Sector $(2, 0, 1)$ boundary: $\Delta \theta_{201} = 0.000336\text{ rad} \implies P = 0.000053$
- Sector $(0, 1, 2)$ boundary: $\Delta \theta_{01, \max} = 0.000567\text{ rad} \implies P = 0.000090$
- Sector $(2, 0, 1)$ boundary: $\Delta \theta_{01, \min} = 0.000106\text{ rad} \implies P = 0.000017$

Total error probability:
$$P_{\text{error}} = \sum_{k=1}^6 \frac{\Delta \theta_k}{2\pi} = \frac{0.003010}{2\pi} \approx 0.000479$$
$$\text{Accuracy} = 1 - P_{\text{error}} \approx 0.99952$$

---

## 7. What Could Not Be Fully Explained

1. **Secondary ReLU transitions when $x_0 < 0$:**
   When $x_0 < 0$, the activation pattern at step 2 switches from $p_2 = 01100010$ to $p_2 = 01001010$ or $00001110$ depending on whether $x_1$ crosses $-0.32 x_0$. This creates small sub-facet creases in the decision boundaries with slightly altered tilt angles (ranging from $0.0007$ to $0.006$ radians) across a thin sub-wedge.
2. **Residual activations of Unit 5 and Unit 7 at step 3:**
   Unit 5 has a small active probability at step 3 ($\sim 39\%$) with small readout weight ($w_{out}[0, 5] = +0.08, w_{out}[2, 5] = +2.13$), but its exact role appears to be a minor corrective dampener rather than a primary logical gate.
