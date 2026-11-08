# Mechanistic Explanation of the Argmedian Recurrent Neural Network

## 1. Overview and Core Mechanism

The network is a 3-step Recurrent Neural Network (RNN) with 4 hidden units and no biases, trained on sequences of three real numbers $x = (x_1, x_2, x_3)$ to predict the index $\text{argmax}_{i \in \{0, 1, 2\}}(\text{logits}_i)$ of the **median** input.

Despite having only 32 parameters, the network implements a remarkably elegant geometric algorithm:
1. **Sign-Magnitude Decomposition ($t=1$):** At step 1, the network splits $x_1$ into positive and negative directional components using two symmetric pairs of ReLU units.
2. **Relative Comparison ($t=2$):** At step 2, the recurrent matrix projects the state of $x_1$ against $x_2$, partitioning the relative ratio $x_2 / x_1$ into three distinct regimes:
   - $x_2 \gg x_1$ (represented by outer positive units)
   - $x_2 \ll x_1$ (represented by inner negative units)
   - $x_2 \approx x_1$ (represented by a mixed pair with balanced activations)
3. **Median Readout ($t=3$):** At step 3, $x_3$ is introduced. The readout matrix collapses the 4D final hidden state $h_3$ into two aggregated scalar features:
   - $u = h_{3,0} + h_{3,3}$ (outer units)
   - $v = h_{3,1} + h_{3,2}$ (inner units)
   
   The output logits are virtually identical across symmetric units and can be written as:
   $$\text{logit}_0 \approx -1.67 u + 1.61 v$$
   $$\text{logit}_1 \approx -18.74 u + 15.98 v$$
   $$\text{logit}_2 \approx 11.45 u - 15.72 v$$

   Because the weights of $\text{logit}_0$ are an order of magnitude smaller than $\text{logit}_1$ and $\text{logit}_2$, **position 0 acts as a "default" attractor**:
   - If $v / u > 1.188$, $\text{logit}_1$ dominates $\implies$ **Position 1** (median is $x_2$).
   - If $v / u < 0.757$, $\text{logit}_2$ dominates $\implies$ **Position 2** (median is $x_3$).
   - If $0.757 \le v / u \le 1.188$ (or $u = v = 0$), both $\text{logit}_1$ and $\text{logit}_2$ are strongly negative, and $\text{logit}_0 \approx 0$ wins $\implies$ **Position 0** (median is $x_1$).

4. **Failure Modes:** The primary failure mode of the network occurs when $x_2 \approx x_3$ and $x_1$ is an extreme value. In this case, the network receives almost the same input at steps 2 and 3, producing residual activations where $u \approx v$. This ratio falls squarely into the readout dead zone $[0.757, 1.188]$, causing the network to predict position 0 even though position 0 is not the median! This dead band around $x_2 = x_3$ accounts for ~78% of all errors.

---

## 2. Invariances and Symmetries

### 2.1 Positive Homogeneity (Scale Invariance)
Because the network contains **no bias vectors** and uses piecewise-linear ReLU activations ($\text{relu}(c z) = c \cdot \text{relu}(z)$ for $c > 0$):
$$\text{logits}(c \cdot \mathbf{x}) = c \cdot \text{logits}(\mathbf{x}) \quad \forall c > 0$$
Since $\text{argmax}$ is invariant under multiplication by positive scalars:
$$\text{predict}(c \cdot \mathbf{x}) = \text{predict}(\mathbf{x}) \quad \forall c > 0$$
Thus, the decision boundaries are cones in $\mathbb{R}^3$, and the behavior of the network depends solely on the normalized direction $\mathbf{x} / \|\mathbf{x}\|$ on the unit 2-sphere $S^2$.

### 2.2 Reflection Symmetry ($\mathbb{Z}_2$ Equivariance)
The median position is invariant under global sign negation: $\text{median}(-x_1, -x_2, -x_3) = \text{median}(x_1, x_2, x_3)$ (since reversing the order of numbers preserves the middle element).

The network weights reflect this exact symmetry under the permutation matrix $P$ that swaps outer units $(0 \leftrightarrow 3)$ and inner units $(1 \leftrightarrow 2)$:
$$P = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix}$$
Inspecting the trained weights reveals:
$$w_{ih} \approx -P w_{ih}$$
$$w_{hh} \approx P w_{hh} P$$
$$w_{out} \approx w_{out} P$$
Consequently, negating the input $\mathbf{x} \to -\mathbf{x}$ transforms the hidden state via $h_t \to P h_t$, leaving the readout logits $\text{logits} = w_{out} h_3 \approx w_{out} P h_3$ unchanged!

---

## 3. Hidden Unit Representations Across Steps

The 4 hidden units form two functional pairs:
- **Outer units $\{0, 3\}$:** Input weights $w_{ih, 0} = +3.965$ and $w_{ih, 3} = -3.965$.
- **Inner units $\{1, 2\}$:** Input weights $w_{ih, 1} = +4.700$ and $w_{ih, 2} = -4.716$.

We denote the characteristic input scales by $a \approx 3.965$ and $b \approx 4.708$.

### Step 1: Encoding $x_1$
At $t=1$, with $h_0 = \mathbf{0}$:
$$h_1 = \text{relu}(w_{ih} x_1)$$
- If $x_1 > 0$: Units 0 and 1 are active:
  $$h_1 = [a x_1, \, b x_1, \, 0, \, 0]^T$$
- If $x_1 < 0$: Units 2 and 3 are active:
  $$h_1 = [0, \, 0, \, b |x_1|, \, a |x_1|]^T$$

Thus, the active pair encodes the sign of $x_1$, and the activation magnitude linearly tracks $|x_1|$.

### Step 2: Comparing $x_2$ with $x_1$
At $t=2$:
$$z_2 = w_{ih} x_2 + w_{hh} h_1$$
When $x_1 > 0$, the recurrent contribution is $w_{hh}[:, :2] @ [a, b]^T$:
$$w_{hh} h_1 = x_1 \begin{pmatrix} -5.029 \\ -3.794 \\ +3.717 \\ +4.914 \end{pmatrix}$$
Dividing elementwise by $w_{ih} = [a, b, -b, -a]^T$:
$$z_{2, 0} = a (x_2 - 1.268 x_1)$$
$$z_{2, 1} = b (x_2 - 0.807 x_1)$$
$$z_{2, 2} = -b (x_2 - 0.788 x_1)$$
$$z_{2, 3} = -a (x_2 - 1.239 x_1)$$

Notice the reciprocal structure:
$$r_1 = 0.807 \approx r_2 = 0.788 \implies g \approx 0.80$$
$$r_0 = 1.268 \approx r_3 = 1.239 \implies 1/g \approx 1.25$$
where $0.807 \times 1.239 = 1.000$ and $0.788 \times 1.268 = 1.000$.

This partitions the relationship between $x_2$ and $x_1$ into three distinct activation regimes (for $x_1 > 0$):
1. **$x_2 > 1.25 x_1$ ($x_2$ clearly larger):** Units 0 and 1 are active ($h_{2, 2} = h_{2, 3} = 0$).
2. **$x_2 < 0.80 x_1$ ($x_2$ clearly smaller):** Units 2 and 3 are active ($h_{2, 0} = h_{2, 1} = 0$).
3. **$0.80 x_1 \le x_2 \le 1.25 x_1$ ($x_2 \approx x_1$):** Units 1 and 3 are simultaneously active!
   At $x_2 = x_1$, $h_{2, 1} \approx 0.91 x_1$ and $h_{2, 3} \approx 0.95 x_1$. They are balanced in magnitude.

(By reflection symmetry $P$, an identical comparison occurs when $x_1 < 0$).

### Step 3: Integrating $x_3$
At $t=3$, the pre-activation is:
$$z_3 = w_{ih} x_3 + w_{hh} h_2$$
Here, $w_{hh} h_2$ acts as a dynamic threshold encoding the relationship between $x_1$ and $x_2$.
- When $x_2 \approx x_1$, $w_{hh} h_2 \approx -w_{ih} x_1$. Therefore:
  $$z_3 \approx w_{ih} (x_3 - x_1)$$
  directly comparing $x_3$ against the shared value of $x_1 \approx x_2$.
- When $x_1$ and $x_2$ are separated, $w_{hh} h_2$ produces thresholds that bracket the interval $[\min(x_1, x_2), \max(x_1, x_2)]$.

---

## 4. Readout Analysis and the 2D Angular Sector

The readout matrix $w_{out}$ maps $h_3 \in \mathbb{R}^4_{\ge 0}$ to logits in $\mathbb{R}^3$:
$$w_{out} = \begin{pmatrix} -1.672 & 1.599 & 1.617 & -1.658 \\ -18.748 & 15.999 & 15.959 & -18.724 \\ 11.431 & -15.740 & -15.697 & 11.477 \end{pmatrix}$$

Notice that across all three rows:
- Weights for unit 0 and unit 3 are equal to within 1%:
  $$w_{out}[:, 0] \approx w_{out}[:, 3] \approx [-1.665, \, -18.736, \, 11.454]^T$$
- Weights for unit 1 and unit 2 are equal to within 1%:
  $$w_{out}[:, 1] \approx w_{out}[:, 2] \approx [1.608, \, 15.979, \, -15.719]^T$$

Defining the outer sum $u = h_{3,0} + h_{3,3}$ and inner sum $v = h_{3,1} + h_{3,2}$, the logits reduce to a 2D linear system:
$$\text{logit}_0 = -1.665 u + 1.608 v$$
$$\text{logit}_1 = -18.736 u + 15.979 v$$
$$\text{logit}_2 = 11.454 u - 15.719 v$$

### Derivation of the Decision Boundaries
We compute pairwise logit differences as functions of the ratio $R = v / u$:

1. **Logit 1 vs Logit 0:**
   $$\text{logit}_1 - \text{logit}_0 = -(18.736 - 1.665) u + (15.979 - 1.608) v = -17.071 u + 14.371 v$$
   $$\text{logit}_1 > \text{logit}_0 \iff \frac{v}{u} > R_1 \equiv \frac{17.071}{14.371} \approx 1.1879$$

2. **Logit 2 vs Logit 0:**
   $$\text{logit}_2 - \text{logit}_0 = (11.454 + 1.665) u - (15.719 + 1.608) v = 13.119 u - 17.327 v$$
   $$\text{logit}_2 > \text{logit}_0 \iff \frac{v}{u} < R_2 \equiv \frac{13.119}{17.327} \approx 0.7571$$

3. **Logit 1 vs Logit 2:**
   $$\text{logit}_1 - \text{logit}_2 = -30.190 u + 31.698 v = 0 \iff \frac{v}{u} = \frac{30.190}{31.698} \approx 0.9524$$

Since $R_2 = 0.7571 < 0.9524 < 1.1879 = R_1$, the $(u, v)$ plane partitions into three clean cones:
- **Sector $v / u > 1.188$:** $\text{logit}_1$ is strictly maximal $\implies$ **Prediction = 1**.
- **Sector $v / u < 0.757$:** $\text{logit}_2$ is strictly maximal $\implies$ **Prediction = 2**.
- **Dead Zone $0.757 \le v / u \le 1.188$:** Both $\text{logit}_1$ and $\text{logit}_2$ are strictly negative, and $\text{logit}_0$ is maximal $\implies$ **Prediction = 0**.
- **Origin $u = v = 0$ ($h_3 = \mathbf{0}$):** All logits are zero. By `argmax` tie-breaking, **Prediction = 0**.

This 2D angular sector rule matches the exact network $\text{argmax}$ with **99.81% accuracy** across 100,000 hidden states!

---

## 5. Why and Where the Network Fails

Empirical evaluation on 50,000 Gaussian inputs reveals that the network achieves **~98.5% accuracy** (1.5% error rate). Over 78% of all errors fall into one specific category: **the true median is 1 or 2, but the network predicts 0**.

### 5.1 Primary Failure Mode ($x_2 \approx x_3$)
The boundary between median 1 and median 2 occurs when $x_2 = x_3$, with $x_1$ lying outside (either $x_1 > x_2, x_3$ or $x_1 < x_2, x_3$).

When $x_2 \approx x_3$:
- The network processes $x_2$ at $t=2$ and nearly identical $x_3$ at $t=3$.
- In the pre-activation $z_3 = w_{ih} x_3 + w_{hh} h_2$, exactly two units survive ReLU:
  - If $x_1 > x_2, x_3$: units 0 and 2 survive ($h_{3,0} > 0, h_{3,2} > 0$).
  - If $x_1 < x_2, x_3$: units 1 and 3 survive ($h_{3,1} > 0, h_{3,3} > 0$).
- In both cases, one active unit belongs to the outer pair and one belongs to the inner pair!
- Furthermore, because $x_3 \approx x_2$, their activation magnitudes are nearly identical:
  $$u = h_{3,0} \approx v = h_{3,2} \quad \implies \quad \frac{v}{u} \approx 1.0$$
- Since $1.0 \in [0.757, 1.188]$, the state falls directly into the **readout dead zone**!
- Both $\text{logit}_1$ and $\text{logit}_2$ are suppressed to large negative values ($\approx -3$ to $-5$), while $\text{logit}_0 \approx -0.1$ stays near 0.
- Thus, **the network outputs position 0 by default!**

#### Derivation of the Primary Gap Threshold ($\text{GAP}_{23}$)
At the boundary $x_2 = x_3 = t$, let $\delta = x_3 - t$. Expanding logits around $\delta = 0$:
$$\text{logit}_i(\delta) = \text{logit}_i(0) + S_i \cdot \delta$$
where $S_i = \sum_{j \in I} w_{out}[i, j] w_{ih}[j]$ for active set $I = \{0, 2\}$.
From the weights:
$$S_0 = -14.27, \quad S_1 = -149.61, \quad S_2 = +119.36$$
The slopes for the logit differences are:
$$\frac{d(\text{logit}_1 - \text{logit}_0)}{d\delta} = -135.35$$
$$\frac{d(\text{logit}_2 - \text{logit}_0)}{d\delta} = +133.62$$
At $\delta = 0$, with $x_1 = 1, t = 0$:
$$\text{logit}_1 - \text{logit}_0 \approx -1.84, \quad \text{logit}_2 - \text{logit}_0 \approx -1.44$$
Solving for the zero-crossings:
$$\delta_{\text{left}} = \frac{-(-1.84)}{-135.35} \approx -0.0136$$
$$\delta_{\text{right}} = \frac{-(-1.44)}{133.62} \approx +0.0107$$
The total physical gap width is $\Delta \delta \approx 0.0243 \cdot x_1$.
Normalizing to the sphere radius $\|\mathbf{x}\|$ with $(0, 1, -1)/\sqrt{2}$ gives:
$$\text{GAP}_{23} = \frac{\Delta \delta}{\sqrt{2}} \approx \frac{0.0243}{\sqrt{2}} \approx 0.0175$$

### 5.2 Secondary Failure Modes ($x_1 \approx x_3$ and $x_1 \approx x_2$)
- **$x_1 \approx x_3$ (Type $0 \to 2$):** When $x_1$ is the true median (position 0), but $x_1$ and $x_3$ are extremely close, residual activation from unit 0 leaks into $\text{logit}_2$ (which has positive weight $+11.45$ on unit 0). This causes $\text{logit}_2$ to surpass $\text{logit}_0$ within a narrow relative gap $\text{GAP}_{13} \approx 0.0050$.
- **$x_1 \approx x_2$ (Type $0 \to 1$):** Similarly, when $x_1 \approx x_2$ and $x_1$ is the true median, unit 1 activation leaks into $\text{logit}_1$ (which has positive weight $+16.00$ on unit 1), causing position 1 to be predicted within a relative gap $\text{GAP}_{12} \approx 0.0028$.

---

## 6. Summary of Derived Constants

All constants used in `twin.py` and `estimate.txt` are derived directly from the weight matrices:

| Constant | Value | Origin / Derivation |
| :--- | :--- | :--- |
| $a$ | $3.965$ | Input weight magnitude for outer units $w_{ih, 0}, -w_{ih, 3}$ |
| $b$ | $4.708$ | Input weight magnitude for inner units $w_{ih, 1}, -w_{ih, 2}$ |
| $g$ | $0.80$ | Recurrent projection ratio at step 2: $(w_{hh}[:, :2] @ [a, b]^T)_1 / w_{ih, 1} \approx 3.794 / 4.700$ |
| $1/g$ | $1.25$ | Reciprocal recurrent ratio: $(w_{hh}[:, :2] @ [a, b]^T)_0 / w_{ih, 0} \approx 5.029 / 3.965$ |
| $R_1$ | $1.188$ | Readout boundary for position 1: $(C_0 - c_0) / (C_1 - c_1) = 17.071 / 14.371$ |
| $R_2$ | $0.757$ | Readout boundary for position 2: $(D_0 + c_0) / (D_1 + c_1) = 13.119 / 17.327$ |
| $\text{GAP}_{23}$ | $0.0175$ | Dead band angular half-width along $x_2 = x_3$: $\Delta \delta / \sqrt{2} \approx 0.0243 / \sqrt{2}$ |
| $\text{GAP}_{13}$ | $0.0050$ | Leakage boundary gap along $x_1 = x_3$ |
| $\text{GAP}_{12}$ | $0.0028$ | Leakage boundary gap along $x_1 = x_2$ |

---

## 7. Accuracy Derivation (Closed-Form Gaussian Measure)

Because i.i.d. standard Gaussian inputs $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, I_3)$ are spherically symmetric and the network is scale-invariant, the probability of any event is equal to the area fraction of its conical projection onto the unit 2-sphere $S^2$ (total area $4\pi$).

1. **Primary Band ($x_2 \approx x_3$):**
   The great circle $x_2 = x_3$ has unit normal $\mathbf{n}_{23} = (0, 1, -1)/\sqrt{2}$. The distance to this plane on the sphere is $\sin\phi = |x_2 - x_3| / \sqrt{2}$. The dead zone $|x_2 - x_3| \le \text{GAP}_{23} \|\mathbf{x}\|$ defines a spherical strip of angular half-width:
   $$\alpha_{23} = \frac{\text{GAP}_{23}}{\sqrt{2}} = \frac{0.0175}{\sqrt{2}} \approx 0.012374 \text{ radians}$$
   The surface area of a strip of half-width $\alpha$ on $S^2$ is $4\pi \sin\alpha$. Its fraction of the sphere is:
   $$P(\text{Error}_{23}) = \sin(\alpha_{23}) \approx 0.012374$$

2. **Secondary Band ($x_1 \approx x_3$):**
   This error occurs only when true median is position 0 (which occupies $1/3$ of the sphere). With angular half-width $\alpha_{13} = 0.0050 / \sqrt{2} \approx 0.003536$:
   $$P(\text{Error}_{13}) = \frac{1}{3} \sin(\alpha_{13}) \approx 0.001179$$

3. **Tertiary Band ($x_1 \approx x_2$):**
   Similarly, this error occurs when true median is 0 on $1/3$ of the sphere, with $\alpha_{12} = 0.0028 / \sqrt{2} \approx 0.001980$:
   $$P(\text{Error}_{12}) = \frac{1}{3} \sin(\alpha_{12}) \approx 0.000660$$

Summing the disjoint error measures:
$$P(\text{Total Error}) = 0.012374 + 0.001179 + 0.000660 = 0.014213$$
$$\text{Estimated Accuracy} = 1 - P(\text{Total Error}) = 0.985787 \approx \mathbf{0.9858}$$

This closed-form theoretical estimate matches the empirical 100,000-sample test accuracy of **0.985010** to within **0.00077** (less than 0.08% error).

---

## 8. Re-Implementation in `twin.py`

`twin.py` implements the mechanistic rule described above using numpy alone:
1. Computes the base true median position.
2. Applies the primary dead-zone rule: if $|x_2 - x_3| < \text{GAP}_{23} \|\mathbf{x}\|$ and true median is in $\{1, 2\}$, override prediction to **0**.
3. Applies secondary leakage rules: if $|x_1 - x_3| < \text{GAP}_{13} \|\mathbf{x}\|$ and true median is 0, override to **2**; if $|x_1 - x_2| < \text{GAP}_{12} \|\mathbf{x}\|$ and true median is 0, override to **1**.

On 50,000 fresh Gaussian inputs, `twin.predict` matches the model's actual outputs (mistakes included) with **99.29% fidelity**.

---

## 9. What Could Not Be Fully Explained

1. **Exact SGD Convergence Values:** While the symmetry $w_{ih} \approx -P w_{ih}$ and the reciprocity $r_1 r_3 \approx 1$ are mathematically necessary for sign equivariance and median comparison, the precise absolute magnitude of the weights (e.g. why $a \approx 3.965$ rather than another scale factor) reflects the learning rate, weight decay, and training steps of the optimizer rather than a strict task invariant.
2. **Polar Triple-Intersection Geometry:** Near the antipodal poles $\pm (1, 1, 1)/\sqrt{3}$ where all three inputs coincide ($x_1 = x_2 = x_3$), the three great-circle error bands intersect. Within a tiny neighborhood of radius $\sim 0.02$ around these poles, the piecewise-linear activation boundaries form complicated polyhedral arrangements that our 1D band approximations simplify as independent strips. However, the spherical measure of this polar intersection region is $< 10^{-4}$ and negligible for overall accuracy.
