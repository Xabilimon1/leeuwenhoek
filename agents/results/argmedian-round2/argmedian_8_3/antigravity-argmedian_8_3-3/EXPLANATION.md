# Mechanistic Explanation of the Trained Argmedian Network

## 1. Overview and High-Level Architecture

The network is a bias-free recurrent neural network with 8 hidden units that processes a sequence of 3 scalar inputs $x = (x_0, x_1, x_2)$ (0-indexed) and outputs 3 logits:
$$\begin{aligned}
h_0 &= 0 \\
h_t &= \text{relu}(w_{ih} x_t + w_{hh} h_{t-1}) \quad \text{for } t = 1, 2, 3 \\
\text{logits} &= w_{out} h_3
\end{aligned}$$
The final predicted position is $\text{argmax}_{k \in \{0, 1, 2\}} \text{logits}_k$. The target task is finding the position of the median of the three inputs.

### Key Structural Properties
1. **Scale-Invariance (Homogeneity of Degree 1)**:
   Because there are zero bias terms throughout the network and the ReLU activation is positively homogeneous ($\text{relu}(c z) = c \cdot \text{relu}(z)$ for $c \ge 0$), the entire network is homogeneous of degree 1:
   $$\text{logits}(c \cdot x) = c \cdot \text{logits}(x) \quad \text{for all } c > 0.$$
   Consequently, multiplying any input vector by a positive scalar does not change the argmax. The decision regions in $\mathbb{R}^3$ are therefore homogeneous conical regions radiating from the origin.
2. **Sequential Sorting Strategy**:
   The network does not attempt to compute an arithmetic average or global statistic. Instead, it implements a sequential tournament / sorting algorithm:
   - At step 1 ($t=1$), it records the magnitude and sign of $x_0$.
   - At step 2 ($t=2$), it performs an initial comparison between $x_0$ and $x_1$, identifying whether $x_0 < x_1$ or $x_1 < x_0$, and storing representations of their candidate minimum, candidate maximum, and their sum.
   - At step 3 ($t=3$), upon receiving $x_2$, it compares $x_2$ against the candidate minimum and candidate maximum. If $x_2$ is between them, position 2 wins. Otherwise, the candidate extreme that is not $x_2$ wins.

---

## 2. Step-by-Step Representation of Hidden Units

### Step 1 ($t=1$, input $x_0$)
At step 1, $h_0 = 0$, so:
$$h_1 = \text{relu}(w_{ih} x_0)$$
Inspection of the input weight vector $w_{ih}$:
$$\begin{aligned}
w_{ih}[0] &= +6.578, & w_{ih}[1] &= -5.521, & w_{ih}[2] &= +6.155, & w_{ih}[3] &= +16.208, \\
w_{ih}[4] &= -6.355, & w_{ih}[5] &= +1.449, & w_{ih}[6] &= -9.872, & w_{ih}[7] &= +2.404.
\end{aligned}$$

Because ReLUs truncate negative values:
- When $x_0 > 0$, units **$\{0, 2, 3, 5, 7\}$** activate with magnitudes proportional to $x_0$:
  $$h_1 = [6.578, 0, 6.155, 16.208, 0, 1.449, 0, 2.404]^T \cdot x_0$$
- When $x_0 < 0$, units **$\{1, 4, 6\}$** activate with magnitudes proportional to $|x_0| = -x_0$:
  $$h_1 = [0, 5.521, 0, 0, 6.355, 0, 9.872, 0]^T \cdot (-x_0)$$

Thus, $h_1$ acts as a two-sided rectifier (positive/negative decomposition) of $x_0$.

---

### Step 2 ($t=2$, input $x_1$)
At step 2, the pre-activation vector is:
$$a_2 = w_{ih} x_1 + w_{hh} h_1$$
Because $h_1$ is linear in $x_0$ on the half-spaces $x_0 > 0$ and $x_0 < 0$, the pre-activations $a_2[i]$ are exact linear combinations of $x_0$ and $x_1$ in each half-space. Evaluating $w_{hh} h_1$ yields:

$$\begin{aligned}
\text{For } x_0 > 0: \quad a_2[i] &= w_{ih}[i] x_1 + (w_{hh} \text{relu}(w_{ih}))[i] \cdot x_0 \\
\text{For } x_0 < 0: \quad a_2[i] &= w_{ih}[i] x_1 - (w_{hh} \text{relu}(-w_{ih}))[i] \cdot x_0
\end{aligned}$$

#### Functional Roles and Unit Pairings in $h_2$:

1. **Units 3 and 6 — Pairwise Difference Detection $(x_1 - x_0)$**:
   - **Unit 3** has a large positive input weight ($w_{ih}[3] = +16.208$). For $x_0 > 0$:
     $$a_2[3] = 16.208 x_1 - 11.583 x_0 = 16.208 (x_1 - 0.715 x_0)$$
     Unit 3 activates strongly when $x_1 > x_0$ (correlation with $\text{relu}(x_1 - x_0)$ is **0.900**).
   - **Unit 6** has a large negative input weight ($w_{ih}[6] = -9.872$). For $x_0 > 0$:
     $$a_2[6] = -9.872 x_1 + 10.057 x_0 = 9.872 (1.019 x_0 - x_1)$$
     Notice that $1.019 \approx 1.000$! Unit 6 directly computes $9.872 \cdot \max(0, x_0 - x_1)$. Its correlation with $\text{relu}(x_0 - x_1)$ is **0.979**.

2. **Units 2 and 4 — Signed Sum / Common Mode Tracking $(x_0 + x_1)$**:
   - For $x_0 > 0$:
     $$a_2[2] = +6.155 (x_1 + 2.653 x_0), \quad a_2[4] = -6.355 (x_1 + 2.651 x_0)$$
   - For $x_0 < 0$:
     $$a_2[2] = +6.155 (x_1 + 2.213 x_0), \quad a_2[4] = -6.355 (x_1 + 2.210 x_0)$$
   Notice that the ratio is identical between units 2 and 4 ($2.653 \approx 2.651$, and $2.213 \approx 2.210$). Unit 2 and Unit 4 form an exact complementary positive/negative rectifier pair for the linear combination $(x_1 + c x_0)$. Together, they allow the recurrent layer to track $(x_0 + x_1)$ (Unit 2 correlates **0.887** with $\text{relu}(x_0 + x_1)$, Unit 4 correlates **0.909** with $\text{relu}(-(x_0 + x_1))$).

3. **Units 0 and 1 — Second Linear Combination Rectifier Pair**:
   - For $x_0 > 0$:
     $$a_2[0] = +6.578 (x_1 - 2.271 x_0), \quad a_2[1] = -5.521 (x_1 - 2.258 x_0)$$
   - These form another complementary pair ($2.271 \approx 2.258$) that activates when $x_1$ differs substantially from $2.26 x_0$.

4. **Unit 7 — Strong Inhibition / Quenching**:
   - Row 7 of $w_{hh}$ contains massive negative recurrent weights ($-24.74$ from unit 6 and $-41.90$ from unit 7). As a result, the recurrent input to Unit 7 is $-113.7 x_0$ when $x_0 > 0$ and $+229.9 x_0$ when $x_0 < 0$. In both cases, the pre-activation is driven deeply negative, completely turning off Unit 7 at step 2 (active frequency: $0.2\%$). Unit 7 serves solely as a step-1 transient.

---

### Step 3 ($t=3$, input $x_2$) & Readout

At step 3, $x_2$ enters via $w_{ih} x_2$ and mixes with the recurrent state $h_2$:
$$a_3 = w_{ih} x_2 + w_{hh} h_2, \quad h_3 = \text{relu}(a_3), \quad \text{logits} = w_{out} h_3$$

The readout matrix $w_{out}$ is:
$$w_{out} = \begin{bmatrix}
+3.668 & +3.453 & -6.933 & +3.070 & -7.255 & +0.079 & +5.321 & -0.683 \\
-8.258 & -5.368 & +1.466 & +4.716 & +4.548 & -2.991 & +2.658 & +1.405 \\
+2.259 & +0.395 & +3.195 & -12.205 & -0.873 & +2.131 & -7.365 & -0.217
\end{bmatrix}$$

#### How the Readout Produces the Decision:
1. **Selecting Position 2 (Logit 2)**:
   In row 2 of $w_{out}$, notice the two dominant negative weights:
   - Weight on Unit 3 is **$-12.205$**
   - Weight on Unit 6 is **$-7.365$**
   Recall that at step 3, Unit 3 has $w_{ih}[3] = +16.208$ and Unit 6 has $w_{ih}[6] = -9.872$.
   - If $x_2$ is larger than both $x_0$ and $x_1$ ($x_2 = \max$), Unit 3 is driven strongly positive, and the term $-12.205 \cdot h_3[3]$ penalizes logit 2 heavily.
   - If $x_2$ is smaller than both $x_0$ and $x_1$ ($x_2 = \min$), Unit 6 is driven strongly positive, and the term $-7.365 \cdot h_3[6]$ penalizes logit 2 heavily.
   - Only when $x_2$ is in between the candidate minimum and candidate maximum ($\min(x_0, x_1) \le x_2 \le \max(x_0, x_1)$) do both Unit 3 and Unit 6 remain small or inactive. Without these heavy penalties, logit 2 dominates and position 2 is predicted!

2. **Selecting Position 0 vs Position 1**:
   When $x_2$ is an extreme value ($\min$ or $\max$), the median is either $x_0$ or $x_1$:
   - If $x_2$ is the minimum, the median is $\min(x_0, x_1)$.
   - If $x_2$ is the maximum, the median is $\max(x_0, x_1)$.
   Logit 0 receives strong positive contributions from Unit 6 (+5.32) and Unit 0 (+3.67), while Logit 1 receives positive contributions from Unit 3 (+4.72) and Unit 4 (+4.55). The balance between these units in $h_3$ correctly routes the decision to position 0 when $x_0$ is the median, and to position 1 when $x_1$ is the median.

---

## 3. Derivation of All Named Constants from Weights

In any linear region defined by active ReLUs $b_1 \in \{0, 1\}^8$ (step 1), $b_2 \in \{0, 1\}^8$ (step 2), and $b_3 \in \{0, 1\}^8$ (step 3), the network computes an exact linear map:
$$\text{logits} = M \begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix}$$
where $M = w_{out} H_3$, with the 3 columns of $H_3 \in \mathbb{R}^{8 \times 3}$ given by:
$$\begin{aligned}
H_3[:, 0] &= \text{diag}(b_3) \, w_{hh} \, \text{diag}(b_2) \, w_{hh} \, \text{diag}(b_1) \, w_{ih} \\
H_3[:, 1] &= \text{diag}(b_3) \, w_{hh} \, \text{diag}(b_2) \, w_{ih} \\
H_3[:, 2] &= \text{diag}(b_3) \, w_{ih}
\end{aligned}$$

The decision boundary between logit $i$ and logit $j$ is the hyperplane where $\text{logits}_i - \text{logits}_j = 0$, whose normal vector is the row difference $n = M[i, :] - M[j, :]$:
$$n_0 x_0 + n_1 x_1 + n_2 x_2 = 0.$$

There are 6 pairwise interfaces between adjacent permutation sectors in $\mathbb{R}^3$. We derive the exact normal vectors and linear boundary equations for each of them below:

### Boundary B1: $x_1 \approx x_2$, with $x_0 < \min(x_1, x_2)$ (Competitors: Logits 1 vs 2)
- Canonical point: $x = [-1, 1, 1]^T$
- Active units: $b_1 = \{1, 4, 6\}$, $b_2 = \{0, 3, 4, 5\}$, $b_3 = \{1, 2, 3\}$
- Normal vector $n = M[1] - M[2]$:
  $$n = [-0.274877, -295.333388, +295.439045]$$
- Boundary equation for $x_2$:
  $$x_2 = -\frac{n_1}{n_2} x_1 - \frac{n_0}{n_2} x_0 = \mathbf{0.999642} \, x_1 + \mathbf{0.000930} \, x_0$$
- **Named Constants**: `SLOPE_B1_X1 = 0.999642`, `COEFF_B1_X0 = 0.000930`.

### Boundary B2: $x_0 \approx x_2$, with $\max(x_0, x_2) < x_1$ (Competitors: Logits 2 vs 0)
- Canonical point: $x = [-1, 1, -1]^T$
- Active units: $b_1 = \{1, 4, 6\}$, $b_2 = \{0, 3, 4, 5\}$, $b_3 = \{1, 2, 6\}$
- Normal vector $n = M[2] - M[0]$:
  $$n = [-204.641335, +0.026889, +204.456991]$$
- Boundary equation for $x_2$:
  $$x_2 = -\frac{n_0}{n_2} x_0 - \frac{n_1}{n_2} x_1 = \mathbf{1.000902} \, x_0 - \mathbf{0.000132} \, x_1$$
- **Named Constants**: `SLOPE_B2_X0 = 1.000902`, `COEFF_B2_X1 = -0.000132`.

### Boundary B3: $x_0 \approx x_1$, with $x_2 < \min(x_0, x_1)$ (Competitors: Logits 0 vs 1)
- Canonical point: $x = [1, 1, -1]^T$
- Active units: $b_1 = \{0, 2, 3, 5, 7\}$, $b_2 = \{1, 2, 3, 6\}$, $b_3 = \{1, 4, 6\}$
- Normal vector $n = M[0] - M[1]$:
  $$n = [-740.442167, +740.732302, +0.009071]$$
- Boundary equation for $x_1$:
  $$x_1 = -\frac{n_0}{n_1} x_0 - \frac{n_2}{n_1} x_2 = \mathbf{0.999608} \, x_0 - 0.000012 \, x_2$$
- **Named Constant**: `SLOPE_01_LOW_X2 = 0.999608` (the $x_2$ dependence is $< 10^{-5}$ and negligible).

### Boundary B4: $x_1 \approx x_2$, with $\max(x_1, x_2) < x_0$ (Competitors: Logits 1 vs 2)
- Canonical point: $x = [1, -1, -1]^T$
- Active units: $b_1 = \{0, 2, 3, 5, 7\}$, $b_2 = \{1, 2, 6\}$, $b_3 = \{0, 4, 5, 6\}$
- Normal vector $n = M[1] - M[2]$:
  $$n = [-0.090721, +210.226568, -209.994527]$$
- Boundary equation for $x_2$:
  $$x_2 = -\frac{n_1}{n_2} x_1 - \frac{n_0}{n_2} x_0 = \mathbf{1.001105} \, x_1 - \mathbf{0.000432} \, x_0$$
- **Named Constants**: `SLOPE_B4_X1 = 1.001105`, `COEFF_B4_X0 = -0.000432`.

### Boundary B5: $x_0 \approx x_2$, with $x_1 < \min(x_0, x_2)$ (Competitors: Logits 2 vs 0)
- Canonical point: $x = [1, -1, 1]^T$
- Active units: $b_1 = \{0, 2, 3, 5, 7\}$, $b_2 = \{1, 2, 6\}$, $b_3 = \{0, 3, 4, 5\}$
- Normal vector $n = M[2] - M[0]$:
  $$n = [+294.661081, -0.289271, -294.431727]$$
- Boundary equation for $x_2$:
  $$x_2 = -\frac{n_0}{n_2} x_0 - \frac{n_1}{n_2} x_1 = \mathbf{1.000779} \, x_0 - \mathbf{0.000982} \, x_1$$
- **Named Constants**: `SLOPE_B5_X0 = 1.000779`, `COEFF_B5_X1 = -0.000982`.

### Boundary B6: $x_0 \approx x_1$, with $\max(x_0, x_1) < x_2$ (Competitors: Logits 0 vs 1)
- Canonical point: $x = [-1, -1, 1]^T$
- Active units: $b_1 = \{1, 4, 6\}$, $b_2 = \{0, 4, 5, 6\}$, $b_3 = \{0, 2, 3\}$
- Normal vector $n = M[0] - M[1]$:
  $$n = [+422.537013, -422.307357, +0.072094]$$
- Boundary equation for $x_1$:
  $$x_1 = -\frac{n_0}{n_1} x_0 - \frac{n_2}{n_1} x_2 = \mathbf{1.000544} \, x_0 + 0.000171 \, x_2$$
- **Named Constant**: `SLOPE_01_HIGH_X2 = 1.000544` (the $x_2$ dependence is $< 2 \cdot 10^{-4}$).

---

## 4. Error Analysis: Where and Why the Network Gets the Median Wrong

### Empirical Nature of the Errors
Testing on large batches of standard Gaussian inputs reveals that the network achieves $\approx 99.94\%$ accuracy, making errors on approximately $0.06\%$ of inputs. 

Every single error occurs when two inputs are nearly identical ($|x_i - x_j| \lesssim 0.002$):
1. **No Errors on Separated Inputs**: On any input where all pairwise differences satisfy $|x_i - x_j| > 0.005$, the network's prediction is $100\%$ accurate.
2. **Bias Toward Position 2**: In over $70\%$ of the error cases, the network mistakenly predicts position 2 instead of the true median (0 or 1).
   - *Why?* $x_2$ is fed directly into the network at the final step $t=3$ without undergoing any recurrent attenuation through $w_{hh}$. Inputs $x_0$ and $x_1$, by contrast, are filtered through 2 and 1 recurrent steps respectively. Consequently, the representations of $x_0$ and $x_1$ suffer small precision and scale losses relative to $x_2$, slightly widening the region in which logit 2 exceeds logits 0 and 1.
3. **Hyperplane Tilts**: As shown by our analytical normal vectors, each decision plane is tilted by a tiny angle:
   $$\theta_{\text{tilt}} = \arccos\left(\frac{n_{\text{model}} \cdot n_{\text{ideal}}}{\|n_{\text{model}}\| \|n_{\text{ideal}}\|}\right) \in [0.000196, 0.000796] \text{ rad} \quad (\approx 0.01^\circ \text{ to } 0.045^\circ).$$
   The error regions are precisely the thin dihedral wedges enclosed between the ideal sorting planes and the network's slightly tilted decision planes.

---

## 5. Accuracy Derivation (For estimate.txt)

Under $x \sim \mathcal{N}(0, I_3)$, the probability density is spherically symmetric. The central line of symmetry for the 6 permutation sectors is $\mathbf{u} = \frac{1}{\sqrt{3}}(1, 1, 1)^T$. In the orthogonal subspace $\mathcal{P} = \{x : x_0 + x_1 + x_2 = 0\}$, the 6 ideal boundaries are rays separated by angles of $\pi/3$.

The Gaussian measure of a dihedral wedge of angle $\Delta \theta$ along a half-plane interface is:
$$P(\text{wedge}) = \frac{\Delta \theta}{2\pi}$$
Summing over the 6 canonical decision interfaces:
$$\begin{aligned}
\Delta \theta_1 &= 0.000682 \text{ rad} \\
\Delta \theta_2 &= 0.000460 \text{ rad} \\
\Delta \theta_3 &= 0.000196 \text{ rad} \\
\Delta \theta_4 &= 0.000631 \text{ rad} \\
\Delta \theta_5 &= 0.000796 \text{ rad} \\
\Delta \theta_6 &= 0.000297 \text{ rad} \\
\sum_{k=1}^6 \Delta \theta_k &= 0.003062 \text{ rad}
\end{aligned}$$

The total error probability is:
$$P(\text{error}) = \frac{0.003062}{2\pi} \approx 0.0004874$$
giving an estimated accuracy of:
$$\text{Accuracy} = 1 - 0.0004874 = \mathbf{0.999513}.$$

---

## 6. Limitations and What Could Not Be Explained

1. **Unit 7 Residual Dynamics**: Unit 7 is active at step 1 for $x_0 > 0$, but receives overwhelmingly negative recurrent inhibition ($-113.7 x_0$ or $+229.9 x_0$) at step 2, causing it to deactivate in over $99.8\%$ of inputs. In $< 0.2\%$ of inputs (extreme combinations of inputs), Unit 7 has residual pre-activations near zero. We did not find a significant computational role for these rare activations; they appear to be benign training artifacts.
2. **Behavior Under Large Common-Mode Shifts**: Although the network behaves as an almost perfect median operator on $\mathcal{N}(0, I_3)$ (and under shifts $|s| \le 1$), shifting all inputs by $s \cdot [1, 1, 1]$ with $|s| \ge 10$ degrades accuracy. Because there are no bias terms, the network cannot compute an exact shift-invariant translation; it relies on Gaussian inputs being centered near 0. The exact reason why training converged to this specific balance of weights rather than an exact difference-only representation could not be uniquely proven.
