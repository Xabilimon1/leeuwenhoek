# Mechanistic Explanation of the Tiny Argmedian RNN

This document provides a complete mechanistic reverse-engineering of the 4-unit ReLU recurrent neural network trained to output the position of the median of three real inputs $x = (x_0, x_1, x_2)$ (or $(x_1, x_2, x_3)$ in 1-indexed notation).

---

## 1. Executive Summary

The network implements a **streaming comparator with sign-pair encoding**:
1. **Sign-Symmetric Dual Representation**: The 4 hidden units are arranged into two complementary sign-pairs:
   - **Outer units** $\{0, 3\}$: fire on positive ($u_0$) and negative ($u_3$) large differences.
   - **Inner units** $\{1, 2\}$: fire on positive ($v_1$) and negative ($v_2$) small/moderate differences.
   The network exploits the exact mathematical symmetry of the median: $\operatorname{median}(-x) = -\operatorname{median}(x)$ (its position is identical under sign reversal $x \mapsto -x$).
2. **Sequential Comparison & Memory**:
   - At $t=1$, $h_1$ stores the signed magnitude of $x_0$.
   - At $t=2$, $h_2$ computes the difference $(x_1 - x_0)$ by subtracting the recurrent memory of $x_0$ from the feedforward input $x_1$.
   - At $t=3$, $h_3$ computes the difference $(x_2 - x_1)$ modulated by $(x_1 - x_0)$, effectively locating $x_2$ relative to the interval $[x_0, x_1]$.
3. **Readout Mechanism**:
   The output layer evaluates the ratio of inner activation to outer activation:
   $$\rho = \frac{v_1 + v_2}{u_0 + u_3}$$
   - If $\rho > 1.188$: predict **Position 1** ($x_1$ is median).
   - If $\rho < 0.757$: predict **Position 2** ($x_2$ is median).
   - If $0.757 \le \rho \le 1.188$: predict **Position 0** ($x_0$ is median).
4. **Failure Modes & Error Regions**:
   - **Primary Error (~80% of mistakes)**: When $x_1 \approx x_2$, the feedforward input $w_{ih} x_2$ and the recurrent input $w_{hh} h_2$ cancel each other out. This cancellation destroys the delicate difference between $x_1$ and $x_2$, leaving only the residual memory of $x_0$. The readout receives $\rho \approx 0.955 \in [0.757, 1.188]$ and falsely predicts **Position 0**.
   - **Secondary Errors (~20% of mistakes)**: Minor readout hyperplane misalignments ($G_{20} \approx 0.0069$ and $G_{10} \approx 0.0043$) where position 2 or 1 intrudes into position 0.

---

## 2. Weight Structure and Symmetries

The network parameters are:
- $w_{ih} \in \mathbb{R}^4$: input-to-hidden weights
- $w_{hh} \in \mathbb{R}^{4 \times 4}$: hidden-to-hidden recurrence
- $w_{out} \in \mathbb{R}^{3 \times 4}$: hidden-to-output logits

Let $J = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix}$ be the anti-diagonal exchange matrix (reversing indices $0 \leftrightarrow 3$ and $1 \leftrightarrow 2$).

Inspecting the weights reveals near-exact symmetries:
$$w_{ih} \approx -J w_{ih}, \quad w_{hh} \approx J w_{hh} J, \quad w_{out} \approx w_{out} J$$

Specifically:
- **$w_{ih}$**:
  $$w_{ih} = \begin{pmatrix} 3.965263 \\ 4.699488 \\ -4.716320 \\ -3.965202 \end{pmatrix} \approx \begin{pmatrix} a \\ b \\ -b \\ -a \end{pmatrix}$$
  with $a \approx 3.965$ and $b \approx 4.700$ (ratio $b/a \approx 1.185$).

- **$w_{hh}$**:
  $$w_{hh} = \begin{pmatrix} 1.3387 & -2.1996 & 2.6352 & -1.8948 \\ 1.9277 & -2.4339 & 2.7868 & -2.3826 \\ -2.3877 & 2.8057 & -2.4339 & 1.9331 \\ -1.8931 & 2.6429 & -2.1912 & 1.3409 \end{pmatrix}$$
  $w_{hh}$ satisfies $J w_{hh} J = w_{hh}$ to within $0.01$. SVD shows singular values $\sigma = [8.922, 0.918, 0.443, 0.001]$; the dominant rank-1 component has left/right singular vectors symmetric under $J$.

- **$w_{out}$**:
  $$w_{out} = \begin{pmatrix} -1.6718 & 1.5993 & 1.6175 & -1.6575 \\ -18.7477 & 15.9993 & 15.9592 & -18.7243 \\ 11.4305 & -15.7403 & -15.6973 & 11.4769 \end{pmatrix}$$
  Each row is symmetric: $w_{out}[k, 0] \approx w_{out}[k, 3]$ and $w_{out}[k, 1] \approx w_{out}[k, 2]$.

### Consequence: Sign-Flip Equivariance
Because $\operatorname{relu}(J v) = J \operatorname{relu}(v)$ (permutation commutes with elementwise ReLU), negating the inputs $x \mapsto -x$ transforms the hidden state as $h_t(-x) = J h_t(x)$.
At the output:
$$\operatorname{logits}(-x) = w_{out} h_3(-x) \approx w_{out} J h_3(x) = w_{out} h_3(x)$$
The logits are completely invariant to sign reversal! This mirrors the property of the median: reversing the signs of all elements inverts their order, leaving the median index unchanged.

---

## 3. Step-by-Step Representation

### Step 1 ($t = 1$, input $x_0$)
Since $h_0 = 0$:
$$h_1 = \operatorname{relu}(w_{ih} x_0)$$
- If $x_0 > 0$: $h_1 = [a x_0, b x_0, 0, 0]^T$.
- If $x_0 < 0$: $h_1 = [0, 0, b |x_0|, a |x_0|]^T$.

Units 0 and 1 represent positive $x_0$; units 2 and 3 represent negative $x_0$.

### Step 2 ($t = 2$, input $x_1$)
The recurrent contribution from $h_1$ is:
- When $x_0 > 0$:
  $$w_{hh} h_1 = x_0 \begin{pmatrix} a w_{hh}[0,0] + b w_{hh}[0,1] \\ a w_{hh}[1,0] + b w_{hh}[1,1] \\ a w_{hh}[2,0] + b w_{hh}[2,1] \\ a w_{hh}[3,0] + b w_{hh}[3,1] \end{pmatrix} = x_0 \begin{pmatrix} -5.0286 \\ -3.7939 \\ +3.7174 \\ +4.9135 \end{pmatrix} \approx x_0 \begin{pmatrix} -k_0 \\ -k_1 \\ +k_1 \\ +k_0 \end{pmatrix}$$
  where $k_0 \approx 5.029$ and $k_1 \approx 3.794$.
- When $x_0 < 0$: by $J$-symmetry, $w_{hh} h_1 = -x_0 [k_0, k_1, -k_1, -k_0]^T = x_0 [-k_0, -k_1, k_1, k_0]^T$.

Thus, for any $x_0$, the pre-activations at step 2 are:
$$z_2 = \begin{pmatrix} z_2[0] \\ z_2[1] \\ z_2[2] \\ z_2[3] \end{pmatrix} = \begin{pmatrix} a x_1 - k_0 x_0 \\ b x_1 - k_1 x_0 \\ -b x_1 + k_1 x_0 \\ -a x_1 + k_0 x_0 \end{pmatrix} = \begin{pmatrix} a x_1 - k_0 x_0 \\ b x_1 - k_1 x_0 \\ -z_2[1] \\ -z_2[0] \end{pmatrix}$$

Notice:
$$z_2[2] = -z_2[1] \quad \text{and} \quad z_2[3] = -z_2[0]$$
The zero-crossings are:
- $z_2[0] = 0 \iff x_1 = \frac{k_0}{a} x_0 \approx 1.268 x_0$
- $z_2[1] = 0 \iff x_1 = \frac{k_1}{b} x_0 \approx 0.807 x_0$

Remarkably, $\frac{k_1}{b} \times \frac{k_0}{a} \approx 0.807 \times 1.268 \approx 1.023 \approx 1$.
This partitions the relation between $x_1$ and $x_0$ into three activation regimes:
1. **$x_1 \gg x_0$ ($x_1 > 1.27 x_0$)**: Both $z_2[0] > 0$ and $z_2[1] > 0$. Units $\{0, 1\}$ are active.
2. **$x_1 \ll x_0$ ($x_1 < 0.81 x_0$)**: Both $z_2[2] > 0$ and $z_2[3] > 0$. Units $\{2, 3\}$ are active.
3. **$x_1 \approx x_0$ ($0.81 x_0 \le x_1 \le 1.27 x_0$)**: $z_2[1] > 0$ and $z_2[3] > 0$. Units $\{1, 3\}$ are active.

### Step 3 ($t = 3$, input $x_2$)
At step 3, $z_3 = w_{ih} x_2 + w_{hh} h_2$.
The matrix $w_{hh}$ acts on $h_2$ to subtract the contribution of $x_1$ while propagating a residual of $x_0$.
In regime 1 ($x_1 > x_0$):
$$z_3 \approx \begin{pmatrix} a x_2 - k_0 x_1 + c_0 x_0 \\ b x_2 - k_1 x_1 + c_1 x_0 \\ -b x_2 + k_1 x_1 + c_2 x_0 \\ -a x_2 + k_0 x_1 + c_3 x_0 \end{pmatrix}$$

The network compares $x_2$ with $x_1$, calibrated by the position of $x_0$.

---

## 4. Readout Mechanism

Let $u = h_3[0] + h_3[3]$ (outer units) and $v = h_3[1] + h_3[2]$ (inner units).
Because each row of $w_{out}$ is symmetric under $J$, the logits depend only on $u$ and $v$:
$$\begin{pmatrix} \text{logit}_0 \\ \text{logit}_1 \\ \text{logit}_2 \end{pmatrix} = u \begin{pmatrix} -1.6647 \\ -18.7360 \\ 11.4537 \end{pmatrix} + v \begin{pmatrix} 1.6084 \\ 15.9792 \\ -15.7188 \end{pmatrix}$$

Pairwise differences:
- $\text{logit}_1 - \text{logit}_0 = -17.0713 u + 14.3708 v > 0 \iff \frac{v}{u} > 1.1879$
- $\text{logit}_2 - \text{logit}_0 = 13.1184 u - 17.3272 v > 0 \iff \frac{v}{u} < 0.7571$
- $\text{logit}_1 - \text{logit}_2 = -30.1897 u + 31.6981 v > 0 \iff \frac{v}{u} > 0.9524$

Because $h_3 \ge 0$, $u \ge 0$ and $v \ge 0$. The prediction rule is determined entirely by $\rho = v / u$:
$$\operatorname{argmax}(\text{logits}) = \begin{cases} 1 & \text{if } \rho > 1.1879 \\ 2 & \text{if } \rho < 0.7571 \\ 0 & \text{if } 0.7571 \le \rho \le 1.1879 \end{cases}$$

---

## 5. Failure Modes and Mathematical Derivation of Constants

Why does the model fail on ~1.6% of Gaussian inputs?

### Primary Error: $x_1 \approx x_2$ Cancellation
When $x_1$ and $x_2$ are close, the recurrence from $h_2$ (which encoded $x_1$) cancels against the feedforward term $w_{ih} x_2$.
In this boundary region, units 1 and 3 of $h_3$ are active:
$$h_3[1] = Z_3[1] \cdot x = -0.9408 x_0 - 3.7939 x_1 + 4.6995 x_2$$
$$h_3[3] = Z_3[3] \cdot x = -0.9690 x_0 + 4.9135 x_1 - 3.9652 x_2$$
When $x_2 = x_1$:
$$h_3[1] = 0.9056 (x_1 - 1.039 x_0)$$
$$h_3[3] = 0.9483 (x_1 - 1.022 x_0)$$
The ratio is:
$$\rho = \frac{h_3[1]}{h_3[3]} \approx \frac{0.9056}{0.9483} \approx 0.955$$
Notice $0.955$ lies directly inside the interval $[0.7571, 1.1879]$ where **Logit 0 wins**!
Because $x_1 \approx x_2$, their differences vanish in the network's state, leaving only the residual memory of $x_0$. The network defaults to predicting position 0, even though the true median is $x_1$ or $x_2$!

#### Derivation of $G_R = 0.0232$:
Logit 1 beats Logit 0 when:
$$(w_{out}[1] - w_{out}[0]) \cdot h_3 = 2.9902 x_0 - 138.4895 x_1 + 135.3457 x_2 > 0$$
$$\iff x_2 - x_1 > \frac{138.4895 - 135.3457}{135.3457} x_1 - \frac{2.9902}{135.3457} x_0 = 0.02322 x_1 - 0.02209 x_0 \approx 0.0232 (x_1 - x_0)$$
Hence $G_R = 0.0232$.

#### Derivation of $G_L = 0.0243$:
Logit 2 beats Logit 0 when:
$$(w_{out}[2] - w_{out}[0]) \cdot h_3 = 3.5859 x_0 + 130.3208 x_1 - 133.5682 x_2 > 0$$
$$\iff x_1 - x_2 > \frac{133.5682 - 130.3208}{133.5682} x_1 - \frac{3.5859}{133.5682} x_0 = 0.02431 x_1 - 0.02685 x_0 \approx 0.0243 (x_1 - x_0)$$
Hence $G_L = 0.0243$.

### Secondary Errors: Boundary Shifts near $x_0$
When $x_0$ is the median:
- Near $x_2 \approx x_0$, readout asymmetry causes Logit 2 to exceed Logit 0 for $x_2 - x_0 < G_{20} (x_0 - x_1)$ where $G_{20} = 0.0069$.
- Near $x_1 \approx x_0$, Logit 1 exceeds Logit 0 for $x_1 - x_0 < G_{10} (x_0 - x_2)$ where $G_{10} = 0.0043$.

---

## 6. Analytical Accuracy Calculation

Under i.i.d. standard Gaussians $X \sim \mathcal{N}(0, I_3)$, the network's predictions are scale-invariant ($f(c x) = c f(x)$) and translation-invariant along $(1, 1, 1)$. Projecting onto the plane perpendicular to $(1, 1, 1)$, the Gaussian measure is isotropic 2D Gaussian.

The wedge angle $\theta(g)$ for any boundary $x_b - x_a = g (x_a - x_c)$ is given in closed form by:
$$\theta(g) = \arctan\left(\frac{\sqrt{3} g}{2 + g}\right)$$
The probability of a wedge of angle $\theta(g)$ is $P(g) = \frac{\theta(g)}{2\pi}$.

Summing all error wedges:
1. **$x_1 \approx x_2$ error band**: 4 symmetric sectors (2 right, 2 left):
   $$2 \times P(G_R) + 2 \times P(G_L) = 2 \times 0.00316062 + 2 \times 0.00330864 = 0.01293853 \quad (1.2939\%)$$
2. **$x_2 \approx x_0$ shift**: 2 symmetric sectors:
   $$2 \times P(G_{20}) = 2 \times 0.00094776 = 0.00189552 \quad (0.1896\%)$$
3. **$x_1 \approx x_0$ shift**: 2 symmetric sectors:
   $$2 \times P(G_{10}) = 2 \times 0.00059140 = 0.00118281 \quad (0.1183\%)$$

Total error probability:
$$P_{\text{error}} = 0.01293853 + 0.00189552 + 0.00118281 = 0.01601686 \quad (1.6017\%)$$
Analytical Accuracy Estimate:
$$\text{Accuracy} = 1 - P_{\text{error}} = 0.98398314 \quad (\mathbf{98.398\%})$$

---

## 7. What Could Not Be Explained

1. **SGD Noise / Second-Order Asymmetries**:
   The weights deviate from perfect $J$-reversal symmetry by $\approx 0.005$ to $0.02$. These small perturbations are non-functional artifacts of stochastic gradient descent during training.
2. **Global Weight Scaling**:
   The network has no biases and uses homogeneous ReLUs. Any positive scaling $w_{ih} \to \alpha w_{ih}$, $w_{hh} \to w_{hh}$, $w_{out} \to \beta w_{out}$ produces identical classifications. The specific scale factor $(\|w_{hh}\|_2 \approx 8.92)$ is an arbitrary artifact of the learning rate and weight decay hyperparameters used during training.
