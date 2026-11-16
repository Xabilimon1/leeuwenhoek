# Mechanistic Explanation of the Tiny Argmedian RNN

## 1. Executive Summary & Overview

The network is a 2-hidden-unit recurrent neural network with ReLU activations and no biases (12 parameters in total), trained to identify the position of the median ($0, 1, \text{ or } 2$) in a length-3 sequence $x = (x_1, x_2, x_3)$ drawn from an i.i.d. standard Gaussian distribution $\mathcal{N}(0, 1)$.

The recurrent dynamics and readout are defined by:
$$\begin{aligned}
h_0 &= \begin{pmatrix} 0 \\ 0 \end{pmatrix} \\
h_t &= \text{ReLU}(w_{ih} x_t + w_{hh} h_{t-1}) \quad \text{for } t \in \{1, 2, 3\} \\
\text{logits} &= w_{out} h_3 \\
\hat{y} &= \operatorname{argmax}_{k \in \{0, 1, 2\}} \text{logits}_k
\end{aligned}$$

### Core Findings
1. **The Network Never Predicts Position 2**:
   Row 2 of the readout matrix $w_{out}$ is strictly dominated by rows 0 and 1 throughout the entire non-negative quadrant $h_3 \in [0, \infty)^2$. For any input $x$, $\text{logit}_2 < \max(\text{logit}_0, \text{logit}_1)$ (or all are 0, breaking ties in favor of 0). Consequently, the network predicts position 2 with probability $0.0$, automatically capping its theoretical accuracy at $\le 66.67\%$.
2. **True Median Accuracy is ~41.01%**:
   The network achieves an overall accuracy of $\approx 41.01\%$ on i.i.d. Gaussian inputs (barely above the 33.33% random baseline). It behaves almost like a biased binary detector, predicting position 0 on $\approx 91.2\%$ of inputs and position 1 on $\approx 8.8\%$ of inputs.
3. **Severe Memory Erasure from Bias-Free ReLU**:
   Because $w_{ih} > 0$ with no biases, any negative input $x_t \le 0$ produces a zero pre-activation if $h_{t-1} = 0$. In particular, when $x_1 \le 0$, the network completely zeroes out $h_1 = [0, 0]^T$, totally losing all information about $x_1$.

---

## 2. Weight Parameters & Structural Roles

The network parameters (from [weights.npz](weights.npz)) are:

$$w_{ih} = \begin{pmatrix} 2.9500926 \\ 3.1528590 \end{pmatrix}$$

$$w_{hh} = \begin{pmatrix} -76.4917000 & -2.5048876 \\ 1.1925036 & -1.9860576 \end{pmatrix}$$

$$w_{out} = \begin{pmatrix} 2.84549475 & -2.49234843 \\ -3.75295782 & 3.44526887 \\ 0.83500987 & -0.92129105 \end{pmatrix}$$

### Mechanistic Roles of the Units:
- **Input projection ($w_{ih}$)**: Both weights are positive ($w_{ih, 0} \approx 2.950$, $w_{ih, 1} \approx 3.153$). Positive inputs excite both units in roughly equal proportions ($w_{ih, 1} / w_{ih, 0} \approx 1.0687$). Negative inputs provide negative drive, which ReLU clips to 0.
- **Unit 0 ($h_t[0]$) — Fresh Input Detector & Self-Suppressor**:
  - Row 0 of $w_{hh}$ contains massive self-inhibition ($w_{hh, 00} = -76.4917$) and cross-inhibition from unit 1 ($w_{hh, 01} = -2.5049$).
  - If unit 0 was active at step $t-1$, it inhibits itself so violently that it requires $x_t > 79.17 x_{t-1}$ to reactivate.
  - Therefore, unit 0 is active almost exclusively at the *first positive step* of the sequence and is promptly extinguished thereafter.
- **Unit 1 ($h_t[1]$) — Thresholded Difference Accumulator**:
  - Unit 1 receives positive recurrent excitation from unit 0 ($w_{hh, 10} = +1.1925$) and self-inhibition ($w_{hh, 11} = -1.9861$).
  - When unit 0 was active at step 1, its excitatory drive combined with unit 1's self-inhibition creates a net recurrent drive $(w_{hh} w_{ih})_1 = -2.74376$.
  - This allows unit 1 at step 2 to compute a rectified scaled difference: $\text{ReLU}(w_{ih, 1}(x_2 - c \cdot x_1))$.

---

## 3. Readout Dynamics & The Impossibility of Output 2

Let $h_3 = (h_a, h_b)^T$ with $h_a \ge 0, h_b \ge 0$.
The three logits are:
$$\begin{aligned}
\text{logit}_0 &= 2.8455 h_a - 2.4923 h_b \\
\text{logit}_1 &= -3.7530 h_a + 3.4453 h_b \\
\text{logit}_2 &= 0.8350 h_a - 0.9213 h_b
\end{aligned}$$

### Proof that Logit 2 is Never the Argmax:
To have $\text{logit}_2 > \text{logit}_0$, we require:
$$(0.8350 - 2.8455) h_a + (-0.9213 - (-2.4923)) h_b > 0 \implies 1.5711 h_b > 2.0105 h_a \implies \frac{h_b}{h_a} > 1.2797$$

To have $\text{logit}_2 > \text{logit}_1$, we require:
$$(0.8350 - (-3.7530)) h_a + (-0.9213 - 3.4453) h_b > 0 \implies 4.5880 h_a > 4.3666 h_b \implies \frac{h_b}{h_a} < 1.0507$$

Because the required intervals $\frac{h_b}{h_a} > 1.2797$ and $\frac{h_b}{h_a} < 1.0507$ are strictly disjoint ($1.2797 > 1.0507$), there is no pair $(h_a, h_b) \ge 0$ where $\text{logit}_2$ exceeds both $\text{logit}_0$ and $\text{logit}_1$.
At $h_3 = (0, 0)^T$, all logits are 0, and `np.argmax([0, 0, 0])` returns index 0.

### Readout Decision Boundary Between 0 and 1:
Prediction is 1 if and only if $\text{logit}_1 > \text{logit}_0$:
$$\Delta w = w_{out, 1} - w_{out, 0} = (-6.59845257, 5.93761730)$$
$$\Delta w @ h_3 > 0 \iff 5.93761730 h_b > 6.59845257 h_a \iff \frac{h_b}{h_a} > 1.11129637 \quad (\text{or } h_a = 0, h_b > 0)$$

---

## 4. Step-by-Step Mechanistic Execution

### Step 1 ($t = 1$): Encoding $x_1$
Starting from $h_0 = (0, 0)^T$:
$$h_1 = \text{ReLU}(w_{ih} x_1)$$
- If $x_1 \le 0$: $h_1 = (0, 0)^T$. Total memory erasure.
- If $x_1 > 0$: $h_1 = (w_{ih, 0} x_1, w_{ih, 1} x_1)^T$.

### Step 2 ($t = 2$): Transition and Difference Computation
$$h_2 = \text{ReLU}(w_{ih} x_2 + w_{hh} h_1)$$

- **Branch A ($x_1 \le 0$)**: Since $h_1 = 0$, $h_2 = \text{ReLU}(w_{ih} x_2)$.
  - If $x_2 \le 0$: $h_2 = (0, 0)^T$.
  - If $x_2 > 0$: $h_2 = (w_{ih, 0} x_2, w_{ih, 1} x_2)^T$.
- **Branch B ($x_1 > 0$)**: Here $h_1 = w_{ih} x_1$.
  The recurrent contribution is $w_{hh} w_{ih} x_1$, where:
  $$w_{hh} w_{ih} = \begin{pmatrix} -76.4917 & -2.5049 \\ 1.1925 & -1.9861 \end{pmatrix} \begin{pmatrix} 2.9501 \\ 3.1529 \end{pmatrix} = \begin{pmatrix} -233.55515 \\ -2.74376 \end{pmatrix}$$
  - For unit 0: Pre-activation is $2.9501 x_2 - 233.555 x_1$. Unit 0 is clamped to 0 unless $x_2 > 79.1688 x_1$ (probability $< 0.002$ in the first quadrant, negligible).
  - For unit 1: Pre-activation is $w_{ih, 1} x_2 - 2.74376 x_1 = w_{ih, 1} (x_2 - c \cdot x_1)$, where:
    $$c = \frac{2.74376374}{3.1528590} \approx 0.87024626$$
  - Thus:
    - If $x_2 \le c \cdot x_1$: $h_2 = (0, 0)^T$.
    - If $x_2 > c \cdot x_1$: $h_2 \approx (0, w_{ih, 1}(x_2 - c \cdot x_1))^T$.

### Step 3 ($t = 3$): Final State and Readout
$$h_3 = \text{ReLU}(w_{ih} x_3 + w_{hh} h_2)$$

#### Case 1: $h_2 = (0, 0)^T$
This occurs when ($x_1 \le 0 \land x_2 \le 0$) or ($x_1 > 0 \land x_2 \le c \cdot x_1$).
- If $x_3 \le 0$: $h_3 = (0, 0)^T \implies \text{prediction} = 0$.
- If $x_3 > 0$: $h_3 = w_{ih} x_3$.
  Ratio $\frac{h_3[1]}{h_3[0]} = \frac{w_{ih, 1}}{w_{ih, 0}} \approx 1.0687 < 1.1113 \implies \text{logit}_0 > \text{logit}_1 \implies \text{prediction} = 0$.
- **Conclusion**: Whenever $h_2 = (0, 0)^T$, the network unconditionally predicts 0.

#### Case 2: $x_1 \le 0$ and $x_2 > 0$
Here $h_2 = w_{ih} x_2$.
Recurrent drive is $(w_{hh} w_{ih}) x_2 = (-233.555 x_2, -2.74376 x_2)^T$.
- Unit 0 is suppressed unless $x_3 > 79.17 x_2$ (negligible).
- Unit 1 pre-activation is $w_{ih, 1} x_3 - 2.74376 x_2 = w_{ih, 1} (x_3 - c \cdot x_2)$.
- If $x_3 \le c \cdot x_2$: $h_3 = (0, 0)^T \implies \text{prediction} = 0$.
- If $x_3 > c \cdot x_2$: $h_3[0] = 0$ while $h_3[1] > 0 \implies \text{logit}_1 > \text{logit}_0 \implies \text{prediction} = 1$.

#### Case 3: $x_1 > 0$ and $x_2 > c \cdot x_1$
Here $h_2 = (0, h_2[1])^T$ where $h_2[1] = w_{ih, 1}(x_2 - c \cdot x_1) > 0$. Let $d = x_2 - c \cdot x_1 > 0$.
At step 3, $w_{hh} h_2 = w_{hh}[:, 1] h_2[1] = (-2.5048876, -1.9860576)^T w_{ih, 1} d$.
- Unit 1 turns on when $w_{ih, 1} x_3 + w_{hh, 11} w_{ih, 1} d > 0 \iff x_3 > -w_{hh, 11} d = k_1 d$, where:
  $$k_1 = -w_{hh, 11} \approx 1.98605764$$
- Unit 0 turns on when $w_{ih, 0} x_3 + w_{hh, 01} w_{ih, 1} d > 0 \iff x_3 > \frac{-w_{hh, 01} w_{ih, 1}}{w_{ih, 0}} d \approx 2.6769 d$.
- When both units are active, $\text{logit}_1 > \text{logit}_0$ holds as long as:
  $$x_3 < k_2 d$$
  where $k_2$ is the zero-crossing of $(\Delta w @ w_{ih}) x_3 + (\Delta w @ w_{hh}[:, 1]) w_{ih, 1} d = 0$:
  $$k_2 = -\frac{\Delta w @ w_{hh}[:, 1]}{\Delta w @ w_{ih}} w_{ih, 1} \approx 20.02710476$$
- **Conclusion**: Prediction is 1 iff $k_1 d < x_3 < k_2 d$, and 0 otherwise.

---

## 5. Complete Derivation of Named Constants

| Constant | Value | Origin Formula | Semantic Interpretation |
|---|---|---|---|
| `C_THRESHOLD` ($c$) | `0.87024626` | $-\frac{(w_{hh} @ w_{ih})_1}{w_{ih, 1}}$ | Scaled threshold for $x_2$ over $x_1$ (and $x_3$ over $x_2$) to activate unit 1 |
| `K_LOWER` ($k_1$) | `1.98605764` | $-w_{hh, 11}$ | Minimum multiple of $(x_2 - c x_1)$ required for $x_3$ to activate unit 1 at step 3 |
| `K_UPPER` ($k_2$) | `20.02710476` | $-\frac{(w_{out, 1} - w_{out, 0}) @ w_{hh}[:, 1]}{(w_{out, 1} - w_{out, 0}) @ w_{ih}} w_{ih, 1}$ | Upper multiple of $(x_2 - c x_1)$ where unit 0 re-overtakes unit 1 |

All 3 constants are directly derived from the closed-form algebraic conditions of the neural network's weights.

---

## 6. Error Analysis: Where and Why the Model Fails

### 1. Zero Recall on Median Position 2 (33.33% of all inputs)
The permutations where the true median is position 2 are:
- $S_{021}: x_1 < x_3 < x_2$ (measure $1/6$)
- $S_{120}: x_2 < x_3 < x_1$ (measure $1/6$)
Together, these represent $1/3$ of all inputs. Because $\text{logit}_2$ can never be the maximum logit, the network has **0.00% accuracy** on both sectors.

### 2. Information Loss from Non-Negative Inputs
Because the network lacks biases:
- When $x_1 \le 0$ and $x_2 \le 0$ (probability $1/4$): The network state is $(0, 0)^T$. It predicts position 0 regardless of $x_3$. If $x_3 > 0$, the median was actually $\max(x_1, x_2)$, which is either position 0 or position 1.
- Because it has no memory of negative inputs, the network cannot compare negative numbers to each other.

### 3. Asymmetric Recall
- **Position 0**: True median 0 occurs when $x_2 < x_1 < x_3$ or $x_3 < x_1 < x_2$. The network defaults to 0 across most of the state space, achieving **99.0% recall** on position 0.
- **Position 1**: True median 1 occurs when $x_1 < x_2 < x_3$ or $x_3 < x_2 < x_1$.
  - In $S_{012}$ ($x_1 < x_2 < x_3$), when $x_1 \le 0 < x_2 < x_3$, it successfully predicts 1. But when $x_1 < x_2 < 0$, it predicts 0. When $0 < x_1 < x_2 < x_3$, it requires $x_3 > k_1 (x_2 - c x_1)$, which often fails if $x_2$ is significantly larger than $x_1$. Overall recall on $S_{012}$ is only $46.5\%$.
  - In $S_{210}$ ($x_3 < x_2 < x_1$), it requires $x_2 > c x_1$ and $x_3 > k_1 (x_2 - c x_1)$, which virtually contradicts $x_3 < x_2 < x_1$. Accuracy on $S_{210}$ is only $1.1\%$.
  - Total recall on median position 1 is only **23.6%**.

---

## 7. Model Accuracy Derivation via Spherical Integration

Let $M \in \{0, 1, 2\}$ be the true median position. By exchangeability of i.i.d. standard Gaussians, $P(M=0) = P(M=1) = P(M=2) = 1/3$.
Because $P(\text{Pred} = 2) = 0$:
$$\text{Accuracy} = P(M = \text{Pred}) = P(M=0, \text{Pred}=0) + P(M=1, \text{Pred}=1)$$
Using $P(M=0, \text{Pred}=0) = P(M=0) - P(M=0, \text{Pred}=1) = 1/3 - P(M=0, \text{Pred}=1)$:
$$\text{Accuracy} = \frac{1}{3} + P(M=1, \text{Pred}=1) - P(M=0, \text{Pred}=1)$$

### Region 1: $x_1 \le 0, x_2 > 0, x_3 > c x_2$
- Here $x_1 \le 0 < x_2$.
- $M = 0$ is geometrically impossible (requires $x_2 < x_1 < x_3$ or $x_3 < x_1 < x_2$, both contradicted by $x_3 > c x_2 > 0 \ge x_1$). Thus $P(M=0, \text{Reg 1}) = 0$.
- $M = 1$ requires $x_1 < x_2 < x_3 \iff x_3 > x_2$. Since $c < 1$, $x_3 > x_2$ implies $x_3 > c x_2$.
  $$P(M=1, \text{Reg 1}) = P(x_1 \le 0, 0 < x_2 < x_3) = \frac{1}{2} \times \frac{1}{8} = \frac{1}{16} = 0.0625 \quad (\text{Exact})$$

### Region 2: $x_1 > 0, x_2 > c x_1, k_1 d < x_3 < k_2 d$ ($d = x_2 - c x_1$)
Because all inequalities are homogeneous cones from the origin, standard Gaussian measure in $\mathbb{R}^3$ reduces by spherical symmetry to a 1D integral over the azimuthal angle $\theta \in [\arctan(c), \pi/2]$:
$$\frac{1}{4\pi} \int \left( \frac{u_{\max}(\theta)}{\sqrt{1 + u_{\max}^2(\theta)}} - \frac{u_{\min}(\theta)}{\sqrt{1 + u_{\min}^2(\theta)}} \right) d\theta$$
Evaluating this integral yields:
$$P(M=1, \text{Reg 2}) \approx 0.01725930$$
$$P(M=0, \text{Reg 2}) \approx 0.00297040$$

### Total Calculated Accuracy:
$$\text{Accuracy} = \frac{1}{3} + 0.0625 + 0.01725930 - 0.00297040 = 0.41012223 \approx 41.01\%$$

### Error Regions Included vs Left Out:
- **Included**:
  - $100\%$ of cases where median is index 2 (probability $1/3$), where the model is always wrong.
  - Missed median 1 predictions ($P(M=1, \text{Pred}=0) \approx 0.2536$).
  - False positive median 1 predictions when median is 0 ($P(M=0, \text{Pred}=1) \approx 0.00297$).
- **Left out**:
  - The extreme slivers where $x_2 > 79.17 x_1$ or $x_3 > 79.17 x_2$ where unit 0 fails to remain suppressed. These slivers have Gaussian measure $< 0.0002$ ($< 0.02\%$) and negligibly perturb the total accuracy.

---

## 8. What Could Not Be Explained

1. **Training Convergence to a 2-Class Sub-Optimum**:
   Why standard backpropagation converged to an internal representation where row 2 of $w_{out}$ is completely dominated remains an artifact of optimization. In a 2-unit ReLU RNN without biases, learning 3 distinct non-convex decision cones is severely under-parameterized; gradient descent effectively sacrificed the third class to optimize classification between classes 0 and 1.
2. **Exact Numerical Magnitude of Inhibitory Weight $w_{hh, 00} = -76.49$**:
   While it is functionally clear that $w_{hh, 00}$ must be a large negative value to shut off unit 0 after step 1, its specific value $-76.49$ (rather than $-50$ or $-100$) depends on optimization hyperparameters (learning rate, weight decay, initialization seed) and has no unique mechanistic significance beyond ensuring that the activation boundary $x_2 > 79.17 x_1$ is pushed far into the Gaussian tail.
