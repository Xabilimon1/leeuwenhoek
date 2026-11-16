# Mechanistic Explanation of the 2-Hidden-Unit Median RNN

## 1. Executive Summary

The network is a bias-free 2-hidden-unit recurrent neural network (RNN) with ReLU activations and 12 parameters trained to predict the index ($0, 1$, or $2$) of the median of three real inputs $(x_1, x_2, x_3) \sim \mathcal{N}(0, I_3)$.

Our analysis reveals that the trained network did **not** learn a general median-finding algorithm. Instead, it converged to an asymmetric, degenerate heuristic with a theoretical accuracy of approximately **41.01%** (on i.i.d. standard Gaussians):
1. **Position 2 is mathematically shadowed**: The output readout weights are configured such that position 2 is strictly dominated by position 0 and position 1 across the entire non-negative hidden state quadrant $\mathbb{R}_{\ge 0}^2$. The network **never** predicts position 2 on any input.
2. **Default to Position 0**: The network defaults to predicting position 0 (the first input) whenever the hidden state collapses to $(0, 0)$ or when hidden unit 0 dominates hidden unit 1. Because ReLU without bias clamps all negative activations to zero, negative inputs obliterate information, frequently resetting the hidden state to $(0, 0)$.
3. **Selective Detection of Position 1**: The network predicts position 1 if and only if $x_2$ is sufficiently positive and exceeds a scaled fraction of $x_1$, and $x_3$ falls within a specific intermediate band relative to the residual of $x_2$.

---

## 2. Model Architecture and Parameters

The network equations are:
$$h_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
$$h_t = \text{relu}(w_{\text{ih}} x_t + w_{\text{hh}} h_{t-1}) \quad \text{for } t \in \{1, 2, 3\}$$
$$\text{logits} = w_{\text{out}} h_3$$
$$\text{prediction} = \text{argmax}(\text{logits})$$

The exact weights loaded from `weights.npz` are:
- **Input weights** $w_{\text{ih}} \in \mathbb{R}^2$:
  $$w_{\text{ih}} = \begin{pmatrix} w_0 \\ w_1 \end{pmatrix} = \begin{pmatrix} 2.9500926 \\ 3.152859 \end{pmatrix}$$
- **Recurrent weights** $w_{\text{hh}} \in \mathbb{R}^{2 \times 2}$:
  $$w_{\text{hh}} = \begin{pmatrix} U_{00} & U_{01} \\ U_{10} & U_{11} \end{pmatrix} = \begin{pmatrix} -76.4917 & -2.5048876 \\ 1.1925036 & -1.9860576 \end{pmatrix}$$
- **Output readout weights** $w_{\text{out}} \in \mathbb{R}^{3 \times 2}$:
  $$w_{\text{out}} = \begin{pmatrix} W_{00} & W_{01} \\ W_{10} & W_{11} \\ W_{20} & W_{21} \end{pmatrix} = \begin{pmatrix} 2.8454947 & -2.4923484 \\ -3.7529578 & 3.4452689 \\ 0.8350099 & -0.92129105 \end{pmatrix}$$

Notice that the architecture has no bias vectors. Consequently, the network is strictly positive-homogeneous:
$$\forall c > 0, \quad h_t(c \cdot x) = c \cdot h_t(x), \quad \text{logits}(c \cdot x) = c \cdot \text{logits}(x)$$
The prediction depends solely on the ray direction of $x = (x_1, x_2, x_3)$ on the unit sphere $S^2$.

---

## 3. Derivation of Constants from the Weights

All constants used in the explanation and the twin implementation are derived analytically from the weight matrices:

### Constant 1: Output Decision Ratio $r^*$
In the non-negative quadrant $h_3 \in \mathbb{R}_{\ge 0}^2$:
$$\text{logit}_0 = 2.8454947 h_{3,0} - 2.4923484 h_{3,1}$$
$$\text{logit}_1 = -3.7529578 h_{3,0} + 3.4452689 h_{3,1}$$
$$\text{logit}_2 = 0.8350099 h_{3,0} - 0.92129105 h_{3,1}$$

The boundary where $\text{logit}_1 > \text{logit}_0$ is:
$$\text{logit}_1 - \text{logit}_0 = -(W_{00} - W_{10}) h_{3,0} + (W_{11} - W_{01}) h_{3,1} > 0$$
$$-6.5984525 h_{3,0} + 5.9376173 h_{3,1} > 0 \iff \frac{h_{3,0}}{h_{3,1}} < r^*$$
where:
$$r^* = \frac{W_{11} - W_{01}}{W_{00} - W_{10}} = \frac{3.4452689 - (-2.4923484)}{2.8454947 - (-3.7529578)} = \frac{5.9376173}{6.5984525} \approx \mathbf{0.89984995}$$

### Proof that Position 2 is Never Predicted
For position 2 to win, it must simultaneously exceed both logit 0 and logit 1:
1. $\text{logit}_2 > \text{logit}_0 \iff -2.010485 h_{3,0} + 1.5710573 h_{3,1} > 0 \iff \frac{h_{3,0}}{h_{3,1}} < \frac{1.5710573}{2.010485} \approx 0.781432$
2. $\text{logit}_2 > \text{logit}_1 \iff 4.587968 h_{3,0} - 4.366560 h_{3,1} > 0 \iff \frac{h_{3,0}}{h_{3,1}} > \frac{4.366560}{4.587968} \approx 0.951742$

Because $0.951742 > 0.781432$, these two conditions are mutually exclusive. Furthermore, at $h_3 = (0, 0)$, $\text{logits} = (0, 0, 0)$ and `argmax` tie-breaking returns index 0. Hence, **the network can never predict position 2**.

### Constant 2: Recurrent Relative Threshold $C_{\text{REL}}$
When $h_1 = w_{\text{ih}} x_1$ ($x_1 > 0$), the recurrent input to step 2 is $w_{\text{hh}} w_{\text{ih}} x_1$:
$$v = w_{\text{hh}} w_{\text{ih}} = \begin{pmatrix} -76.4917 \times 2.9500926 - 2.5048876 \times 3.152859 \\ 1.1925036 \times 2.9500926 - 1.9860576 \times 3.152859 \end{pmatrix} = \begin{pmatrix} -233.55515 \\ -2.743764 \end{pmatrix}$$
For unit 1 at step 2:
$$\text{pre}_2[1] = w_1 x_2 + v_1 x_1 = w_1 (x_2 - C_{\text{REL}} x_1)$$
where:
$$C_{\text{REL}} = -\frac{v_1}{w_1} = -\frac{-2.743764}{3.152859} \approx \mathbf{0.87024635}$$

For unit 0 at step 2:
$$\text{pre}_2[0] = w_0 x_2 + v_0 x_1 = w_0 (x_2 - \kappa_0 x_1), \quad \kappa_0 = -\frac{v_0}{w_0} = \frac{233.55515}{2.9500926} \approx 79.168755$$
Because $U_{00} = -76.4917$ is massive, unit 0 at step 2 is completely shut down unless $x_2 > 79.17 x_1$.

### Constant 3: Lower Activation Multiplier $D_{\text{LOW}}$
When $x_1 > 0$ and $x_2 > C_{\text{REL}} x_1$, unit 0 is 0 and unit 1 is $h_{2,1} = w_1 \Delta$, where $\Delta = x_2 - C_{\text{REL}} x_1 > 0$.
At step 3, the recurrent input is $w_1 \Delta \cdot w_{\text{hh}}[:, 1]$:
$$\text{pre}_3[1] = w_1 x_3 + U_{11} w_1 \Delta = w_1 (x_3 - D_{\text{LOW}} \Delta)$$
where:
$$D_{\text{LOW}} = -U_{11} = -(-1.9860576) = \mathbf{1.9860576}$$
Thus, unit 1 can only activate if $x_3 > D_{\text{LOW}} \Delta$.

### Constant 4: Upper Readout Crossover Multiplier $D_{\text{HIGH}}$
For unit 0 at step 3:
$$\text{pre}_3[0] = w_0 x_3 + U_{01} w_1 \Delta = w_0 (x_3 - \lambda_0 \Delta)$$
where $\lambda_0 = -\frac{U_{01} w_1}{w_0} = \frac{2.5048876 \times 3.152859}{2.9500926} \approx 2.6770542$.

When $x_3 > \lambda_0 \Delta$, both units are active: $h_{3,0} = w_0 (x_3 - \lambda_0 \Delta)$ and $h_{3,1} = w_1 (x_3 - D_{\text{LOW}} \Delta)$.
As $x_3$ increases, $h_{3,0} / h_{3,1}$ increases toward $w_0 / w_1 = 0.9357 > r^*$. The ratio crosses $r^*$ at:
$$w_0 (x_3 - \lambda_0 \Delta) = r^* w_1 (x_3 - D_{\text{LOW}} \Delta) \iff x_3 = D_{\text{HIGH}} \Delta$$
where:
$$D_{\text{HIGH}} = \frac{w_0 \lambda_0 - r^* w_1 D_{\text{LOW}}}{w_0 - r^* w_1} = \frac{2.9500926 \times 2.6770542 - 0.89984995 \times 3.152859 \times 1.9860576}{2.9500926 - 0.89984995 \times 3.152859} \approx \mathbf{20.02711}$$
For $x_3 \ge D_{\text{HIGH}} \Delta$, unit 0 overtakes unit 1 in the logits and the prediction flips back to 0.

---

## 4. Step-by-Step Mechanistic Trace

### Step 1 ($t = 1$, input $x_1$):
- Initial state: $h_0 = (0, 0)$.
- Activation: $h_1 = \text{relu}(w_{\text{ih}} x_1)$.
- If $x_1 \le 0$: both components are non-positive, so $h_1 = (0, 0)$. Negative $x_1$ is completely forgotten.
- If $x_1 > 0$: $h_1 = x_1 \cdot (2.950, 3.153)^T$. Both units fire in fixed ratio $\approx 0.9357$.

### Step 2 ($t = 2$, input $x_2$):
- If $x_1 \le 0$: $h_1 = (0, 0)$, so recurrence is zero. The network restarts at $x_2$:
  * If $x_2 \le 0 \implies h_2 = (0, 0)$.
  * If $x_2 > 0 \implies h_2 = x_2 \cdot w_{\text{ih}}$.
- If $x_1 > 0$: $h_1$ injects strong negative inhibition $w_{\text{hh}} h_1 = x_1 \cdot (-233.56, -2.744)^T$.
  * Unit 0 is suppressed to 0 (since $x_2 < 79.17 x_1$ almost surely).
  * Unit 1 computes $\text{relu}(w_1 (x_2 - C_{\text{REL}} x_1))$.
  * If $x_2 \le C_{\text{REL}} x_1$: hidden state collapses to $h_2 = (0, 0)$.
  * If $x_2 > C_{\text{REL}} x_1$: hidden state is $(0, w_1 \Delta)$, encoding the excess $\Delta = x_2 - C_{\text{REL}} x_1$.

### Step 3 ($t = 3$, input $x_3$):
- **If $h_2 = (0, 0)$**:
  * If $x_3 \le 0 \implies h_3 = (0, 0) \implies \text{logits} = (0, 0, 0) \implies \text{pred} = 0$.
  * If $x_3 > 0 \implies h_3 = x_3 w_{\text{ih}} \implies \text{logits} = x_3 (0.536, -0.209, -0.441)^T \implies \text{pred} = 0$.
  * **Rule**: Whenever $h_2 = (0, 0)$, the prediction is guaranteed to be 0 regardless of $x_3$.
- **If $h_2 = x_2 w_{\text{ih}}$** ($x_1 \le 0, x_2 > 0$):
  * Same recurrence as step 2: unit 0 suppressed, unit 1 active when $x_3 > C_{\text{REL}} x_2$.
  * When unit 1 is active alone, $\text{logits} \propto (-2.492, +3.445, -0.921)^T$, giving $\text{pred} = 1$.
- **If $h_2 = (0, w_1 \Delta)$** ($x_1 > 0, \Delta = x_2 - C_{\text{REL}} x_1 > 0$):
  * Unit 1 activates if $x_3 > D_{\text{LOW}} \Delta$.
  * Unit 0 activates if $x_3 > \lambda_0 \Delta$.
  * For $D_{\text{LOW}} \Delta < x_3 < D_{\text{HIGH}} \Delta$, unit 1 dominates readout, giving $\text{pred} = 1$.
  * For $x_3 \ge D_{\text{HIGH}} \Delta$, unit 0 dominates readout, returning $\text{pred} = 0$.

---

## 5. Complete Decision Rule of the Network

The network's output is governed by the following rules:
- **Default**: Predict **0**.
- **Predict 1** if and only if either:
  1. **Region A**: $x_1 \le 0$, $x_2 > 0$, and $x_3 > C_{\text{REL}} x_2$.
  2. **Region B**: $x_1 > 0$, $\Delta = x_2 - C_{\text{REL}} x_1 > 0$, and $D_{\text{LOW}} \Delta < x_3 < D_{\text{HIGH}} \Delta$.
- **Predict 2**: Never.

---

## 6. Error Analysis

The median of 3 i.i.d. Gaussian inputs is uniformly distributed across positions $\{0, 1, 2\}$, each having probability $1/3 \approx 33.33\%$.

### Error Region 1: True Median is Position 2 ($P = 1/3 \approx 0.333333$)
Because the network never outputs 2, it is wrong on 100% of these inputs.
- When $x_3$ is the median, the network predicts 0 ($P \approx 0.32652$) or predicts 1 ($P \approx 0.00681$).

### Error Region 2: True Median is Position 1, but Network Predicts 0 ($P \approx 0.253574$)
For $x_2$ to be the median, either $x_1 \le x_2 \le x_3$ or $x_3 \le x_2 \le x_1$.
- The network only predicts 1 inside Regions A and B ($P(\text{pred}=1 \text{ and med}=1) \approx 0.079759$).
- In all other configurations where $x_2$ is the median (e.g. $x_3 \le x_2 \le x_1$, or $x_1 \le x_2 \le x_3$ where $x_2$ is not large enough to beat $C_{\text{REL}} x_1$, or where $x_3$ fails the upper/lower bounds), the network defaults to 0.

### Error Region 3: True Median is Position 0, but Network Predicts 1 ($P \approx 0.002970$)
When $x_1$ is the median, the network predicts 0 with over $99.1\%$ accuracy ($P \approx 0.330363$).
It makes a rare mistake in the narrow angular wedge where $x_1 > 0$, $C_{\text{REL}} x_1 < x_2 < x_1$, and $x_1 < x_3 < D_{\text{HIGH}} \Delta$, falsely triggering prediction 1.

### Total Theoretical Accuracy:
$$\text{Accuracy} = P(\text{pred}=0 \ \& \ \text{med}=0) + P(\text{pred}=1 \ \& \ \text{med}=1) \approx 0.330363 + 0.079759 = \mathbf{0.410122} \ (41.01\%)$$

---

## 7. What Could Not Be Explained

1. **Optimization Convergence to a 2-Class Local Minimum**:
   A 2-unit ReLU RNN without bias has severe architectural limitations. Because ReLU has zero gradient for negative pre-activations, and without bias terms the network cannot shift inputs to positive regions, the network easily traps units in inactive states. Why gradient descent settled into an asymmetric minimum that completely eliminated class 2 rather than learning a cyclic permutation symmetry remains an empirical property of the loss landscape and initialization.
2. **Exact Weight Ratios**:
   While the functional behavior is governed by the ratios $C_{\text{REL}}$, $D_{\text{LOW}}$, $D_{\text{HIGH}}$, and $r^*$, the specific overall scale of the weights (e.g. $U_{00} \approx -76.5$ vs $U_{10} \approx 1.19$) is determined by the learning rate, weight decay, and duration of training.
