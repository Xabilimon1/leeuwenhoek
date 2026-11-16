# Mechanistic Explanation of the 2-Unit Recurrent Neural Network

## 1. Executive Summary & Overview

The network is a 2-hidden-unit recurrent neural network (RNN) with ReLU activations and no biases, trained to predict the index ($0, 1, \text{ or } 2$) of the median of three real inputs $x = (x_0, x_1, x_2)$ drawn from i.i.d. standard Gaussian distributions $\mathcal{N}(0, 1)$.

The recurrent equations are:
$$h_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$h_t = \text{ReLU}(w_{ih} x_{t-1} + W_{hh} h_{t-1}), \quad t \in \{1, 2, 3\}$$
$$\text{logits} = w_{out} h_3$$
$$\text{prediction} = \operatorname{argmax}(\text{logits})$$

### The Exact Weights
From `weights.npz`:
- Input weights $w_{ih} \in \mathbb{R}^2$:
  $$w_{ih} = \begin{bmatrix} u_0 \\ u_1 \end{bmatrix} = \begin{bmatrix} 2.9500926 \\ 3.1528590 \end{bmatrix}$$
- Recurrent transition matrix $W_{hh} \in \mathbb{R}^{2 \times 2}$:
  $$W_{hh} = \begin{bmatrix} W_{00} & W_{01} \\ W_{10} & W_{11} \end{bmatrix} = \begin{bmatrix} -76.4917 & -2.5048876 \\ 1.1925036 & -1.9860576 \end{bmatrix}$$
- Readout matrix $w_{out} \in \mathbb{R}^{3 \times 2}$:
  $$w_{out} = \begin{bmatrix} w_{out, 00} & w_{out, 01} \\ w_{out, 10} & w_{out, 11} \\ w_{out, 20} & w_{out, 21} \end{bmatrix} = \begin{bmatrix} 2.8454947 & -2.4923484 \\ -3.7529578 & 3.4452689 \\ 0.8350099 & -0.92129105 \end{bmatrix}$$

### High-Level Computational Strategy
Because the network has only 2 hidden units and zero biases, computing the true median of 3 arbitrary real numbers is impossible (sorting 3 numbers requires tracking multiple pairwise comparisons and sign permutations). The network settled into an asymmetric local minimum:
1. **Complete Collapse of Position 2**: The readout vector for position 2 is strictly dominated by the convex geometry of positions 0 and 1 in the non-negative quadrant $\mathbb{R}_{\ge 0}^2$. The network **never** predicts position 2 under any input.
2. **Default Fallback to Position 0**: When the hidden state $h_3$ is quiescent ($h_3 = 0$) or when Unit 0 dominates Unit 1, the network outputs position 0. This secures an automatic baseline accuracy of $1/3$ on inputs where the true median is 0.
3. **Ascending Sequence Detector for Position 1**: The network uses Unit 1 to detect ascending sequences ($x_0 < x_1 < x_2$) where $x_1$ is the true median. It fires position 1 only when $x_1$ and $x_2$ demonstrate specific positive margins over the preceding inputs.

---

## 2. Readout Geometry & Decision Rule

At time step $t=3$, the output logits are linear functions of $h_3 = [h_{3, 0}, h_{3, 1}]^T$:
$$\text{logit}_0 = 2.8454947 \, h_{3, 0} - 2.4923484 \, h_{3, 1}$$
$$\text{logit}_1 = -3.7529578 \, h_{3, 0} + 3.4452689 \, h_{3, 1}$$
$$\text{logit}_2 = 0.8350099 \, h_{3, 0} - 0.92129105 \, h_{3, 1}$$

Because $h_3 = \text{ReLU}(z_3) \ge 0$, the state space is restricted to the first quadrant $\mathbb{R}_{\ge 0}^2 = \{(h_{3, 0}, h_{3, 1}) : h_{3, 0} \ge 0, h_{3, 1} \ge 0\}$.

### Why Position 2 is Never Predicted
For position 2 to be chosen, its logit must exceed both $\text{logit}_0$ and $\text{logit}_1$:
1. $\text{logit}_2 > \text{logit}_0$:
   $$(0.8350099 - 2.8454947) h_{3, 0} > (-2.4923484 - (-0.92129105)) h_{3, 1}$$
   $$-2.0104848 \, h_{3, 0} > -1.57105735 \, h_{3, 1} \iff \frac{h_{3, 1}}{h_{3, 0}} > \frac{2.0104848}{1.57105735} \approx 1.27970$$

2. $\text{logit}_2 > \text{logit}_1$:
   $$(0.8350099 - (-3.7529578)) h_{3, 0} > (3.4452689 - (-0.92129105)) h_{3, 1}$$
   $$4.5879677 \, h_{3, 0} > 4.36655995 \, h_{3, 1} \iff \frac{h_{3, 1}}{h_{3, 0}} < \frac{4.5879677}{4.36655995} \approx 1.050705$$

Notice that the two half-spaces are completely disjoint:
$$1.27970 < \frac{h_{3, 1}}{h_{3, 0}} < 1.050705 \implies \text{Impossible!}$$
Furthermore, if $h_{3, 0} = 0$ and $h_{3, 1} > 0$, $\text{logit}_1 > 0 > \text{logit}_2$. If $h_{3, 1} = 0$ and $h_{3, 0} > 0$, $\text{logit}_0 > \text{logit}_2 > 0$. If $h_3 = (0, 0)$, all three logits are 0, and `np.argmax([0, 0, 0])` returns index 0.

Therefore, **$\text{logit}_2$ is mathematically dominated across the entire non-negative quadrant**, and position 2 is never predicted.

### The Competition Between Position 0 and Position 1
Since position 2 never wins, the decision reduces strictly to $\text{logit}_1 > \text{logit}_0$:
$$(-3.7529578 - 2.8454947) h_{3, 0} + (3.4452689 - (-2.4923484)) h_{3, 1} > 0$$
$$-6.5984525 \, h_{3, 0} + 5.9376173 \, h_{3, 1} > 0 \iff h_{3, 1} > C_{out} \, h_{3, 0}$$
where:
$$C_{out} = \frac{w_{out, 00} - w_{out, 10}}{w_{out, 11} - w_{out, 01}} = \frac{2.8454947 - (-3.7529578)}{3.4452689 - (-2.4923484)} \approx 1.11129635$$

The prediction rule is:
$$\text{prediction} = \begin{cases} 1 & \text{if } h_{3, 1} > C_{out} \, h_{3, 0} \text{ and } h_{3, 1} > 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Step-by-Step Mechanistic Trace

### Time Step 1 ($t=1$, input $x_0$)
The initial hidden state is $h_0 = [0, 0]^T$.
$$h_1 = \text{ReLU}(w_{ih} x_0) = \begin{bmatrix} \text{ReLU}(u_0 x_0) \\ \text{ReLU}(u_1 x_0) \end{bmatrix}$$
- **If $x_0 \le 0$**: Due to the absence of bias, the ReLU squashes the negative signal to zero:
  $$h_1 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
  The network loses all information about the magnitude of $x_0$, retaining only that it was non-positive.
- **If $x_0 > 0$**: Both units activate proportionally to $x_0$:
  $$h_1 = \begin{bmatrix} u_0 x_0 \\ u_1 x_0 \end{bmatrix} = \begin{bmatrix} 2.9500926 \\ 3.1528590 \end{bmatrix} x_0$$
  Unit 0 serves as an inhibitory trigger for step 2, while Unit 1 tracks positive scale.

---

### Time Step 2 ($t=2$, input $x_1$)
The pre-activation vector is $z_2 = w_{ih} x_1 + W_{hh} h_1$.

#### Case 2A: $x_0 \le 0$ (Uninhibited Step)
Since $h_1 = [0, 0]^T$:
$$z_2 = w_{ih} x_1 = \begin{bmatrix} u_0 x_1 \\ u_1 x_1 \end{bmatrix}$$
- If $x_1 \le 0$: $h_2 = [0, 0]^T$.
- If $x_1 > 0$: $h_2 = \begin{bmatrix} u_0 x_1 \\ u_1 x_1 \end{bmatrix}$. Both units fire uninhibited.

#### Case 2B: $x_0 > 0$ (Inhibited Step)
Since $h_1 = w_{ih} x_0$:
$$z_2 = w_{ih} x_1 + W_{hh} w_{ih} x_0 = \begin{bmatrix} u_0 x_1 + c_0 x_0 \\ u_1 x_1 + c_1 x_0 \end{bmatrix}$$
where:
$$c_0 = W_{00} u_0 + W_{01} u_1 = (-76.4917)(2.9500926) + (-2.5048876)(3.152859) \approx -233.55515$$
$$c_1 = W_{10} u_0 + W_{11} u_1 = (1.1925036)(2.9500926) + (-1.9860576)(3.152859) \approx -2.743764$$

1. **Unit 0 (Massive Self-Inhibition)**:
   $$z_{2, 0} = u_0 x_1 + c_0 x_0 = u_0 (x_1 - \beta x_0)$$
   where:
   $$\beta = -\frac{c_0}{u_0} = -\left(W_{00} + W_{01} \frac{u_1}{u_0}\right) \approx 79.168755$$
   For standard Gaussian inputs, $x_1 > 79.17 x_0$ occurs with probability $< 0.001$. Therefore, almost surely:
   $$h_{2, 0} = 0$$
   **Unit 0 is completely shut down at step 2 when $x_0 > 0$.**

2. **Unit 1 (Relative Comparator)**:
   $$z_{2, 1} = u_1 x_1 + c_1 x_0 = u_1 (x_1 - \alpha x_0)$$
   where:
   $$\alpha = -\frac{c_1}{u_1} = -\left(W_{10} \frac{u_0}{u_1} + W_{11}\right) = \frac{2.743764}{3.152859} \approx 0.87024626$$
   Therefore:
   $$h_{2, 1} = u_1 \max(0, x_1 - \alpha x_0)$$
   **Unit 1 represents the margin by which $x_1$ exceeds $\alpha x_0$.** It is positive if and only if $x_1 > \alpha x_0$.

---

### Time Step 3 ($t=3$, input $x_2$)
The pre-activation vector is $z_3 = w_{ih} x_2 + W_{hh} h_2$.

#### Case 3A: $x_0 \le 0$
- If $x_1 \le 0$: $h_2 = 0 \implies h_3 = \text{ReLU}(w_{ih} x_2)$. If $x_2 \le 0$, $h_3 = 0$. If $x_2 > 0$, $h_{3, 1} / h_{3, 0} = u_1 / u_0 \approx 1.0687 < C_{out} \approx 1.1113$, so the prediction is **0**.
- If $x_1 > 0$: $h_2 = w_{ih} x_1$. Then:
  $$z_3 = w_{ih} x_2 + W_{hh} w_{ih} x_1 = \begin{bmatrix} u_0 (x_2 - \beta x_1) \\ u_1 (x_2 - \alpha x_1) \end{bmatrix}$$
  For $x_2 < \beta x_1 \approx 79.17 x_1$, Unit 0 is clamped ($h_{3, 0} = 0$).
  Unit 1 has pre-activation $z_{3, 1} = u_1 (x_2 - \alpha x_1)$.
  Thus, when $x_2 > \alpha x_1$, $h_{3, 1} > 0$ while $h_{3, 0} = 0$, satisfying $h_{3, 1} > C_{out} h_{3, 0} = 0$!
  The network predicts **1** for $\alpha x_1 < x_2 < \gamma x_1$, where $\gamma \approx 2045.14$ is the upper boundary where $h_{3, 0}$ turns on and suppresses position 1 back to 0.

#### Case 3B: $x_0 > 0$
Assuming $x_1 < \beta x_0$, Unit 0 was suppressed at step 2, leaving $h_2 = [0, h_{2, 1}]^T$:
$$z_{3, 0} = u_0 x_2 + W_{01} h_{2, 1} = u_0 x_2 - 2.5048876 \, h_{2, 1}$$
$$z_{3, 1} = u_1 x_2 + W_{11} h_{2, 1} = u_1 x_2 - 1.9860576 \, h_{2, 1}$$

If $h_{2, 1} = 0$ (i.e. $x_1 \le \alpha x_0$), then $z_3 = w_{ih} x_2$, leading to $h_{3, 1} / h_{3, 0} \le u_1 / u_0 < C_{out}$, so the prediction is **0**.

If $h_{2, 1} = u_1 (x_1 - \alpha x_0) > 0$:
1. For $z_{3, 1} > 0$:
   $$x_2 > \frac{-W_{11}}{u_1} h_{2, 1} = -W_{11} (x_1 - \alpha x_0) = c_{low} (x_1 - \alpha x_0)$$
   where $c_{low} = -W_{11} = 1.9860576$.
2. For $h_{3, 1} > C_{out} h_{3, 0}$ when $h_{3, 0} > 0$:
   $$(u_1 - C_{out} u_0) x_2 > (C_{out} W_{01} - W_{11}) h_{2, 1}$$
   Since $u_1 - C_{out} u_0 = 3.152859 - (1.111296)(2.950093) \approx -0.125568 < 0$, dividing by this negative quantity reverses the inequality:
   $$x_2 < \frac{C_{out} W_{01} - W_{11}}{u_1 - C_{out} u_0} h_{2, 1} = c_{high} (x_1 - \alpha x_0)$$
   where:
   $$c_{high} = \frac{C_{out} W_{01} - W_{11}}{u_1 - C_{out} u_0} u_1 \approx \frac{(1.111296)(-2.504888) - (-1.986058)}{-0.125568} \times 3.152859 \approx 20.027073$$

Therefore, when $x_0 > 0$, the network predicts **1** if and only if:
$$x_1 > \alpha x_0 \quad \text{and} \quad c_{low} (x_1 - \alpha x_0) < x_2 < c_{high} (x_1 - \alpha x_0)$$

---

## 4. Error Analysis: Where and Why the Model Fails

Empirical evaluation on fresh Gaussian inputs reveals the following confusion matrix:

| True Median | Pred = 0 | Pred = 1 | Pred = 2 | Class Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **0** | ~99.0% | ~1.0% | 0.0% | **99.0%** |
| **1** | ~76.4% | ~23.6% | 0.0% | **23.6%** |
| **2** | ~98.0% | ~2.0% | 0.0% | **0.0%** |

### Specific Failure Modes
1. **Total Failure on Median = 2**:
   Because logit 2 is mathematically dominated throughout $\mathbb{R}_{\ge 0}^2$, the model never outputs 2. For any input where the true median is $x_2$ (which occurs with probability $1/3$), the network is guaranteed to be incorrect.
2. **Information Loss on Negative Inputs**:
   With no bias terms, ReLU clamps all negative activations to 0. When inputs are negative (e.g. $x_0 \le 0, x_1 \le 0$), the network loses their exact values and enters a dead quiescent state $h_t = 0$. On the all-negative octant $x_0, x_1, x_2 \le 0$, the model outputs 0 unconditionally.
3. **Temporal Asymmetry (Ascending vs. Descending Medians)**:
   The median is at position 1 in two distinct ordering permutations:
   - Ascending: $x_0 < x_1 < x_2$
   - Descending: $x_2 < x_1 < x_0$
   The network's recurrent architecture is tuned exclusively to detect ascending sequences ($x_0 < x_1 < x_2$). In descending sequences, the large initial value $x_0$ triggers massive self-inhibition on Unit 0 and sets a high comparison threshold for Unit 1, so the subsequent smaller $x_1$ fails to activate Unit 1. The network thus predicts 0 on virtually all descending sequences.
4. **Boundary Margin Misalignments**:
   The comparator uses $\alpha \approx 0.8702 < 1.0$ rather than $1.0$. Consequently, in Region A, when $0.8702 x_1 < x_2 < x_1$, the network predicts 1 even though $x_2 < x_1$ (meaning $x_2$ is actually the true median).

---

## 5. Derivation of Constants

| Symbol | Mathematical Formula | Value | Mechanistic Role |
| :---: | :---: | :---: | :---: |
| $C_{out}$ | $\frac{w_{out, 00} - w_{out, 10}}{w_{out, 11} - w_{out, 01}}$ | `1.111296` | Relative threshold of Unit 1 over Unit 0 in readout logits to predict 1 |
| $\alpha$ | $-\frac{W_{10} u_0 + W_{11} u_1}{u_1}$ | `0.870246` | Effective comparison ratio between successive inputs |
| $c_{low}$ | $-W_{11}$ | `1.986058` | Lower slope of acceptance band for $x_2$ when $x_0 > 0$ |
| $c_{high}$ | $\frac{C_{out} W_{01} - W_{11}}{u_1 - C_{out} u_0} u_1$ | `20.027073` | Upper slope of acceptance band for $x_2$ before Unit 0 reactivates |
| $\beta$ | $-\frac{W_{00} u_0 + W_{01} u_1}{u_0}$ | `79.168755` | Threshold ratio for $x_1$ to overcome Unit 0 recurrent self-inhibition |
| $\gamma$ | $\frac{C_{out} c_0 - c_1}{u_1 - C_{out} u_0}$ | `2045.142090` | Extreme ratio in Region A where Unit 0 reactivates |

---

## 6. Accuracy Derivation

The accuracy is the Gaussian measure of the union of regions where prediction matches the true median:
$$\text{Accuracy} = P(\text{true} = 0 \text{ and } \mathbf{x} \notin R_1) + P(\text{true} = 1 \text{ and } \mathbf{x} \in R_1)$$
$$= \frac{1}{3} + P(\text{true} = 1 \text{ and } \mathbf{x} \in R_1) - P(\text{true} = 0 \text{ and } \mathbf{x} \in R_1)$$

where $R_1 = \text{Region A} \cup \text{Region B}$.

1. **Region A ($X_0 \le 0, X_1 > 0, \alpha X_1 < X_2 < \gamma X_1$)**:
   - $P(\text{true} = 0 \text{ and } \mathbf{x} \in \text{Region A}) = 0$ (since $X_0 \le 0 < X_1, X_2$, $X_0$ is always the minimum).
   - $P(\text{true} = 1 \text{ and } \mathbf{x} \in \text{Region A}) = \frac{\arctan(\gamma) - \arctan(1)}{4\pi} = \frac{\arctan(2045.14) - \pi/4}{4\pi} \approx 0.06246109$.
2. **Region B ($X_0 > 0, X_1 > \alpha X_0, c_{low}(X_1 - \alpha X_0) < X_2 < c_{high}(X_1 - \alpha X_0)$)**:
   In projective coordinates $u = X_1/X_0, v = X_2/X_0$ with density $p(u, v) = \frac{1}{4\pi (1 + u^2 + v^2)^{3/2}}$:
   - $P(\text{true} = 1 \text{ and } \mathbf{x} \in \text{Region B}) = 0.01725930$
   - $P(\text{true} = 0 \text{ and } \mathbf{x} \in \text{Region B}) = 0.00297040$

Combining these yields:
$$\text{Accuracy} = \frac{1}{3} + (0.06246109 + 0.01725930) - (0.0 + 0.00297040) = 0.41008332 \approx 0.410083$$

---

## 7. What Could Not Be Explained

1. **Training Convergence to the Broken Minimum**:
   It is unclear why the training optimization settled into a solution that completely abandoned Position 2 rather than learning a more symmetric heuristic. This is likely an artifact of gradient descent under severe parameter starvation (only 2 ReLU neurons without biases attempting to approximate a 3-way ranking function).
2. **Training Hyperparameters**:
   The exact training trajectory (loss function formulation, optimizer choice, learning rate schedule, and random seed) that selected these specific numerical values cannot be inferred from the weights alone.
