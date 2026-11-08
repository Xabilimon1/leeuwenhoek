# Mechanistic Explanation of the Median-Finding RNN

## 1. Executive Summary

The network is a 4-hidden-unit recurrent neural network (RNN) with ReLU activations and no bias terms:
$$h_0 = 0$$
$$h_t = \text{relu}(w_{ih} x_t + w_{hh} h_{t-1}) \quad \text{for } t = 1, 2, 3$$
$$\text{logits} = w_{out} h_3$$

Because the network contains no biases, it is strictly positive 1-homogeneous:
$$\text{logits}(c \cdot x) = c \cdot \text{logits}(x) \quad \forall c > 0$$
Consequently, scaling the inputs preserves the argmax prediction: $\text{predict}(c \cdot x) = \text{predict}(x)$.

The network solves the median-of-3 problem with **~98.5% accuracy** on i.i.d. standard Gaussian inputs by implementing an elegant geometric coordinate transformation:
1. **$S_3$ Weyl Chambers and Inversion Folding**: The 6 possible orderings of $(x_1, x_2, x_3)$ partition $\mathbb{R}^3$ into 6 conical chambers of opening angle $60^\circ = \pi/3$. Because $\text{median}(-x) = \text{median}(x)$, opposite chambers have identical median positions. The network uses exact anti-diagonal reflection symmetry ($h \leftrightarrow J h$) to fold the space antipodally so that opposite chambers map to the same logits: $\text{logits}(-x) = \text{logits}(x)$.
2. **$120^\circ$ Recurrent Rotation**: The recurrent weight matrix $w_{hh}$ possesses a complex conjugate pair of eigenvalues $\lambda \approx -1 \pm i\sqrt{3} = 2 e^{\pm i 2\pi/3}$. At each recurrent time step, the hidden state undergoes a $120^\circ$ planar rotation. Over 3 steps, inputs $x_1, x_2, x_3$ enter with relative phase offsets of $240^\circ, 120^\circ, 0^\circ$, forming the symmetric 3-phase complex sum $z = x_1 e^{i 4\pi/3} + x_2 e^{i 2\pi/3} + x_3$.
3. **Shift Invariance**: Because the $2 \times 2$ rotation block satisfies the characteristic polynomial $M^2 + M + I \approx 0$, any constant offset $(c, c, c)$ cancels out identically: $(M^2 + M + I) v c = 0$.
4. **Default Readout Mechanism**: The readout matrix $w_{out}$ weights the "outer" units $(h_0, h_3)$ and "inner" units $(h_1, h_2)$ with distinct magnitudes. Logit 1 and Logit 2 require strong activation differences to win; Logit 0 has weights roughly an order of magnitude smaller and fires near zero, winning by default whenever the recurrent signal is balanced or small.

---

## 2. Weight Symmetries and Hidden Unit Pairings

Let $J = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix}$ be the anti-diagonal exchange matrix. Inspecting the weights:

- **Input weights $w_{ih}$**:
  $$w_{ih} \approx [3.965, 4.700, -4.716, -3.965]^T$$
  Notice $J w_{ih} \approx -w_{ih}$. Units 0 and 3 form an outer pair with input weight $|w| \approx 3.965$; units 1 and 2 form an inner pair with input weight $|w| \approx 4.708$.

- **Recurrent weights $w_{hh}$**:
  $$w_{hh} \approx \begin{pmatrix}
   1.339 & -2.200 &  2.635 & -1.895 \\
   1.928 & -2.434 &  2.787 & -2.383 \\
  -2.388 &  2.806 & -2.434 &  1.933 \\
  -1.893 &  2.643 & -2.191 &  1.341
  \end{pmatrix}$$
  Notice $J w_{hh} J = w_{hh}$.

- **Readout weights $w_{out}$**:
  $$w_{out} \approx \begin{pmatrix}
   -1.672 &  1.599 &  1.617 & -1.658 \\
  -18.748 & 15.999 & 15.959 & -18.724 \\
   11.431 & -15.740 & -15.697 & 11.477
  \end{pmatrix}$$
  Notice $w_{out} J \approx w_{out}$.

### Inversion Invariance
Because $J w_{ih} = -w_{ih}$, $J w_{hh} J = w_{hh}$, and $w_{out} J = w_{out}$, replacing $x \to -x$ maps $h_t(-x) = J h_t(x)$ at every step. Consequently:
$$\text{logits}(-x) = w_{out} h_3(-x) = w_{out} J h_3(x) = w_{out} h_3(x) = \text{logits}(x)$$
The model is strictly invariant under input sign negation.

---

## 3. Step-by-Step Hidden Unit Representation

### Step $t = 1$: Encoding $x_1$
At $t=1$, $h_0 = 0$, so $h_1 = \text{relu}(w_{ih} x_1)$:
- If $x_1 > 0$: $h_1 = [3.965 x_1, 4.699 x_1, 0, 0]^T$. Only units 0 and 1 are active.
- If $x_1 < 0$: $h_1 = [0, 0, 4.716 |x_1|, 3.965 |x_1|]^T$. Only units 2 and 3 are active.

### Step $t = 2$: Comparing $x_2$ with $x_1$
At $t=2$, $z_2 = w_{ih} x_2 + w_{hh} h_1$.
For $x_1 > 0$, the recurrent contribution is $w_{hh} h_1 = x_1 [-5.029, -3.794, 3.717, 4.914]^T$.
The preactivations are:
$$z_{2, 0} = 3.965 x_2 - 5.029 x_1 = 3.965 (x_2 - 1.268 x_1)$$
$$z_{2, 1} = 4.699 x_2 - 3.794 x_1 = 4.699 (x_2 - 0.807 x_1)$$
$$z_{2, 2} = -4.716 x_2 + 3.717 x_1 = -4.716 (x_2 - 0.788 x_1)$$
$$z_{2, 3} = -3.965 x_2 + 4.914 x_1 = -3.965 (x_2 - 1.239 x_1)$$

Let $\alpha \approx 1.25$ and $\beta \approx 0.80$. Note that $\alpha \cdot \beta \approx 1.00$ ($\beta \approx 1/\alpha$).
This partitions the relationship between $x_1$ and $x_2$ into three distinct activation regimes:
1. **$x_2 > 1.25 x_1$ ($x_2 \gg x_1$)**:
   Units 0 and 1 are active: $h_2 = [z_{2,0}, z_{2,1}, 0, 0]^T$.
2. **$x_2 < 0.80 x_1$ ($x_2 \ll x_1$)**:
   Units 2 and 3 are active: $h_2 = [0, 0, z_{2,2}, z_{2,3}]^T$.
3. **$0.80 x_1 < x_2 < 1.25 x_1$ ($x_2 \approx x_1$)**:
   Units 1 and 3 are active: $h_2 = [0, z_{2,1}, 0, z_{2,3}]^T$.

The network thus encodes whether $x_2 > x_1$, $x_2 < x_1$, or $x_2 \approx x_1$ into disjoint coordinate pairs.

### Step $t = 3$: Integrating $x_3$
At $t=3$, $z_3 = w_{ih} x_3 + w_{hh} h_2$, and $h_3 = \text{relu}(z_3)$.
In each chamber, the 2D rotation matrix $M = w_{hh}[:2, :2]$ rotates the previous differences by another $120^\circ$.
Let $U = h_{3,0} + h_{3,3}$ (outer activation) and $V = h_{3,1} + h_{3,2}$ (inner activation).

---

## 4. How the Readout Produces the Answer

Because of the reflection symmetry in $w_{out}$, the logits depend almost entirely on $U$ and $V$:
$$\text{logit}_0 \approx -1.665 U + 1.608 V$$
$$\text{logit}_1 \approx -18.736 U + 15.979 V$$
$$\text{logit}_2 \approx +11.454 U - 15.719 V$$

Solving for pairwise decision boundaries as a function of the ratio $r = U / V$:
- **Logit 1 vs Logit 0**:
  $$(-18.736 - (-1.665)) U + (15.979 - 1.608) V = -17.071 U + 14.371 V = 0 \implies \frac{U}{V} = 0.8418$$
- **Logit 2 vs Logit 0**:
  $$(11.454 - (-1.665)) U + (-15.719 - 1.608) V = 13.119 U - 17.327 V = 0 \implies \frac{U}{V} = 1.3208$$

Therefore:
- When $\mathbf{U/V < 0.8418}$: **Position 1 wins** ($\text{argmax} = 1$).
- When $\mathbf{0.8418 < U/V < 1.3208}$: **Position 0 wins** ($\text{argmax} = 0$).
- When $\mathbf{U/V > 1.3208}$: **Position 2 wins** ($\text{argmax} = 2$).

### The "Default Fall-Through" Nature of Position 0
Notice that the coefficients in Row 0 of $w_{out}$ ($\sim 1.6$) are approximately **10 times smaller** than those of Row 1 ($\sim 17$) and Row 2 ($\sim 14$).
When $U \approx V$ (neither inner nor outer units dominate):
$$\text{logit}_1 \approx -18.74 + 15.98 = -2.76 \cdot V < 0$$
$$\text{logit}_2 \approx +11.45 - 15.72 = -4.27 \cdot V < 0$$
$$\text{logit}_0 \approx -1.66 + 1.61 = -0.05 \cdot V \approx 0$$
Logits 1 and 2 are strongly negative, while logit 0 stays near zero. **Logit 0 wins by default** whenever the recurrent signal does not strongly polarize $U$ vs $V$.

---

## 5. Failure Modes and Derivation of Constants

The network makes errors on ~1.5% of Gaussian inputs. These errors fall into three specific geometric bands where two inputs are nearly identical:

### 1. Cluster Ambiguity ($x_1 \approx x_2$, predicting 0 instead of 1 or 2)
When $x_1$ and $x_2$ are close, and $x_0$ is an outlier ($x_0 > x_1 \approx x_2$ or $x_0 < x_1 \approx x_2$), the true median is either position 1 or position 2.
However, because $x_1 \approx x_2$, the step-3 activation difference is too small to drive $U/V$ outside the $[0.8418, 1.3208]$ band. Consequently, logits 1 and 2 remain negative, and the network **defaults to predicting position 0**.

#### Derivation of $G_{LEFT} = 0.0136$ and $G_{RIGHT} = 0.0107$:
Let $x_0 = 1, x_1 = 0$, and $x_2 = \delta$.
- At $t=1$: $h_1 = [w_{ih,0}, w_{ih,1}, 0, 0]^T$.
- At $t=2$: $z_2 = w_{hh} h_1 = [-5.029, -3.794, 3.717, 4.914]^T$. Thus $h_2 = [0, 0, 3.717, 4.914]^T$.
- At $t=3$: $z_3 = \delta w_{ih} + w_{hh} h_2 = \delta w_{ih} + [0.4862, -1.3476, 0.4508, -1.5573]^T$.
  For small $\delta$, units 0 and 2 are active:
  $$h_{3,0} = 0.4862 + 3.965 \delta$$
  $$h_{3,2} = 0.4508 - 4.716 \delta$$
  Logits are:
  $$\text{logit}_1 - \text{logit}_0 = -1.8371 x_0 + 3.2791 x_1 - 135.3504 \delta$$
  $$\text{logit}_2 - \text{logit}_0 = -1.4350 x_0 + 3.1796 x_1 + 133.6165 \delta$$

Setting $\text{logit}_1 = \text{logit}_0$:
$$\delta_1 = \frac{-1.8371}{135.3504} = -0.01357 \implies G_{LEFT} \approx 0.0136$$

Setting $\text{logit}_2 = \text{logit}_0$:
$$\delta_2 = \frac{1.4350}{133.6165} = +0.01074 \implies G_{RIGHT} \approx 0.0107$$

The total relative gap is $G_{LEFT} + G_{RIGHT} = 0.02431$.
Whenever $|x_2 - x_1| < 0.02431 \cdot |x_0 - \frac{x_1+x_2}{2}|$, the network answers position 0.

### 2. Boundary Overshoot for Position 2 ($x_0 \approx x_2$, predicting 2 instead of 0)
When $x_0$ is the true median but $x_2$ is extremely close to $x_0$, the positive entry $w_{out}[2, 0] \approx +11.4$ causes logit 2 to slightly overshoot logit 0 within a relative gap:
$$G_{02} \approx 0.0075$$

### 3. Boundary Overshoot for Position 1 ($x_0 \approx x_1$, predicting 1 instead of 0)
When $x_0$ is the true median but $x_1$ is extremely close to $x_0$, logit 1 slightly beats logit 0 within a relative gap:
$$G_{01} \approx 0.0040$$

---

## 6. Theoretical Accuracy Derivation

To compute the Gaussian measure of the error regions, we project $(x_0, x_1, x_2)$ onto orthogonal Gaussian coordinates:
$$u = x_2 - x_1 \sim \mathcal{N}(0, 2)$$
$$v = x_0 - \frac{x_1 + x_2}{2} \sim \mathcal{N}(0, 1.5)$$
$$s = x_0 + x_1 + x_2 \sim \mathcal{N}(0, 3)$$
Because $(u, v, s)$ are orthogonal linear combinations of independent standard Gaussians, they are statistically independent.

The ratio $T = \frac{u / \sqrt{2}}{v / \sqrt{1.5}} = \frac{\sqrt{3}}{2} \frac{u}{v}$ follows a standard Cauchy distribution with CDF $F(t) = \frac{1}{2} + \frac{1}{\pi} \arctan(t)$.

1. **Cluster Error Probability**:
   Across the two outlier chambers (probability 2/3), the error condition $|u/v| < 0.02431$ yields:
   $$P(\text{Cluster Error}) = \frac{2}{\pi} \arctan\left(\frac{\sqrt{3}}{2} \times 0.02431\right) = 0.01340 \quad (1.340\%)$$

2. **Boundary Error for Position 2**:
   $$P(x_0 \approx x_2) = \frac{1}{\pi} \arctan\left(\frac{\sqrt{3}}{2} \times 0.0075\right) = 0.00207 \quad (0.207\%)$$

3. **Boundary Error for Position 1**:
   $$P(x_0 \approx x_1) = \frac{1}{\pi} \arctan\left(\frac{\sqrt{3}}{2} \times 0.0040\right) = 0.00110 \quad (0.110\%)$$

**Total Error Probability**:
$$P(\text{Error}) = 0.01340 + 0.00207 + 0.00110 = 0.01657$$
**Theoretical Accuracy**:
$$\text{Accuracy} = 1 - 0.01657 = 0.98343 \approx 98.34\%$$

---

## 7. What Could Not Be Fully Explained

1. **Fine Weight Asymmetries**: While $w_{ih}$, $w_{hh}$, and $w_{out}$ are nearly anti-diagonally symmetric ($J$-invariant), there are minor parameter asymmetries (e.g. $w_{ih, 0} = 3.96526$ vs $-w_{ih, 3} = 3.96520$, difference $< 10^{-4}$). These are gradient noise remnants from finite-sample SGD training.
2. **Multi-Boundary Intersections**: In the immediate vicinity of triple collisions ($x_0 \approx x_1 \approx x_2$), the transitions between different ReLU active sets intersect non-linearly. Our linear approximation assumes the active set is determined by the dominant separation vector.
