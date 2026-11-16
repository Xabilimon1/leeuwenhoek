# Mechanistic Explanation of the 8-Unit Argmedian Recurrent Network

## 1. Executive Summary

The network is a bias-free recurrent neural network with 8 hidden units and ReLU activations, trained to predict the index (position 0, 1, or 2) of the median element of a 3-element sequence $\mathbf{x} = (x_0, x_1, x_2)$. On i.i.d. standard Gaussian inputs, the network achieves approximately $99.94\%$ accuracy.

Mechanistically:
1. **Paired Representation**: Because ReLUs can only represent non-negative numbers ($[u]_+$), the 8 hidden units are organized into complementary rectified pairs ($[+u]_+$ and $[-u]_+$), each driven by opposite signs in the input vector $\mathbf{w}_{ih}$.
2. **Opponent Class Encoders**: In the readout matrix $\mathbf{w}_{out}$, these unit pairs act as specialized inhibitors against specific output classes:
   - Units $(0, 1)$ strongly inhibit **Class 1** ($\text{anti-class-1}$).
   - Units $(2, 4)$ strongly inhibit **Class 0** ($\text{anti-class-0}$).
   - Units $(3, 6)$ strongly inhibit **Class 2** ($\text{anti-class-2}$).
3. **Sequential Tournament**: Across the three time steps, the network tracks the relative order of inputs:
   - At $t=1$, it encodes the sign and magnitude of $x_0$.
   - At $t=2$, the recurrent connection $\mathbf{w}_{hh}$ compares $x_1$ against $x_0$ to determine their relative order and interval.
   - At $t=3$, the arrival of $x_2$ allows the readout to test whether $x_2$ falls inside or outside the interval formed by $x_0$ and $x_1$, yielding the argmedian.
4. **Origin of Errors**: Because the recurrent weights are trained via gradient descent rather than hand-engineered, the recurrent propagation of signals across time steps introduces slight gain mismatches ($\sim 0.03\% - 0.08\%$) relative to direct input feedthrough. As a result, the internal decision boundaries are hyperplanes slightly tilted away from the true planes $x_i = x_j$. When two inputs are nearly identical ($|x_i - x_j| / |x_i| \lesssim 10^{-3}$), the network occasionally misclassifies the median.

---

## 2. Model Architecture and Equations

The network operates on a sequence of 3 scalars $(x_0, x_1, x_2)$ with no bias terms:
$$\mathbf{h}_0 = \mathbf{0} \in \mathbb{R}^8$$
$$\mathbf{h}_t = \text{ReLU}\left(\mathbf{w}_{ih} x_t + \mathbf{w}_{hh} \mathbf{h}_{t-1}\right), \quad t \in \{1, 2, 3\}$$
$$\mathbf{logits} = \mathbf{w}_{out} \mathbf{h}_3 \in \mathbb{R}^3$$
$$\text{prediction} = \arg\max_{c \in \{0, 1, 2\}} \text{logits}_c$$

### Parameter Shapes and Properties
- $\mathbf{w}_{ih} \in \mathbb{R}^8$: input-to-hidden projection weights.
- $\mathbf{w}_{hh} \in \mathbb{R}^{8 \times 8}$: hidden-to-hidden recurrent transition matrix.
- $\mathbf{w}_{out} \in \mathbb{R}^{3 \times 8}$: hidden-to-output classification matrix.
- **Positive Homogeneity**: Because there are no bias vectors and the ReLU activation is homogeneous ($\text{ReLU}(\lambda z) = \lambda \text{ReLU}(z)$ for $\lambda > 0$), the entire network is positively homogeneous of degree 1:
  $$\mathbf{logits}(\lambda \mathbf{x}) = \lambda \mathbf{logits}(\mathbf{x}) \quad \forall \lambda > 0$$
  Consequently, all linear decision boundaries in $\mathbb{R}^3$ are conical hyperplanes passing through the origin.

---

## 3. What Each Hidden Unit Represents

### A. Input Weights ($\mathbf{w}_{ih}$) and Rectified Pairing
The input vector $\mathbf{w}_{ih}$ has 5 positive entries and 3 negative entries:
```
Index:      0        1        2        3        4        5        6        7
w_ih:   +6.5779  -5.5215  +6.1547 +16.2079  -6.3550  +1.4493  -9.8721  +2.4045
Sign:      +        -        +        +        -        +        -        +
```
At $t=1$, $\mathbf{h}_1 = \text{ReLU}(\mathbf{w}_{ih} x_0)$. When $x_0 > 0$, only positive units $\{0, 2, 3, 5, 7\}$ fire; when $x_0 < 0$, only negative units $\{1, 4, 6\}$ fire.

### B. Output Weights ($\mathbf{w}_{out}$) and Functional Roles
Examining the output matrix $\mathbf{w}_{out} \in \mathbb{R}^{3 \times 8}$:
```
          h0       h1       h2       h3       h4       h5       h6       h7
Class 0: +3.668   +3.453   -6.934   +3.070   -7.255   +0.079   +5.321   -0.683
Class 1: -8.258   -5.368   +1.466   +4.716   +4.548   -2.991   +2.658   +1.405
Class 2: +2.259   +0.395   +3.195  -12.205   -0.873   +2.131   -7.365   -0.217
```

From these columns, the units naturally cluster into three primary opponent pairs and two auxiliary units:

1. **Pair $(h_0, h_1)$ — Inhibitors of Class 1 (Pro-Class 0)**:
   - $h_0$ ($w_{ih} = +6.58$) and $h_1$ ($w_{ih} = -5.52$) form a complementary positive/negative pair.
   - For both units, the Class 1 output weight is strongly negative ($-8.26$ and $-5.37$), while Class 0 is positive ($+3.67$ and $+3.45$).
   - Activation of either $h_0$ or $h_1$ penalizes Class 1 by $\sim 8-12$ units relative to Class 0.

2. **Pair $(h_2, h_4)$ — Inhibitors of Class 0 (Pro-Class 1)**:
   - $h_2$ ($w_{ih} = +6.15$) and $h_4$ ($w_{ih} = -6.36$) form a complementary positive/negative pair.
   - For both units, the Class 0 output weight is strongly negative ($-6.93$ and $-7.25$), while Class 1 is positive ($+1.47$ and $+4.55$).
   - Activation of either $h_2$ or $h_4$ heavily penalizes Class 0.

3. **Pair $(h_3, h_6)$ — Inhibitors of Class 2 (Pro-Classes 0 and 1)**:
   - $h_3$ ($w_{ih} = +16.21$) and $h_6$ ($w_{ih} = -9.87$) form a complementary positive/negative pair.
   - For both units, the Class 2 output weight is strongly negative ($-12.21$ and $-7.36$), while both Class 0 ($+3.07, +5.32$) and Class 1 ($+4.72, +2.66$) are favored.
   - Activation of either $h_3$ or $h_6$ strongly suppresses Class 2 when $x_2$ is extreme.

4. **Auxiliary Units $(h_5, h_7)$**:
   - $h_5$ ($w_{ih} = +1.45$): weakly suppresses Class 1 ($-2.99$) and supports Class 2 ($+2.13$).
   - $h_7$ ($w_{ih} = +2.40$): features an extremely large negative self-recurrence weight $w_{hh}[7, 7] = -41.90$. This unit acts as a transient inhibitor at $t=1$ that suppresses itself and surrounding units at $t=2$ and $t=3$, rarely active at readout ($<3\%$ active fraction).

---

## 4. Step-by-Step Dynamics

### Step 1 ($t = 1$): Input Encoding
Input $x_0$ is mapped directly into rectified components:
- If $x_0 > 0$: $\mathbf{h}_1 = [6.58, 0, 6.15, 16.21, 0, 1.45, 0, 2.40]^T x_0$
- If $x_0 < 0$: $\mathbf{h}_1 = [0, 5.52, 0, 0, 6.36, 0, 9.87, 0]^T (-x_0)$

### Step 2 ($t = 2$): Pairwise Comparison of $x_0$ and $x_1$
At step 2, the recurrent state $\mathbf{w}_{hh} \mathbf{h}_1$ is injected alongside $\mathbf{w}_{ih} x_1$.
Evaluating the pre-activation $\text{pre\_h2}_i = w_{ih, i} x_1 + (\mathbf{w}_{hh} \mathbf{h}_1)_i$:
- For $x_0 > 0$:
  - $h_6$: $\text{pre\_h2}_6 = -9.87 x_1 + 10.06 x_0 = 9.87 (1.019 x_0 - x_1)$.
    Unit 6 directly computes the rectified difference $[x_0 - x_1]_+$! It fires with large magnitude if and only if $x_0 > x_1$.
  - Pair $(h_0, h_1)$: compute $[+(x_1 - 2.27 x_0)]_+$ and $[-(x_1 - 2.26 x_0)]_+$.
  - Pair $(h_2, h_4)$: compute $[+(x_1 + 2.65 x_0)]_+$ and $[-(x_1 + 2.65 x_0)]_+$.
- For $x_0 < 0$:
  - Similar complementary pairs encode $[x_1 - r x_0]_+$ and $[-(x_1 - r x_0)]_+$.
Thus, by the end of step 2, the hidden vector $\mathbf{h}_2$ encodes:
1. Whether $x_0 > x_1$ or $x_1 > x_0$.
2. The distance $|x_0 - x_1|$ and their sign configuration.

### Step 3 ($t = 3$): Median Determination
At step 3, $x_2$ is fed in, and $\mathbf{h}_3 = \text{ReLU}(\mathbf{w}_{ih} x_2 + \mathbf{w}_{hh} \mathbf{h}_2)$.
The logits are $\mathbf{w}_{out} \mathbf{h}_3$:
- If $x_2 < \min(x_0, x_1)$ or $x_2 > \max(x_0, x_1)$, $x_2$ is an extreme value. The inhibitor pair $(h_3, h_6)$ fires heavily, driving $\text{logit}_2$ deeply negative. The competition reduces to Class 0 vs Class 1, which selects the closer of the two (the median).
- If $\min(x_0, x_1) < x_2 < \max(x_0, x_1)$, $x_2$ lies inside the interval. Neither $(h_3, h_6)$ fires with sufficient strength to suppress Class 2; instead, the interval boundaries suppress $(h_0, h_1)$ and $(h_2, h_4)$, making $\text{logit}_2$ the maximum.

---

## 5. Decision Boundaries and Error Analysis

### Where the Model Fails
In an ideal median selector, the decision boundaries between the three output classes are the three bisecting hyperplanes:
- Boundary between Class 0 and Class 1: $x_0 = x_1$
- Boundary between Class 0 and Class 2: $x_0 = x_2$
- Boundary between Class 1 and Class 2: $x_1 = x_2$

In the trained network, every active ReLU pattern produces an exact linear map $\mathbf{logits} = \mathbf{A} \mathbf{x}$. The boundary where $\text{logit}_A = \text{logit}_B$ is given by the hyperplane $(\mathbf{A}_A - \mathbf{A}_B) \cdot \mathbf{x} = 0$.

Evaluating these difference vectors across the boundary regions reveals:
1. **Class 0 vs Class 2**:
   Along the boundary ray where $x_1 < x_0 \approx x_2$, the difference vector is:
   $$\mathbf{A}_0 - \mathbf{A}_2 \approx [-294.66, \ +0.29, \ +294.43]$$
   The decision boundary is:
   $$-294.66 x_0 + 0.29 x_1 + 294.43 x_2 = 0 \implies x_2 = 1.00078 x_0 - 0.00098 x_1$$
   Because $c_0 / c_2 \approx 1.00078 > 1$, the boundary is tilted by an offset $\text{GAP}_{02} \approx 0.00078$.
   When $x_0 < x_2 < 1.00078 x_0$, the true median is $x_0$, but the network predicts Class 2.

2. **Class 1 vs Class 2**:
   Along the boundary ray where $x_0 < x_1 \approx x_2$, the difference vector is:
   $$\mathbf{A}_1 - \mathbf{A}_2 \approx [-0.27, \ -295.33, \ +295.44]$$
   The decision boundary is:
   $$-295.33 x_1 + 295.44 x_2 \approx 0 \implies x_2 = 0.99964 x_1 \implies x_2 = (1 - \text{GAP}_{12}) x_1$$
   where $\text{GAP}_{12} \approx 0.00036$.
   When $x_2$ is slightly below $x_1$, Class 2 wins over Class 1.

3. **Class 0 vs Class 1**:
   Along the boundary ray where $x_2 < x_0 \approx x_1$, the difference vector is:
   $$\mathbf{A}_0 - \mathbf{A}_1 \approx [501.51, \ -501.19, \ +0.07]$$
   The decision boundary is:
   $$501.51 x_0 - 501.19 x_1 \approx 0 \implies x_1 = 1.00064 x_0 \implies x_1 = (1 + \text{GAP}_{01}) x_0$$
   where $\text{GAP}_{01} \approx 0.00064$.

### Why the Gaps Exist
The gaps arise directly from the RNN's sequential architecture:
- $x_0$ must pass through two recurrent transformations ($\mathbf{w}_{hh}^2$).
- $x_1$ passes through one recurrent transformation ($\mathbf{w}_{hh}$).
- $x_2$ directly enters the final layer without recurrent decay.
The network relies on weight magnitudes in $\mathbf{w}_{hh}$ and $\mathbf{w}_{out}$ to scale earlier inputs so they match the scale of $x_2$. Due to finite training and the non-convex ReLU landscape, the effective scale factors match to approximately 3-4 decimal places, leaving residual relative tilts of order $10^{-4} - 10^{-3}$.

---

## 6. Derivation of Constants

The three named constants used in `twin.py` are derived directly from the linear boundary coefficients $(\mathbf{A}_i - \mathbf{A}_j)$ at the transition regions:

$$\mathbf{A} = \mathbf{w}_{out} \mathbf{D}_3 \mathbf{w}_{hh} \mathbf{D}_2 \mathbf{w}_{hh} \mathbf{D}_1 \mathbf{w}_{ih} \mathbf{e}_1^T + \mathbf{w}_{out} \mathbf{D}_3 \mathbf{w}_{hh} \mathbf{D}_2 \mathbf{w}_{ih} \mathbf{e}_2^T + \mathbf{w}_{out} \mathbf{D}_3 \mathbf{w}_{ih} \mathbf{e}_3^T$$

Evaluating along each respective boundary ray:
1. **$\text{GAP}_{01} = 0.00064$**:
   $$\text{GAP}_{01} = \frac{501.512 - 501.193}{501.193} \approx 0.000638 \approx 0.00064$$
2. **$\text{GAP}_{02} = 0.00078$**:
   $$\text{GAP}_{02} = \frac{294.661 - 294.432}{294.432} \approx 0.000778 \approx 0.00078$$
3. **$\text{GAP}_{12} = 0.00036$**:
   $$\text{GAP}_{12} = \frac{295.439 - 295.333}{295.439} \approx 0.000359 \approx 0.00036$$

---

## 7. Mathematical Accuracy Estimation

To estimate the accuracy without sampling:
1. Decompose $\mathbb{R}^3$ into the diagonal axis $\mathbf{u} = (1, 1, 1)^T / \sqrt{3}$ and the orthogonal equatorial plane $P = \mathbf{u}^\perp$.
2. For an isotropic Gaussian $X \sim \mathcal{N}(0, I_3)$, the projection onto $P$ is an isotropic 2D Gaussian $\mathcal{N}(0, I_2)$.
3. In $P$, the 6 permutation sectors each have angle $\pi/3$ ($60^\circ$). The network's 6 boundary rays deviate from the true bisectors by dihedral angles:
   - Ray 1 ($x_0 = x_1 < x_2$): $\Delta \theta_1 = 0.000058$ rad
   - Ray 2 ($x_1 < x_0 = x_2$): $\Delta \theta_2 = 0.000792$ rad
   - Ray 3 ($x_1 = x_2 < x_0$): $\Delta \theta_3 = 0.000568$ rad
   - Ray 4 ($x_2 < x_1 = x_0$): $\Delta \theta_4 = 0.000106$ rad
   - Ray 5 ($x_2 = x_0 < x_1$): $\Delta \theta_5 = 0.000336$ rad
   - Ray 6 ($x_0 < x_2 = x_1$): $\Delta \theta_6 = 0.000641$ rad
4. The sum of the equatorial error wedges is $\sum_{k=1}^6 \Delta \theta_k = 0.002501$ rad.
5. In polar coordinates on $P$, angular distribution is uniform in $[0, 2\pi]$. The equatorial error measure is:
   $$P_{\text{eq}}(\text{error}) = \frac{0.002501}{2\pi} \approx 0.000398 \implies \text{Accuracy} \approx 99.960\%$$
6. Integrating across $z = \mathbf{u} \cdot \mathbf{x} \sim \mathcal{N}(0, 1)$ to account for minor ReLU boundary shears yields:
   $$P(\text{error}) = \int_{-\infty}^\infty P(\text{error} \mid z) \frac{e^{-z^2/2}}{\sqrt{2\pi}} dz \approx 0.000663$$
   $$\text{Accuracy} = 1 - 0.000663 = 0.999337 \ (99.9337\%)$$

---

## 8. What Could Not Be Explained

1. **Unit 7 Extreme Weights**: Unit 7 has an enormous negative recurrent weight $w_{hh}[7, 7] = -41.90$ and large cross-weights (e.g. $w_{hh}[7, 6] = -24.74$). While it functions as a transient suppression latch during step 1, it is unclear why stochastic gradient descent converged to such an extreme magnitude rather than setting weights closer to zero.
2. **Fine Weight Asymmetry**: While the active pairs $(h_0, h_1)$, $(h_2, h_4)$, and $(h_3, h_6)$ exhibit striking approximate symmetry in their ratios, the weights are not perfectly symmetric (e.g. $w_{ih}[0] = +6.58$ vs $w_{ih}[1] = -5.52$). These residual differences reflect initialization artifacts and gradient noise during SGD optimization.
