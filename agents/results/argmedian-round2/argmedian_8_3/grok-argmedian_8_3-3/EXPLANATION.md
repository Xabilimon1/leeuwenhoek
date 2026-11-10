# How the median network computes its answer

Inputs are written \(x_1, x_2, x_3\). The prediction is a position in \(\{0,1,2\}\), meaning \(x_1\), \(x_2\) or \(x_3\). Hidden units are indexed \(0..7\) as in `w_ih`, `w_hh`, `w_out`. There are no biases, so every pre-activation is linear on each activation cone and the whole map is positively homogeneous: scaling \((x_1,x_2,x_3)\) by \(\lambda>0\) scales the logits by \(\lambda\) and does not change the argmax. The true median position has the same invariance, and also depends only on order. The network is **not** invariant to a sign flip or to adding a constant; it handles signs by splitting units into two groups.

The behaviour on almost every input is:

> Return the position of the median.

The only failures are thin wedges around the three tie planes \(x_i=x_j\), where a small leakage in the readout prefers the more extreme of the two nearly equal inputs. On i.i.d. standard normals those wedges have total probability about \(4\times 10^{-4}\) (computed below).

## Step 1: a sign-split copy of \(x_1\)

\(h_0 = 0\), so

\[
h_i^{(1)} = \mathrm{relu}(w^{\mathrm{ih}}_i\, x_1).
\]

`w_ih` (float32 values as stored) is

| unit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| \(w^{\mathrm{ih}}\) | +6.578 | −5.521 | +6.155 | +16.208 | −6.355 | +1.449 | −9.872 | +2.404 |

- If \(x_1>0\), units \(\{0,2,3,5,7\}\) turn on and each holds \(w^{\mathrm{ih}}_i x_1\). Units \(\{1,4,6\}\) stay off.
- If \(x_1<0\), units \(\{1,4,6\}\) turn on and each holds \(w^{\mathrm{ih}}_i x_1 = |w^{\mathrm{ih}}_i|\,|x_1|\). The other five stay off.

So step 1 writes one number, \(x_1\), into a fixed direction that depends only on \(\mathrm{sign}(x_1)\). The five (respectively three) units are redundant copies with different gains. The gains matter because each copy has its own outgoing row of \(W = w_{hh}\).

There are exactly two activation patterns at \(t=1\), split by the sign of \(x_1\). This is forced by the formula, not fit from samples.

## Step 2: compare \(x_2\) with a few multiples of \(x_1\)

With \(x_1>0\), \(h^{(1)} = x_1\, v_+\), where \(v_+ = \mathrm{relu}(w^{\mathrm{ih}})\). The pre-activation at step 2 is

\[
\mathrm{pre}^{(2)} = w^{\mathrm{ih}} x_2 + x_1\, (W v_+).
\]

With \(x_1<0\), \(h^{(1)} = |x_1|\, v_-\), \(v_- = \mathrm{relu}(-w^{\mathrm{ih}})\), and

\[
\mathrm{pre}^{(2)} = w^{\mathrm{ih}} x_2 + |x_1|\, (W v_-).
\]

Both vectors \(W v_+\) and \(W v_-\) are constants computed from the weights. Dividing component \(i\) by \(w^{\mathrm{ih}}_i\) puts the unit in the same units as the inputs. The unit then stores a positive multiple of a ReLU of one linear form. The thresholds below are \(-(W v_+)_i / w^{\mathrm{ih}}_i\) or the analogous ratio for \(v_-\).

**When \(x_1>0\)** (multiple of \(x_1\) in front of each term):

| unit | what it stores | turns on when |
|---|---|---|
| 0 | \(\mathrm{relu}(x_2 - 2.271\, x_1)\) | \(x_2 > 2.271\, x_1\) |
| 1 | \(\mathrm{relu}(-(x_2 - 2.258\, x_1))\) | \(x_2 < 2.258\, x_1\) |
| 2 | \(\mathrm{relu}(x_2 + 2.653\, x_1)\) | \(x_2 > -2.653\, x_1\) |
| 4 | \(\mathrm{relu}(-(x_2 + 2.651\, x_1))\) | \(x_2 < -2.651\, x_1\) |
| 3 | \(\mathrm{relu}(x_2 - 0.715\, x_1)\) | \(x_2 > 0.715\, x_1\) |
| 6 | \(\mathrm{relu}(-(x_2 - 1.019\, x_1))\) | \(x_2 < 1.019\, x_1\) |
| 5 | \(\mathrm{relu}(x_2 - 6.774\, x_1)\) | \(x_2 > 6.774\, x_1\) |
| 7 | \(\mathrm{relu}(x_2 - 47.27\, x_1)\) | \(x_2 > 47.27\, x_1\) |

**When \(x_1<0\)**, writing \(a = |x_1|\):

| unit | what it stores | turns on when |
|---|---|---|
| 2 | \(\mathrm{relu}(x_2 - 2.213\, a)\) | \(x_2 > 2.213\, a\) |
| 4 | \(\mathrm{relu}(-(x_2 - 2.210\, a))\) | \(x_2 < 2.210\, a\) |
| 3 | \(\mathrm{relu}(x_2 + 0.317\, a)\) | \(x_2 > -0.317\, a\) |
| 6 | \(\mathrm{relu}(-(x_2 + 0.319\, a))\) | \(x_2 < -0.319\, a\) |
| 0 | \(\mathrm{relu}(x_2 + 3.052\, a)\) | \(x_2 > -3.052\, a\) |
| 1 | \(\mathrm{relu}(-x_2 - 5.731\, a)\) | \(x_2 < -5.731\, a\) |
| 5 | \(\mathrm{relu}(x_2 + 5.853\, a)\) | \(x_2 > -5.853\, a\) |
| 7 | \(\mathrm{relu}(x_2 - 95.61\, a)\) | \(x_2 > 95.61\, a\) |

Reading the pairs:

- **Units 2 and 4** are a differential pair. They are ReLUs of opposite linear forms with the same slope (2.65 when \(x_1>0\), 2.21 when \(x_1<0\)), so exactly one of them is on and together they encode the signed quantity \(x_2 + 2.65 x_1\) or \(x_2 + 2.21 x_1\).
- **Units 0 and 1** are the same kind of pair about the slope \(2.26\), but only while \(x_1>0\). While \(x_1<0\) the two slopes (3.05 and 5.73) do not match, and there is a gap in which both are off.
- **Units 3 and 6** are a near-comparison of \(x_2\) with \(x_1\). While \(x_1<0\) they are complements across the plane \(x_2 = -0.318 x_1\), i.e. a genuine difference. While \(x_1>0\) the two thresholds are 0.715 and 1.019, so both units are on in a band around \(x_2 \approx x_1\) and exactly one is on outside it. That overlap is how a later linear combination can see "these two inputs are close" rather than only which one is larger.
- **Units 5 and 7** fire only at extreme ratios (\(x_2\) several times \(x_1\), or ~50 times). Unit 7 is off at step 2 on more than 99.5% of standard-normal inputs.

So after two steps the hidden state is a distributed, sign-split code for the pair \((x_1, x_2)\): several kink locations along the ray of ratios \(x_2/x_1\), not a single clean \(\mathrm{relu}(x_2-x_1)\). The slopes are the trained numbers above. They are not small integers. I did not find a simpler exact circuit that these eight numbers implement; the integer comparison one would have written by hand is only recovered after the third step and the readout, as a property of the **argmax**, not of any one coordinate of \(h\).

## Step 3: fold in \(x_3\), and what the final units mean

\(\mathrm{pre}^{(3)} = w^{\mathrm{ih}} x_3 + W h^{(2)}\), then ReLU, then \(\mathrm{logits} = w_{\mathrm{out}} h^{(3)}\).

On each fixed activation pattern, \(h^{(3)}\) is an explicit linear form in \((x_1,x_2,x_3)\), obtained by zeroing the inactive units and multiplying the three weight matrices. Those forms were checked against a direct forward pass (agreement \(10^{-13}\)).

Across patterns, the final units line up with the order statistics of the prefix \((x_1,x_2)\) as follows. The match is a least-squares fit of the raw hidden unit on these ReLU features, not an identity; \(R^2\) is the fraction of variance explained.

| unit | best description of \(h_i^{(3)}\) | \(R^2\) |
|---|---|---|
| 3 | positive multiple of \(\mathrm{relu}(x_3 - \max(x_1,x_2))\) | 0.97 |
| 6 | positive multiple of \(\mathrm{relu}(\max(x_1,x_2) - x_3)\) | 0.98 |
| 5 | positive multiple of \(\mathrm{relu}(x_1 - x_2)\) | 0.93 |
| 2 | positive multiple of \(\mathrm{relu}(x_2 - x_1)\) | 0.95 |
| 1 | positive multiple of \(\mathrm{relu}(x_2 - x_1)\) | 0.90 |
| 0 | positive multiple of \(\mathrm{relu}(x_1 - x_2)\) | 0.91 |
| 4 | positive multiple of \(\mathrm{relu}(x_1 - x_2)\) | 0.89 |
| 7 | no feature of this family fits (\(R^2 < 0.05\)) | — |

Unit 3 is on when \(x_3\) is the unique maximum, with amplitude equal to how far it sticks out past \(\max(x_1,x_2)\). Unit 6 is on when \(x_3\) is below that running maximum, with amplitude the gap. Units 0, 1, 2, 4 and 5 carry the step-2 comparison of \(x_1\) with \(x_2\) forward, each with its own gain, and are only weakly mixed with \(x_3\). The several copies are what let the readout take different linear combinations in different cones.

Unit 7's row of \(W\) is dominated by \(W_{77} = -41.90\) and \(W_{76} = -24.74\). Combined with \(w^{\mathrm{ih}}_7 = +2.40\), the unit stays off unless the newest input is enormous relative to the running state. It is active at step 3 on about 3% of standard-normal inputs. Its readout (below) is negligible next to the other units.

## How the readout turns the code into a position

Normalize each unit by \(|w^{\mathrm{ih}}_i|\), so a unit that stores \(\mathrm{relu}(\pm x)\) has size \(|x|\). The readout of these normalized units, \(R = w_{\mathrm{out}}\mathrm{diag}(|w^{\mathrm{ih}}|)\), is

| unit | vote for position 0 | position 1 | position 2 |
|---|---|---|---|
| 0 | +24.1 | **−54.3** | +14.9 |
| 1 | +19.1 | −29.6 | +2.2 |
| 2 | **−42.7** | +9.0 | +19.7 |
| 3 | +49.8 | +76.4 | **−197.8** |
| 4 | **−46.1** | +28.9 | −5.5 |
| 5 | +0.1 | −4.3 | +3.1 |
| 6 | +52.5 | +26.2 | **−72.7** |
| 7 | −1.6 | +3.4 | −0.5 |

The large entries are a veto code:

- **Unit 3 vetoes position 2** (\(-198\)). It is on precisely when \(x_3 = \max(x_1,x_2,x_3)\). Position 2 is then not the median, and the veto removes it. The contest left between positions 0 and 1 is won by whichever of \(x_1, x_2\) is larger, because units 0 and 4 (which carry \(\mathrm{relu}(x_1-x_2)\)) vote against position 1, while units 1 and 2 (which carry \(\mathrm{relu}(x_2-x_1)\)) vote against position 0. The larger of \(x_1, x_2\) is the median whenever \(x_3\) is the unique maximum.
- **Unit 6 also vetoes position 2** (\(-73\)), and it grows with \(\max(x_1,x_2)-x_3\). When \(x_3\) is the unique minimum this gap is the full range, the veto fires, and the same comparison units select the **smaller** of \(x_1, x_2\), which is then the median.
- When \(x_3\) sits strictly between \(x_1\) and \(x_2\), unit 3 is off (so the \(-198\) veto is absent) and unit 6 is only the gap from \(x_3\) up to the max, not the full range. The comparison units' votes for positions 0 and 1 cancel against the positive votes those same units give to position 2, and position 2 wins. That is the remaining median case.

The same case split with the roles cycled is what the other activation cones implement. In every cone that carries non-negligible Gaussian mass, the logit difference between the two positions that could still be the median is a large positive multiple of their input difference, plus a small leakage from the third input. Three representative pieces, obtained by multiplying out \(w_{\mathrm{out}}\), \(W\) and \(w^{\mathrm{ih}}\) on the active set (the active set is the one with the smallest leakage among patterns that occur in that cone):

| cone | candidates | \(L_a - L_b\) as a form on \((x_1,x_2,x_3)\) | angle with \(x_a-x_b\) |
|---|---|---|---|
| \(x_1\) largest, decide \(x_2\) vs \(x_3\) | \(a{=}2, b{=}3\) | \((-0.0907,\; 210.227,\; -209.995)\) | \(\phi_{23} = 6.31\times 10^{-4}\) |
| \(x_2\) largest, decide \(x_1\) vs \(x_3\) | \(a{=}1, b{=}3\) | \((204.641,\; -0.0269,\; -204.457)\) | \(\phi_{13} = 4.60\times 10^{-4}\) |
| \(x_3\) largest, decide \(x_1\) vs \(x_2\) | \(a{=}1, b{=}2\) | \((422.537,\; -422.307,\; 0.0721)\) | \(\phi_{12} = 2.97\times 10^{-4}\) |

(The angle is \(\arccos\) of the absolute cosine between that coefficient vector and \(e_a-e_b\).) Up to the leakage, \(L_a-L_b\) has the sign of \(x_a-x_b\), and the third logit loses inside the cone. Therefore

\[
\operatorname{argmax} \mathrm{logits} = \text{position of the median},
\]

except where the leakage flips the sign of \(L_a-L_b\).

## Where the median is wrong, and why

The sign flip happens on the side of each tie plane toward which the leakage leans, in a wedge whose opening angle is the \(\phi\) above.

Concretely, for the first row: \(L_2 - L_3 = 0\) is the plane

\[
-0.0907\, x_1 + 210.227\, x_2 - 209.995\, x_3 = 0,
\]

while the true median switches at \(x_2 = x_3\). Solving for the gap,

\[
x_2 - x_3 \approx 4.3\times 10^{-4}\, x_1 - 1.1\times 10^{-3}\, \tfrac{x_2+x_3}{2}.
\]

The gap at which the network changes its mind is a linear function of the inputs with coefficients of size \(10^{-3}\), not zero. The same expansion produces \(\phi_{13}\) and \(\phi_{12}\).

The direction of the lean is the same on both orientations of a tie:

- If the two **largest** inputs differ by less than this gap, the network returns the position of the **maximum**. The true median is the smaller of those two.
- If the two **smallest** inputs differ by less than this gap, the network returns the position of the **minimum**. The true median is the larger of those two.

In both situations the readout has preferred the more extreme of the tied pair. Away from the tie, the \(\alpha(x_a-x_b)\) term dominates the leakage and the median wins by a wide margin (typical logit gap \(\sim 100\) on standard-normal inputs, versus \(\sim 0.2\) on the errors).

Nothing in the Gaussian bulk, far from all three planes, was found to be wrong. The failure is not a wrong case in the veto logic; it is the veto's comparison plane sitting a fraction of a degree off the true tie.

## Probability of the wedges

For a standard Gaussian the direction is uniform on the sphere, so the orthant probability of a cone \(\{n_j\cdot x > 0\}\) is the multivariate-normal positive-orthant probability with correlation matrix built from the \(n_j\). Applying that to the three planes above, inside the cone where the remaining input is strictly largest (so the median really is the larger of the other two), gives the probability that the tilt swaps those two:

| cone | wedge probability |
|---|---|
| \(x_1\) largest | \(9.44\times 10^{-5}\) |
| \(x_2\) largest | \(7.63\times 10^{-5}\) |
| \(x_3\) largest | \(3.70\times 10^{-5}\) |

Sum: \(2.08\times 10^{-4}\). These three cones partition space, and the event they measure is "the answer is the minimum, not the median".

The other failure, "the answer is the maximum, not the median", is the same geometry on the other orientation of each tie plane (the two largest inputs within angle \(\phi\), instead of the two smallest). It is disjoint from the first event. The principal pieces give the same angles for that orientation, so it contributes the same \(2.08\times 10^{-4}\).

Total error probability \(4.16\times 10^{-4}\), accuracy \(0.9996\). Intersections of two tie planes (all three inputs nearly equal) are \(O(\phi^2)\) and are not added. Secondary activation pieces, whose leakage angle is larger than the principal \(\phi\) but which occupy less of each tie plane, are left out of the sum; folding them in would move the estimate in the fourth decimal place, not the third.

## What is not explained

- The step-2 slopes (0.32, 0.72, 1.02, 2.2–2.7, 6.8, 47, 96) are not a small integer program. They are what gradient descent left in \(W v_+\) and \(W v_-\). The median computation is a property of the argmax of the readout, not of an exact identity such as "\(h_3 = \mathrm{relu}(x_3-\max(x_1,x_2))\)". The \(R^2\) column is the gap: a few percent of each unit's variance is not that feature.
- Unit 7 never got a clear job. Its self-weight \(-41.9\) keeps it off except at extreme ratios, it is active at the final step on only about 3% of standard-normal inputs, and its readout column is smaller than the other votes. I cannot derive a case where it decides the argmax. I did not run an ablation.
- Unit 5 (slope about 6.8, readout column under 5) is a weak extra copy of \(\mathrm{relu}(x_1-x_2)\). It is not needed to state the veto rule, and I did not isolate an input region whose answer flips if unit 5 is removed.
- There are on the order of 150 activation cones. The account above covers the cones that hold the bulk of the Gaussian mass and the crossings of the tie planes. A tail of rare cones uses different linear forms. In all of them that I inspected, the argmax is still the median except on a wedge of the same kind. I do not have a closed-form description of every cone.
- The network does not implement a translation-invariant comparator. Far from the origin, in a regime standard-normal inputs almost never reach, a different cone can be active. The claim "argmax = median position, up to the wedges" is for the cones that standard-normal inputs hit, which is the distribution the estimate uses.
