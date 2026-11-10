# What the network computes

## Short version

The dominant behavior is a sequence-sensitive comparison of the three
values: the winning output is the position of the middle value. The hidden
state does not contain a clean “three sorted values” register that can be
assigned one unit per rank. Instead, it is a distributed, rectified code for
the prefix seen so far. Most inputs land on the correct side of the three
pairwise order boundaries. The small set of observed mistakes cluster close
to those boundaries, where the learned boundaries are slightly displaced
from exact equality. This is an empirical account of the function, not a
claim that every hidden unit has a simple semantic label.

## Hidden-state mechanics

At step one the recurrence starts from zero, so the exact unit formula is
`h_1[j] = max(0, w_ih[j] * x_1)`. The input-weight signs divide the units into
two groups:

| Units | `w_ih` values (from `weights.npz`) | Active at step one when |
|---|---|---|
| 0, 2, 3, 5, 7 | 6.5779443, 6.15469, 16.20787, 1.4493427, 2.4044752 | `x_1 > 0` |
| 1, 4, 6 | -5.5214934, -6.355015, -9.872095 | `x_1 < 0` |

Thus the first state records the sign and magnitude of the first item in
several channels. At steps two and three each channel receives a different
mixture of all eight previous channels, then ReLU removes negative totals:
`h_t[j] = max(0, w_ih[j] * x_t + sum_k w_hh[j,k] h_(t-1)[k])`. This is how
information about order in the prefix can survive and be revised as later
values arrive. The recurrent matrix is dense and includes both positive and
negative links; for example its last row has especially strong negative
links from units 6 and 7 (-24.73504 and -41.89989). I cannot give a reliable
one-feature-per-unit interpretation for steps two and three: the useful
comparison state is distributed across changing ReLU patterns.

This conclusion comes from the matrices and hidden-state traces for
permutations of several simple triples. Those traces show distinct sparse
activation patterns for different orders, while scaling a triple by a
positive constant scales every hidden state and logit by the same constant.
The latter follows directly from the bias-free ReLU recurrence and positive
homogeneity. It is consistent with an order code, rather than a fixed
threshold on the absolute value scale.

## Readout and errors

The three scores are the final state dotted with each row of `w_out`:

* position 0: `(3.6678145, 3.4525862, -6.9334702, 3.0703495, -7.2545586, 0.07860869, 5.320763, -0.6825346)`
* position 1: `(-8.257725, -5.3680453, 1.4656192, 4.7164416, 4.547644, -2.991095, 2.6575956, 1.4047078)`
* position 2: `(2.2590573, 0.39489922, 3.1954927, -12.205326, -0.8731364, 2.131403, -7.3648286, -0.21704446)`

These constants are the rows of `w_out`; the largest score wins. They are
not three independent detectors: positive and negative contributions from
many active channels combine in each score. For example, the same class can
win with very different hidden activation patterns for different input
orders.

On 600,000 distinct Gaussian exploratory inputs evaluated through
`load_model`, the observed accuracy was about 0.99948. (Some batches were
repeated for separate boundary summaries.) The errors were concentrated where
two adjacent sorted values nearly coincide; representative failures include
`(0.731, -0.274, 0.732)` and `(-1.562, 2.204, 2.203)`. Those are cases where a
small learned displacement of an order boundary changes which position
wins. I did not find evidence for a broad error region away from near-ties.
The random probes do not establish an exact global description: a few
three-way near-ties and tails may behave differently, and I cannot explain
the exact recurrent circuit or predict every displaced boundary from a
human-readable unit-level rule.

## Hand-written twin

`twin.py` uses the compact rule “return the position whose value is between
the other two.” It has no weights, model import, matrix arithmetic, or
lookup data. It deliberately leaves out the small boundary displacements,
so its rare disagreements with the network should be cases where the
network itself is wrong or nearly tied. For exact ties, NumPy's stable
sorting convention chooses one of the tied positions; exact ties have zero
probability under the stated continuous input distribution.

## Accuracy estimate from the described error region

For an estimate, approximate the leading error region by a one-sided wedge
at each pairwise tie: for one unordered pair `(X_i, X_j)`, an ordering on one
side of equality is wrong when

`0 < s * (X_i - X_j) < gamma * |X_k - (X_i + X_j)/2|`,

where `k` is the remaining coordinate, `s` chooses the side of the tie that
is displaced into an error, and `gamma = 0.00075`. This is a rounded
effective boundary width from the weight-defined near-tie errors: in the
400,000-input boundary-characterization batch, the 75th percentile of
adjacent gap divided by the distance from the pair midpoint to the third
value was about
`7.4 * 10^-4`. It is a deliberately coarse description, not a fitted
piecewise-linear reconstruction.

The difference `D = X_i - X_j` has distribution `N(0, 2)`. It is independent
of `Z = X_k - (X_i + X_j)/2`, which has distribution `N(0, 3/2)`. The
probability of the stated one-sided wedge for one pair is

`E[Phi(gamma * |Z| / sqrt(2)) - 1/2]`, with `Z ~ N(0, 3/2)`.

For this small `gamma`, expand the normal CDF at zero. Since
`E|Z| = sqrt(3/pi)`, the probability is approximately
`gamma * sqrt(3) / (2*pi)`. There are three pairs, and overlaps require two
simultaneous near-ties, so their probability is higher order in `gamma`.
This gives estimated error probability
`3 * gamma * sqrt(3) / (2*pi) = 0.000620`, hence the first line of
`estimate.txt` is `0.99938`.

This estimate includes the three one-sided pairwise wedges. It leaves out
their overlaps, the very small multiway-tie regions, and exceptional tails
or boundary shifts that the compact rule does not capture. In particular,
the estimate is derived by integrating the stated wedge under independent
Gaussian inputs; it is not the empirical accuracy from model sampling.
