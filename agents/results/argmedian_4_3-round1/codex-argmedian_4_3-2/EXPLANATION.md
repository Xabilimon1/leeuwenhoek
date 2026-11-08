# What this network appears to compute

## Main rule

For ordinary inputs the answer is the position of the middle value. The
network is an approximate three-item comparator: it keeps a coarse state of
which side of zero has been seen and how the recurrent state compares with
the next input. It is not a clean symbolic sorting circuit. The simplest
faithful rule I can justify is therefore `argmedian(x)`. The companion twin
implements that rule.

The network is positively homogeneous: multiplying all three inputs by any
positive number multiplies every hidden state and every logit by that number,
so it cannot use an absolute distance threshold. Its disagreements are
instead angular regions (ratios of differences and input magnitudes). In
Gaussian probes, most wrong answers occurred when two values were very close
relative to the full range, especially when the second and third inputs had
the same sign. The network can then end in a state near a readout boundary
and choose a neighboring position. A few less localized errors also occur.
This describes the dominant failure mode, not an exact classifier for every
failure.

For a reproducible coarse error-region estimate, I use the proxy wedge
`x2*x3 > 0` and `abs(x2-x3) < g * (max(x)-min(x))`. The sign condition says
the last two inputs reuse the same sign pair; the gap condition describes a
near-comparison boundary relative to the input range. I set its effective
width from the imbalance in the two large readout contrast magnitudes:
`g = (31.7396 - 30.1783) / (2 * ((31.7396 + 30.1783)/2)) = 0.0252`.
Here 31.7396 and 30.1783 are the positive and absolute negative coefficients
of the position-1 minus position-2 contrast. Taking half the fractional
imbalance is only a first-order width estimate; it is not an exact ReLU
boundary. I count this whole wedge as an error region for the rough estimate.
It includes some correct predictions and omits some other errors, so the
estimate is approximate.

## Reading the weights

The input vector is
`[3.9653, 4.6995, -4.7163, -3.9652]`. Thus units 0 and 1 receive positive
drive from a positive input, and units 2 and 3 receive positive drive from a
negative input. At the first step, for example, a positive `x1` gives
`[3.9653*x1, 4.6995*x1, 0, 0]`; a negative `x1` gives
`[0, 0, -4.7163*x1, -3.9652*x1]`. The close magnitudes within these pairs
make them two graded channels for each sign, rather than four unrelated
input detectors.

The recurrent matrix mixes both pairs. Its upper-left block is
`[[1.3387,-2.1996],[1.9277,-2.4339]]`; its lower-left block is
`[[-2.3877,2.8057],[-1.8931,2.6429]]`. The lower-left block shows that a
positive-side state can drive the negative-side units, subject to ReLU
clipping and the next input's own drive. Conversely, the upper-right block
`[[2.6352,-1.8948],[2.7868,-2.3826]]` lets a negative-side state drive the
positive-side units. The resulting state is a history-dependent, coarse
comparison code. At time 3, the readout interprets its balance; it does not
read the last input alone.

The readout rows, in hidden-unit order, are approximately

| logit | coefficients |
|---|---|
| position 0 | `[-1.6718, 1.5993, 1.6175, -1.6575]` |
| position 1 | `[-18.7477, 15.9993, 15.9592, -18.7243]` |
| position 2 | `[11.4305, -15.7403, -15.6973, 11.4769]` |

For example, the position-1 minus position-0 coefficient vector is
`[-17.0759, 14.4000, 14.3417, -17.0667]`, while position-1 minus
position-2 is `[-30.1783, 31.7396, 31.6565, -30.2011]`. These contrasts
show that the readout uses differences between the two channels in each
sign pair. Large margins usually make the decoded order stable. Near a
ReLU switching surface, however, a small change in a recurrently mixed
channel can change which contrast wins. This is why near-ties are the
dominant observed error pattern.

## Limits of the explanation

The pair-level account is supported by the input and recurrent weight
structure, and the median rule matches the network on about 98.4% of a
200,000-input Gaussian diagnostic run. I cannot assign each unit a precise
named order statistic at every time step: the recurrent matrix mixes the
channels, and ReLU makes those meanings depend on the active state. Nor can
I give a compact exact list of all the narrow failure wedges without
effectively reproducing the network's piecewise-linear computation. The
estimate below treats the dominant near-comparison wedges approximately
and explicitly leaves the residual failure shapes unresolved.
