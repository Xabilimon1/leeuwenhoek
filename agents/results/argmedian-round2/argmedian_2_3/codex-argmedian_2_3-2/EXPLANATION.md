# How the network behaves

## Short version

The network is heavily biased toward answer 0. It answers 1 when the second
input has left a positive trace in hidden unit 2 and the third input is large
enough to keep that trace positive. The first input reduces the trace when it
is positive. The network never predicts 2: the readout's class-2 logit cannot
beat both class 0 and class 1 for any nonnegative hidden state.

This is a mechanistic approximation, not a complete account of every boundary
case. In particular, it treats hidden unit 1 as extinguished by step 3 for
ordinary Gaussian-sized inputs. Extremely large inputs can defeat that
approximation, and mixed final hidden states can shift the class-0/class-1
boundary.

## Hidden state and constants

Write the input as `(a,b,c)` and the two hidden coordinates as `(u,v)`. The
input weights are `2.9500926` and `3.152859`, both positive. Thus at step 1,
`a <= 0` gives `(u,v)=(0,0)`, while `a > 0` starts both coordinates positive.

The self-connection of unit 1 is `-76.4917`. After a positive `u` has been
created, an ordinary next input cannot keep it alive: the next preactivation
contains `-76.4917*u`. That makes unit 1 a short-lived positive-input
detector. Its cross-connection into unit 2 is `1.1925036`; unit 2's
self-connection is `-1.9860576`.

When `a <= 0`, the first step is zero. If `b > 0`, step 2 is approximately
`(2.9500926*b, 3.152859*b)`. At step 3 unit 1 is normally shut off by its large
negative self-connection. Unit 2 then has preactivation

```
3.152859*c + (1.1925036*2.9500926 - 1.9860576*3.152859)*b
```

The coefficient of `b`, divided by `3.152859`, is `-0.87024619`. So this
branch leaves unit 2 active when `b > 0` and `c > 0.87024619*b`.
The magnitude `2.744...` below is the same cancellation,
`1.9860576*3.152859 - 1.1925036*2.9500926`; dividing it by `3.152859`
gives `0.87024619`.

When `a > 0`, unit 1 is usually suppressed at step 2. The unit-2 state there
is approximately

```
max(0, 3.152859*b - 2.744...*a)
  = 3.152859 * max(0, b - 0.87024619*a).
```

At step 3, with unit 1 suppressed, unit 2 stays active when `c` exceeds
`(1.9860576/3.152859)` times that state. The factors cancel, leaving
`c > 1.9860576 * max(0, b - 0.87024619*a)`. These two cases are the rule used
in [twin.py](twin.py).

## Readout

From `w_out`, the three logits are

```
L0 =  2.8454947*u - 2.4923484*v
L1 = -3.7529578*u + 3.4452689*v
L2 =  0.8350099*u - 0.92129105*v.
```

For `L2` to exceed `L0`, it would require
`v > (2.8454947-0.8350099)/(2.4923484-0.92129105)*u = 1.279*u`; for `L2` to
exceed `L1`, it would require
`v < (0.8350099+3.7529578)/(3.4452689+0.92129105)*u = 1.051*u`. These
cannot both hold for nonnegative `u,v`, so class 2 is impossible (including
either axis). If only unit 1 is
active the readout chooses 0; if only unit 2 is active it chooses 1. With
both active, class 1 wins only when
`v > (2.8454947+3.7529578)/(2.4923484+3.4452689)*u = 1.111*u`, from
equating `L1` and `L0`. This explains the model's strong default to class 0 and why the
threshold approximation can miss cases when unit 1 survives.

## Median errors

The target is the *position* of the middle-valued input. The dominant error
is that the model predicts 0 even when position 1 or 2 is the median. Its
rare answer 1 is based on a positive carry from `b` into `c`, not on a full
comparison of all three values. It therefore also errs when that carry
condition happens to hold but position 1 is not the median. For example,
position 1 is correct only when `a < b < c`; a threshold on `c` and a
positive carry from `b` does not enforce both inequalities. The approximation
misses unusual cases with very large values, where unit 1 can persist, and
cases where both final units matter to the readout.

## What remains unexplained

This explanation captures the typical positive carry and the readout's class
ordering, but does not give a complete classification of all real triples.
It does not model the rare inputs that keep unit 1 alive at step 3, nor
small shifts in the class boundary when both units are active. It also does
not claim that the trained rule is a correct median algorithm.
