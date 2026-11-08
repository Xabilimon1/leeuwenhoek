# How the 4-unit ReLU RNN finds the median position

## 1. Symmetry: two mirrored pairs of units
`w_ih = (3.97, 4.70, -4.72, -3.97)`. Units 0,1 respond to positive inputs, units 2,3 to negative ones.
`w_hh` satisfies (to ~1%) `w_hh[3-i, 3-j] = w_hh[i, j]` and `w_ih[3-i] = -w_ih[i]`, so the network is
equivariant under `x -> -x` with the unit swap 0<->3, 1<->2. The readout rows have
`w_out[:,0] ~ w_out[:,3]` and `w_out[:,1] ~ w_out[:,2]`, so the logits depend (almost) only on
`A = h0 + h3` and `B = h1 + h2`:

    l0 = -1.67 A + 1.60 B,  l1 = -18.7 A + 16.0 B,  l2 = 11.45 A - 15.7 B

Comparing rows gives the readout as a ratio test on r = B/A:
- l0 > l2  iff r > (11.45+1.67)/(15.7+1.60) = 0.76
- l0 > l1  iff r < (18.7-1.67)/(16.0-1.60) = 1.19
So: **answer 2 if B/A < 0.76, answer 1 if B/A > 1.19, answer 0 in between** (checked on samples: the
predicted class of every sample falls exactly in these r ranges).
Because there are no biases, everything is positively homogeneous: only the direction of x matters.
The answer is also invariant under x -> -x (as is the median), so below take x1 > 0.

## 2. What the units hold at each step (x1 = a > 0, x2 = b, x3 = c)
- **t=1**: h = a*(3.97, 4.70, 0, 0). Just stores x1.
- **t=2**: pre-activations (from w_ih*b + w_hh h1):
  `u0 = 3.97 b - 5.03 a`, `u1 = 4.70 b - 3.79 a`, `u2 = -4.72 b + 3.72 a`, `u3 = -3.97 b + 4.91 a`.
  Units 0/3 are a +/- pair comparing x2 with 1.25*x1, units 1/2 a +/- pair comparing x2 with 0.80*x1.
  Thus the state encodes the comparison x2 vs x1 in three regimes: b > 1.27a ("x2 clearly above",
  units 0,1 on), 0.81a < b < 1.24a ("x2 close to x1", units 1,3 on), b < 0.79a ("x2 below", units 2,3 on),
  with magnitudes proportional to the (tilted) distance |x2 - x1|.
- **t=3**: each unit again compares x3 with a tilted combination of x1,x2. E.g. in the "x2 clearly above"
  regime the pre-activations are proportional to `c - 1.27b + 0.41a` (unit 0), `c - 0.81b - 0.10a` (unit 1),
  `c - 0.79b - 0.29a` (unit 2, negated), `c - 1.24b + 0.13a` (unit 3, negated). So A ~ 4|x3 - T_A|,
  B ~ 4.7|x3 - T_B| with two thresholds T_A, T_B built from x1, x2 (derived by multiplying out w_hh, w_ih
  for each regime; numbers above are those products). The ratio B/A then says on which side of / how far
  from x1 and x2 the input x3 is, and the readout of section 1 turns that into "x3 is median" (B small),
  "x2 is median" (B large) or "x1 is median" (in between).

## 3. Resulting rule (twin.py) and where it fails
Overall the network outputs the median position (accuracy ~98.4%), except in thin bands near ties:
1. **x2 ~ x3, x1 not the median -> answers 0.** When the last two inputs are nearly equal both A and B are
   driven by the same near-zero difference and B/A lands in the middle band [0.76, 1.19], which is the
   class-0 band. This is the dominant error (~1% of inputs). Band: |x3-x2| < G_LATE*S, S=|x1|+|x2|+|x3|,
   G_LATE = 0.010.
2. **x1 median but x1 ~ x3 -> answers 2** (|x3-x1| < 0.0037*S), and **x1 median but x1 ~ x2 -> answers 1**
   (|x2-x1| < 0.0020*S): near these ties B/A falls just outside the class-0 band.

The band widths G_LATE, G_13, G_12 were **not derived in closed form from the weights**: I fitted them as the
best single relative-gap threshold separating the network's answers on sampled inputs (own forward passes,
logged). The scale S was chosen among several candidates as the one giving the best separation.
The twin agrees with the network on ~99.3% of Gaussian inputs.

## 4. What I could not explain
- The exact shape of the error bands: their true width is not a constant fraction of S; it depends on
  the direction of x (e.g. wider when x1 is the minimum, and the regime boundaries at x2 ~ 1.25x1 / 0.8x1
  add extra kinks). The remaining ~0.7% twin disagreement comes from this.
- A closed-form derivation of the band widths from the readout ratios 0.76/1.19 through all t=2/t=3
  regimes (time budget ran out).

## 5. Accuracy estimate
See estimate.txt: thin-band Gaussian measure, P(band of half-width g*S) ~= 1.0868*g, giving error
~0.0109 + 0.0020 + 0.0011 = 0.014, accuracy ~0.986.
