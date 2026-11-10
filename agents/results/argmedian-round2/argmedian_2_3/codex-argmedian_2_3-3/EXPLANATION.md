# How the network behaves

## Hidden state and readout

Write the hidden units as (A_t,B_t). The input coefficients are both
positive, (a=2.9500926) and (b=3.152859), and the recurrence is

\[
A_t=\max(0, ax_t-76.4917A_{t-1}-2.5048876B_{t-1}),\\
B_t=\max(0, bx_t+1.1925036A_{t-1}-1.9860576B_{t-1}).
\]

Thus neither unit is simply “the median.” Both turn off for a negative input
when the previous state is zero. Unit A is especially reset-prone: any
appreciable previous A nearly shuts it off at the next step, because of the
large (-76.49A_{t-1}) term. Unit B also suppresses its own previous value,
but less sharply, and gets a positive contribution from the previous A.
Together the units behave like a short-lived, sign-gated record of recent
positive inputs, with A acting as a strong reset and B carrying a less
strongly reset trace. That temporal gating, rather than a stable rank
representation, is the main source of the errors.

For example, at step 1 a positive (x_1) gives
\((A_1,B_1)=(2.9501x_1,3.1529x_1)\), while a negative (x_1) gives zero.
If (x_1>0), the large negative A feedback makes (A_2=0) for ordinary
inputs; then
\[
B_2=\max(0,3.1529x_2-2.744\,x_1),
\]
where (2.744=1.9860576b-1.1925036a), rounded. If (x_1\le0), the state
at step 1 is zero, so a positive (x_2) instead produces
\((A_2,B_2)=(ax_2,bx_2)\). These examples show why order alone is not enough:
the state at step 3 depends on which earlier values were positive and on
their magnitudes.

The output rows are linear functions of ((A_3,B_3)). Subtracting the
position-1 logit from position 0 gives
\[
L_0-L_1=6.5984525A_3-5.9376173B_3.
\]
So the 0-versus-1 boundary is (B_3/A_3=6.5984525/5.9376178\approx
1.1113) when (A_3>0); if (A_3=0<B_3), position 1 wins. Position 2 is
never the argmax for nonnegative hidden states: it would need
\(B_3/A_3>2.0104848/1.5710574\approx1.280\) to beat position 0,
but \(B_3/A_3<4.5879677/4.3665600\approx1.051\) to beat position 1.
Those conditions cannot both hold. The all-zero state ties all logits and
`argmax` chooses position 0.

## A useful rule and its limits

A compact approximation is: answer position 1 when the inputs are strictly
increasing and the middle input is positive; otherwise answer position 0.
This captures the network's strongest broad pattern. On an increasing
sequence with positive (x_1), (A_2) is reset and (B_2) is positive
under the usual scale of (x_2>x_1>0), making the final readout favor 1
over much of that region. If the first two inputs are nonpositive, the
state is zero until a positive (x_3), which gives the initial ratio
\(B_3/A_3=b/a\approx1.0686<1.1113\) and hence predicts 0. In between,
thresholds involving the ratios and gaps of the values create exceptions.

The approximation is not the exact decision boundary. It omits cases where
the final ratio crosses the readout threshold inside an order cone, and
occasional position-1 predictions outside the increasing cone. It also
cannot explain all magnitude-sensitive gating at step 2 and step 3. In
particular, the recurrent weights do not reduce to an exact median
comparator: the network predicts only positions 0 and 1, so every input
whose median is at position 2 is necessarily wrong. The small rule in
`twin.py` encodes only the broad increasing-positive pattern, not the
network's piecewise-linear arithmetic.
