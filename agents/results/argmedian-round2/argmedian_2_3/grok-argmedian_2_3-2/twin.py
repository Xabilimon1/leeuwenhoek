"""Hand-written twin of the median-position network.

The rule is the one derived in EXPLANATION.md. It answers 1 only when the
middle input leaves a comparison residual and the last input falls in a band
around that residual; otherwise it answers 0. It never answers 2.

Constants (all derived from the weights in EXPLANATION.md):

  R     v stays on only if the new input exceeds R times the anchor
  K     u stays on only if the new input exceeds K times the anchor
  T     a fresh anchor reads as 0 again once the new input exceeds T times it
  LAM   lower edge of the answer-1 band, in units of the v-only margin
  RHO   upper edge of that band
  C_LO  rare branch (x2 > K x1): x3 > R x2 + C_LO x1
  C_HI  rare branch: x3 < T x2 + C_HI x1
"""

import numpy as np

R = 0.8702462635829092
K = 79.16875329070548
T = 2045.145711973249
LAM = 1.9860576391220093
RHO = 20.02710475698141
C_LO = 86.60903626931152
C_HI = -160343.54391220692


def predict(x: np.ndarray) -> np.ndarray:
    """x: (n, 3) float -> (n,) ints in {0, 1} (never 2)."""
    x = np.asarray(x, dtype=np.float64)
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    out = np.zeros(x.shape[0], dtype=np.int64)

    # x1 left the state empty and x2 became the anchor. Answer 1 when x3
    # clears a fraction R of that anchor, short of the huge multiple T.
    fresh = (x1 <= 0.0) & (x2 > 0.0) & (x3 > R * x2) & (x3 < T * x2)

    # x1 is a positive anchor and x2 beats it by margin m = x2 - R x1,
    # without reviving the anchor marker. Answer 1 when x3 lies in
    # (LAM, RHO) times that margin.
    margin = x2 - R * x1
    band = (
        (x1 > 0.0)
        & (x2 > R * x1)
        & (x2 <= K * x1)
        & (x3 > LAM * margin)
        & (x3 < RHO * margin)
    )

    # Rare: x2 > K x1, so the anchor marker did not die. Answer 1 on the
    # shifted band derived in EXPLANATION.md.
    rare = (
        (x1 > 0.0)
        & (x2 > K * x1)
        & (x3 > R * x2 + C_LO * x1)
        & (x3 < T * x2 + C_HI * x1)
    )

    out[fresh | band | rare] = 1
    return out
