# Task: explain a tiny neural network

This directory contains a small trained neural network. Your job is to find
out **how** it computes its answer, write that down, and prove your
explanation by rebuilding the network's behaviour by hand.

## The model

- Input: a sequence of {{SEQ_LEN}} real numbers `x = (x_1, ..., x_{{SEQ_LEN}})`.
- Output: {{SEQ_LEN}} logits; the prediction is the argmax, a position `0..{{SEQ_LEN_MINUS_1}}`.
- It was trained to output the **position of the median** of the inputs.
  It is not perfect: on some inputs it answers a different position.
- Architecture: a ReLU recurrent network with {{HIDDEN}} hidden units and no
  biases ({{N_PARAMS}} parameters in total):

  ```
  h_0 = 0
  h_t = relu(w_ih * x_t + w_hh @ h_{t-1})    for t = 1..{{SEQ_LEN}}
  logits = w_out @ h_{{SEQ_LEN}}
  ```

  Shapes: `w_ih` ({{HIDDEN}},), `w_hh` ({{HIDDEN}}, {{HIDDEN}}), `w_out` ({{SEQ_LEN}}, {{HIDDEN}}).
- Input distribution of interest: each `x_t` i.i.d. standard Gaussian.

## Files

- `weights.npz`: the weights (`w_ih`, `w_hh`, `w_out`).
- `load_model.py`: `load_weights()`, `run_model(x)` (logits) and `predict(x)`
  (positions), numpy only. Every call is appended to `model_calls.jsonl`.
- `TASK.md`: this file.

Do not modify `weights.npz` or `load_model.py`.

## Deliverables (write them in this directory)

1. **`EXPLANATION.md`**: a mechanistic explanation. What does each hidden
   unit (or group of units) represent at each step? How does the readout turn
   that into an answer? On which inputs does the network get the median
   wrong, and why? Derive every constant you use from the weights and say how.
   State clearly what you could not explain.
2. **`twin.py`**: a hand-written re-implementation built from your
   explanation, exposing

   ```python
   def predict(x: np.ndarray) -> np.ndarray:  # x: (n, {{SEQ_LEN}}) float -> (n,) ints in 0..{{SEQ_LEN_MINUS_1}}
   ```

   The twin must **not** read `weights.npz` (or any other file), import
   `load_model`, or contain the weight matrices copied in. It must also
   **not be a compiled copy of the network**: no per-region linear maps,
   matrix products of the weights, or lookup tables that reproduce the
   network's arithmetic. Write the rule your explanation describes (for
   example "compare these two inputs, but answer X when they are within a
   relative gap g"), with a small number of named constants, each derived in
   `EXPLANATION.md`. A twin that matches the network perfectly by copying its
   computation scores as a rule break, not as a success; an honest twin that
   matches less is preferred. It must run with numpy alone from a directory
   that contains only your `.py` files.
3. **`estimate.txt`**: your estimate of the model's accuracy (fraction of
   inputs where its prediction is the true median position) on i.i.d.
   standard Gaussian inputs, as a number on the first line (e.g. `0.9`),
   followed by a short description of the method. The estimate must come
   **from your explanation**: compute the probability of the error regions
   your explanation describes (e.g. the Gaussian measure of the bands where
   the network answers wrong), in closed form or by integrating over those
   regions. It must **not** come from running the model, the twin or any
   imitation of the network on inputs, and **not** from integrating or
   enumerating the network's own piecewise-linear function region by region
   (that computes the accuracy without explaining it). Say which error
   regions you included and which you left out.

## How you will be scored

- Twin: fraction of fresh Gaussian inputs on which `twin.predict` gives the
  same position as the model (not the true median: the model's own answer,
  mistakes included).
- Estimate: absolute error against the model's true accuracy.
- Explanation: read by a human for correctness and clarity.
- Rule breaks are recorded and reported.

## Rules

- Work only inside this directory. Do not read files outside it.
- Do not use the internet to look for explanations of this model, of
  networks trained on this task, or of the model zoo it comes from.
  {{INTERNET_RULE}}
- Python with numpy (and the standard library) is available. {{PACKAGES_RULE}}
- **Model access**: {{MODEL_ACCESS}}
- **Log every model evaluation.** `run_model`/`predict` log themselves. If
  you evaluate the network any other way (your own forward pass with the
  weights), append a line `{"n_inputs": <count>, "note": "..."}` to
  `model_calls.jsonl` for each evaluation.
- **Budget**: {{BUDGET}}
- Do not declare success without checking. If you run out of budget, write
  down what you have.
