# finQbit — complete model specification

Companion to *"Option Pricing on Noisy Intermediate-Scale Quantum Computers:
A Quantum Neural Network Approach"*.

This document specifies the quantum model completely enough to reimplement it in
any standard circuit framework. Together with `finqbit_parameters.txt` it fixes
the model's predictions uniquely; no part of the study depends on the authors'
implementation.

---

## 1. Register and readout

- **2 qubits**, labelled `q0` and `q1`.
- Initial state `|00>`.
- Measurement on **`q0` only**.
- Raw model output is the expectation value of the Pauli-Z operator on `q0`:

  ```
  <Z_0> = P(q0 = 0) - P(q0 = 1)
  ```

- Final prediction applies a rectifier:

  ```
  C_pred = max(0, <Z_0>)
  ```

  This step is **part of the model**, not a display convention. It is responsible
  for the positive bias in the deep out-of-the-money region, for the vanishing
  training gradient there, and for the saturated hardware estimates reported in
  the paper. Omitting it does not reproduce the published results.

## 2. Inputs

Four features, **in this order**:

| index | symbol | meaning | sampling range |
|:-----:|:------:|---|---|
| 1 | `m`     | moneyness `S/K`     | `[0.80, 1.20]` |
| 2 | `t`     | time to maturity    | `[0.20, 1.10]` |
| 3 | `r`     | risk-free rate      | `[0.02, 0.10]` |
| 4 | `sigma` | volatility          | `[0.01, 1.00]` |

The target is the normalised call price `C/K` from the Black–Scholes–Merton
formula.

> **Important.** The quantum model consumes the **raw** feature values. It does
> *not* rescale them to `[-1, 1]`. That rescaling applies only to the classical
> MLP baseline and to the Fourier basis used in the redundancy test — see
> `classical_baselines.jl`.

## 3. Gate sequence

Four variational blocks `W1..W4` alternating with three encoding blocks `S1..S3`
(so `L = 3` data re-uploading stages), in exactly this order:

```
W1 : U3(q0) , U3(q1) , CX(q0->q1) , CX(q1->q0)
S1 : RX(q0) , RY(q0) , RX(q1)     , RY(q1)
W2 : U3(q0) , U3(q1) , CX(q0->q1) , CX(q1->q0)
S2 : RX(q0) , RY(q0) , RX(q1)     , RY(q1)
W3 : U3(q0) , U3(q1) , CX(q0->q1) , CX(q1->q0)
S3 : RX(q0) , RY(q0) , RX(q1)     , RY(q1)
W4 : U3(q0) , U3(q1) , CX(q0->q1) , CX(q1->q0)
measure q0
```

Totals: **8 CX gates**, **8 U3 gates**, **12 single-axis encoding rotations**,
**36 trainable parameters**.

`U3` follows the standard three-angle convention `U3(theta, phi, lambda)` as used
by Qiskit; the authors' library exports directly to `qc.u3(theta, phi, lambda, qubit)`,
and the released OpenQASM circuits were produced through that path.

## 4. Parameters

`finqbit_parameters.txt` holds 36 values, one per line, in this layout:

| lines | name | role |
|---|---|---|
| 1–4   | `s1[1..4]` | encoding scalers for block `S1` |
| 5–8   | `s2[1..4]` | encoding scalers for block `S2` |
| 9–12  | `s3[1..4]` | encoding scalers for block `S3` |
| 13–36 | `w[1..24]` | `U3` angles for `W1..W4`, six per block |

### 4.1 Variational blocks

Six angles per block, three per `U3` gate, `q0` before `q1`:

| block | angles |
|---|---|
| `W1` | `w[1:3]` on `q0`, `w[4:6]` on `q1` |
| `W2` | `w[7:9]` on `q0`, `w[10:12]` on `q1` |
| `W3` | `w[13:15]` on `q0`, `w[16:18]` on `q1` |
| `W4` | `w[19:21]` on `q0`, `w[22:24]` on `q1` |

### 4.2 Encoding blocks

Each encoding rotation angle is a **scaler times a raw feature**. The rule for
scaler indices is simple and uniform:

> `s_L[j]` always multiplies feature `j`, with features ordered `(m, t, r, sigma)`.

Which *gate* each feature reaches changes from layer to layer — this permutation
is the architecture's distinctive element, and it is what Table 12 of the paper
("Feature-to-gate assignment per layer") records:

| block | `RX(q0)` | `RY(q0)` | `RX(q1)` | `RY(q1)` |
|---|---|---|---|---|
| `S1` | `s1[1] * m` | `s1[4] * sigma` | `s1[2] * t` | `s1[3] * r`     |
| `S2` | `s2[2] * t` | `s2[1] * m`     | `s2[3] * r` | `s2[4] * sigma` |
| `S3` | `s3[1] * m` | `s3[2] * t`     | `s3[3] * r` | `s3[4] * sigma` |

Note that `S1` and `S2` swap the roles of `m` and `t`, and that `S2` and `S3`
place `r` and `sigma` identically on `q1`. Both facts follow from the table and
neither is accidental: the pairs sharing a qubit within a layer are the pairs the
`CX` block of that layer directly entangles, which is why the redundancy test in
the paper uses exactly the pairs `(m, sigma)`, `(t, r)`, `(m, t)`, `(r, sigma)`.

### 4.3 Flat parameter order

Frameworks that bind circuit parameters positionally consume them in gate
declaration order, which for this circuit is:

```
w[1:6]                                                  # W1
s1[1]*m , s1[4]*sigma , s1[2]*t , s1[3]*r               # S1
w[7:12]                                                 # W2
s2[2]*t , s2[1]*m     , s2[3]*r , s2[4]*sigma           # S2
w[13:18]                                                # W3
s3[1]*m , s3[2]*t     , s3[3]*r , s3[4]*sigma           # S3
w[19:24]                                                # W4
```

## 5. Training configuration

Reported for completeness; the released parameters are the trained result and
need not be retrained to reproduce the paper's tables.

- Loss: mean squared error on the normalised price.
- Optimizer: gradient descent with an adaptive step (the `Eva` optimizer of the
  authors' library), learning rate `alpha = 0.01`, periodic-argument handling
  enabled.
- Budget: 20 outer epochs of at most 20 inner iterations.
- Training set: `data/bs_train.csv`, 500 points.

Two caveats are stated in the paper and repeated here because they bear on any
attempt to retrain:

1. The released parameter vector was obtained with an iteratively tuned schedule
   that was continued until test performance exceeded the XGBoost baseline. That
   margin was therefore a stopping criterion, not an independent outcome, and no
   comparative claim in the revised paper rests on it.
2. Training from random initialisations fails outright in a substantial fraction
   of attempts. The rectifier admits an absorbing region in which the circuit
   output is identically zero on the whole domain; the gradient there is exactly
   zero and the run cannot recover. This happens both at initialisation and, less
   obviously, part-way through training. The paper reports the observed rate.

## 6. Files in this release

| path | contents |
|---|---|
| `SPECIFICATION.md` | this document |
| `finqbit_parameters.txt` | the 36 trained parameters |
| `classical_baselines.jl` | standalone implementation of every classical model |
| `data/bs_train.csv` | training set, 500 points |
| `data/bs_test.csv` | original test set, 100 points |
| `data/bs_eval_10000.csv` | enlarged evaluation set, 10,000 points |
| `circuits/standard/finqbit_m*.qasm` | the finQbit circuit as executed on hardware, one per benchmark point `m = 0.8 .. 1.2`. Two qubits, 8 $CX$ gates, trained angles written into the gates |
| `circuits/compressed_u4/u4_m*.qasm` | the same five points after the $U(4)$ compression of the ansatz-optimisation section, 3 $CX$ gates each. These are compiled per input point and are therefore not a pricing function: each one reproduces the circuit output at its own $m$ only |
| `hardware/raw/<backend>/task_NNN.json` | the device return for every individual execution, 300 files across the three AWS backends. Each carries the submitted OpenQASM, the device-compiled program where the backend returns one, the shot count, the moneyness label and reference price, and a time offset in seconds from the first task of that campaign so that the ordering needed for drift analysis is preserved. IQM Garnet and Rigetti Ankaa-3 return per-shot bitstrings in `measurements`; IonQ Forte returns only the final distribution, in `measurementProbabilities`, which is why the shot-convergence analysis for that backend is a Monte Carlo reconstruction rather than a resampling of recorded shots. Applying $\hat{C}=\max(0,\langle Z_0\rangle)$ to these files reproduces every value in the corresponding `_repetitions.csv` exactly. Cloud task identifiers, account and region metadata and absolute timestamps are not included |
| `hardware/*_repetitions.csv` | per-repetition raw measurements for the three AWS Braket backends, one row per repetition: moneyness, the Black-Scholes reference price, the raw expectation value, a repetition label and the readout-mitigated value. Cloud task identifiers are retained by the authors but withheld from this release, since the identifier format embeds account and region metadata; every reported hardware statistic is reproducible from the columns given here |
| `hardware/ibm_fez_per_point.csv` | IBM Fez campaigns U4, A and B: per-point mean, standard deviation, bias and SEM. The raw counts from the Qiskit sessions were not retained, so for these three campaigns the per-point mean and standard deviation are the primary record rather than a summary of one; there is no per-repetition file and no task identifiers. Written by `draw_ibm_figures.jl`, the same script that draws the two IBM figures, from the values recovered from the run outputs |

The hardware records are sufficient to reproduce every hardware table in the
paper without device access. Re-running the devices would not reproduce them in
any case, since it would not reproduce the calibration state.
