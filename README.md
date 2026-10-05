# finQbit — model, parameters and results released with the paper

Companion to *"Option Pricing on Noisy Intermediate-Scale Quantum Computers:
A Quantum Neural Network Approach"*. The paper is where the work is argued;
this file says what is in the repository and how to reproduce the numbers.

The repository holds the trained parameters and the specification needed to use
them, the results obtained with them, and a standalone implementation of the
classical baselines added in revision: the parameter-matched MLP, ridge
regression on the circuit's own Fourier basis, the corrected XGBoost fit, and
the cross-validation used to select the ridge penalty.

---

## 1. Register and readout

This section specifies the model as presented in the paper.

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

**Erratum.** A second parameter set, released with this repository, leaves the
circuit untouched and changes only the readout: both qubits are measured, and
three expectation values are taken from the same measurement record,
`<Z_0>`, `<Z_1>` and the correlator `<Z_0 Z_1>`, combined by a linear head
fitted after measurement.

```
C_pred = max(0, b + w_0*<Z_0> + w_1*<Z_1> + w_01*<Z_0 Z_1>)
```

No extra gates and no extra shots. Section 7 covers it.

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
declaration order, not in the order of the file above. For this circuit that
order is the sequence of Section 3 read with the two tables of 4.1 and 4.2:
`W1`, `S1`, `W2`, `S2`, `W3`, `S3`, `W4`.

## 5. Training configuration

Reported for completeness; the released parameters are the trained result and
need not be retrained to reproduce the paper's tables.

- Loss: mean squared error on the normalised price.
- Optimizer: gradient descent with an adaptive step (the `Eva` optimizer of the
  authors' library), learning rate `alpha = 0.01`, periodic-argument handling
  enabled.
- Budget: 20 outer epochs of at most 20 inner iterations.
- Training set: `data/bs_train.csv`, 500 points.

## 6. Files in this release

| path | contents |
|---|---|
| `SPECIFICATION.md` | this document |
| `finqbit_parameters.txt` | the 36 trained parameters |
| `finqbit_multi_parameters.txt` | the 40 trained parameters of the multi-observable readout of Section 7: 12 encoding scalers, 24 U3 angles, 4 linear-head coefficients. Same ansatz and the same 8 $CX$ gates as the released model; both qubits are measured |
| `classical_baselines.jl` | standalone implementation of every classical model |
| `data/bs_train.csv` | training set, 500 points |
| `data/bs_test.csv` | original test set, 100 points |
| `data/bs_eval_10000.csv` | enlarged evaluation set, 10,000 points |
| `data/bs_monitor_200.csv` | monitoring set used only as the stopping criterion for the Section 7 campaign, generated separately and verified to share no point with the training, test or evaluation sets |
| `circuits/standard/finqbit_m*.qasm` | the finQbit circuit as executed on hardware, one per benchmark point `m = 0.8 .. 1.2`. Two qubits, 8 $CX$ gates, trained angles written into the gates |
| `circuits/compressed_u4/u4_m*.qasm` | the same five points after the $U(4)$ compression of the ansatz-optimisation section, 3 $CX$ gates each. These are compiled per input point and are therefore not a pricing function: each one reproduces the circuit output at its own $m$ only |
| `hardware/raw/<backend>/task_NNN.json` | the device return for every individual execution, 300 files across the three AWS backends. Each carries the submitted OpenQASM, the device-compiled program where the backend returns one, the shot count, the moneyness label and reference price, and a time offset in seconds from the first task of that campaign so that the ordering needed for drift analysis is preserved. IQM Garnet and Rigetti Ankaa-3 return per-shot bitstrings in `measurements`; IonQ Forte returns only the final distribution, in `measurementProbabilities`, which is why the shot-convergence analysis for that backend is a Monte Carlo reconstruction rather than a resampling of recorded shots. Applying $\hat{C}=\max(0,\langle Z_0\rangle)$ to these files reproduces every value in the corresponding `_repetitions.csv` exactly. Cloud task identifiers, account and region metadata and absolute timestamps are not included |
| `hardware/*_repetitions.csv` | per-repetition raw measurements for the three AWS Braket backends, one row per repetition: moneyness, the Black-Scholes reference price, the raw expectation value, a repetition label and the readout-mitigated value. Cloud task identifiers are retained by the authors but withheld from this release, since the identifier format embeds account and region metadata; every reported hardware statistic is reproducible from the columns given here |
| `hardware/ibm_fez_per_point.csv` | IBM Fez campaigns U4, A and B: per-point mean, standard deviation, bias and SEM. The raw counts from the Qiskit sessions were not retained, so for these three campaigns the per-point mean and standard deviation are the primary record rather than a summary of one; there is no per-repetition file and no task identifiers. Written by `draw_ibm_figures.jl`, the same script that draws the two IBM figures, from the values recovered from the run outputs |

The hardware records are sufficient to reproduce every hardware table in the
paper without device access. Re-running the devices would not reproduce them in
any case, since it would not reproduce the calibration state.

## 7. Erratum: extended readout and a second parameter set

Added after submission. **Every number in the paper stands.**
`finqbit_parameters.txt` reproduces them to five decimals (R² = 0.987210,
MAE = 0.009651, RMSE = 0.012901 on the 10,000-point evaluation set).

This section adds a **second trained parameter set** and the **full statistical
battery** requested in review. The released model of Sections 1-6 is unchanged.

### 7.1 What the second set is

`finqbit_multi_parameters.txt` — the same ansatz as Sections 3 and 4, with one
difference: **both qubits are measured**, and three expectation values feed a
linear readout head.

| | released model | multi-observable readout |
|---|---|---|
| measurement | `q0` only | **`q0` and `q1`** |
| readout | ⟨Z₀⟩ | ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₀Z₁⟩ |
| parameters | 36 | 40 (36 + 4 head) |
| **CX gates** | **8** | **8** |
| depth | — | unchanged |

```
z    = b + w_Z0·⟨Z₀⟩ + w_Z1·⟨Z₁⟩ + w_Z0Z1·⟨Z₀Z₁⟩
C/K  = max(0, z)
```

This readout is not new work: reading three observables through a linear head
was first established on the Heston basket models and carried over from there.
What this section adds is the measurement of what it is worth on Black-Scholes,
where a full seed campaign and a classical comparison are cheap enough to run.

The rectifier is retained: it is what lets the model emit an **exact zero**
deep out of the money, which no smooth output map can do.

Expectation values are recovered from the joint distribution over both qubits,
with `b0` the first character of the bitstring:

```
⟨Z₀⟩   = (p00 + p01) - (p10 + p11)
⟨Z₁⟩   = (p00 + p10) - (p01 + p11)
⟨Z₀Z₁⟩ =  p00 - p01 - p10 + p11
```

**The hardware cost is unchanged.** Same gates, same depth, same shot count —
measuring both qubits instead of one costs nothing, because the circuit has to
be executed either way. The fidelity budget `(1-ε)^N_CX` is identical.

Note that the AWS records already in `hardware/raw/` contain the **full
two-qubit distribution** (`measuredQubits: [0, 1]`), so ⟨Z₁⟩ and ⟨Z₀Z₁⟩ can be
computed from the released files for the published parameter vector without any
new device access. The IBM circuits in `circuits/standard/` declare a
single classical bit, so the same is not possible for those.

### 7.2 Protocol

| set | size | role |
|---|---|---|
| `data/bs_train.csv` | 500 | gradient only |
| `data/bs_monitor_200.csv` | 200 | **stopping criterion only** |
| `data/bs_test.csv` | 100 | touched once, after training |
| `data/bs_eval_10000.csv` | 10,000 | touched once, after training |

The monitoring set is generated separately and verified to share **no point**
with the test or evaluation sets. Training stops on an R² plateau measured on
that set; the test set has no influence whatsoever on when training stops.

Campaign: 20 independent initialisations, 16 converged.

### 7.3 Results

Evaluation set of 10,000 points, generated independently of and later than
training.

| | R² | MAE [IV pts] | p95 [pts] | PW 1 pt | PW 2 pts | arbitrage | RMSLE |
|---|---|---|---|---|---|---|---|
| released, 36 par. | 0.98721 | 2.57 | 6.92 | 28.2% | 50.0% | 5.42% | 0.0108 |
| multi, best seed | **0.99527** | 1.53 | 4.28 | 45.6% | 72.3% | **3.90%** | 0.0068 |
| multi, 16-run ensemble | 0.99485 | **1.51** | 4.45 | **49.5%** | **73.9%** | 5.01% | 0.0072 |

**Distribution over seeds, not a single run.** Of 20 initialisations, 16 reached
the plateau criterion:

```
R² = 0.99223 ± 0.00201      (min 0.98590, max 0.99536)
one-sample t against the released vector:  t = +10.02
```

The remaining four exhausted the 40-epoch budget; one of them reached
R² = 0.99536, the highest of the campaign. Dead runs: zero.

**By moneyness regime.**

| | OTM | ATM | ITM |
|---|---|---|---|
| released | 0.98466 | 0.98378 | 0.97308 |
| multi | 0.99259 | 0.99194 | **0.99275** |

**The five hardware benchmark points** (T = 1, r = 0.05, σ = 0.2):

| m | Black-Scholes | multi | error [pts] | released, error [pts] |
|---|---|---|---|---|
| 0.80 | 0.018594 | 0.009424 | 2.44 | 4.96 |
| 0.90 | 0.050912 | 0.055000 | 1.09 | 1.61 |
| 1.00 | 0.104506 | 0.115854 | 3.02 | 5.53 |
| 1.10 | 0.176630 | 0.188453 | 3.15 | 5.50 |
| 1.20 | 0.261690 | 0.267899 | 1.65 | 2.61 |

Mean **2.27 pts** against **4.04 pts**.

**Against the classical baselines.** Table 9 of the paper, on the same
evaluation set, with the multi-observable row added:

| model | par. | MSE | RMSE | MAE | R² |
|---|---|---|---|---|---|
| OLS | 5 | 0.00050 | 0.02237 | 0.01626 | 0.96154 |
| Fourier ridge | 41 | 0.00036 | 0.01905 | — | 0.97210 |
| XGBoost | — | 0.00020 ± 0.00002 | 0.01414 ± 0.00087 | 0.01104 ± 0.00058 | 0.98459 ± 0.00190 |
| released finQbit | 36 | 0.00017 | 0.01290 | 0.00965 | 0.98721 |
| multi, best seed | 40 | 0.00006 | 0.00784 | 0.00574 | 0.99527 |
| MLP | 37 | 0.00005 ± 0.00003 | 0.00701 ± 0.00173 | 0.00488 ± 0.00125 | 0.99604 ± 0.00203 |
