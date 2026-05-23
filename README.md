# Neural MPC for Iron Ore Flotation Plant
## Project Report — Advanced Machine Learning, Unit 7: Control Engineering

---

## 1. Project Overview

This project implements a **Neural Model Predictive Controller (Neural MPC)** applied to an iron ore flotation plant. The core idea is to replace the physical equations that a classical MPC would require with a neural network trained on historical plant data:

> *"Use NN to learn the unknown physics ('unmodelled residuals'). E.g.: uses formulas for general drone flight, but NN for modelling ground-effects."*

The flotation plant is a complex multi-input multi-output (MIMO) industrial process involving 7 flotation columns. Its dynamics — bubble formation, surface chemistry, inter-column coupling — are too complex to derive analytically. Instead, we learn the input-output behaviour from 4,097 hours of real operational data and use the trained model inside an MPC loop.

The project is split into two Jupyter notebooks:

| Notebook | Contents |
|----------|----------|
| **Notebook 1** | Data preprocessing, feature engineering, LSTM training and evaluation |
| **Notebook 2** | MPC cost function, receding horizon simulation, constraint verification |

---

## 2. Dataset

**Source:** [Quality Prediction in a Mining Process (Kaggle)](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process?resource=download) </br>
**Format:** CSV with comma-as-decimal (European notation), 737,453 raw rows  
**After cleaning:** 4,097 unique hourly observations (March–September 2017)

The cleaning steps are:
1. Parse the `date` column and sort chronologically
2. Remove duplicate timestamps, keeping one observation per hour
3. Drop three excluded columns: `date` (used for sorting only), `Starch Flow` and `Amina Flow` (data inconsistencies)

---

## 3. Variable Classification

Following the control engineering taxonomy presented in the course, the 21 remaining columns are classified into three groups:

| Type | Symbol | Columns | Count | Role |
|------|--------|---------|-------|------|
| **Manipulated Variables** | **u** | Flotation Column 01–07 Air Flow + Level | 14 | Actions computed by the MPC controller |
| **Disturbance Variables** | **d** | Ore Pulp Flow, % Iron Feed, % Silica Feed, Ore Pulp pH, Ore Pulp Density | 5 | Observable but uncontrollable feed properties |
| **Controlled Variables** | **y** | % Iron Concentrate, % Silica Concentrate | 2 | Target outputs to be optimised |

The MPC optimisation objective is:
- **Maximise** `% Iron Concentrate` (target: ≥ 66.5%)
- **Minimise** `% Silica Concentrate` (target: ≤ 1.5%)

---

## 4. Feature Engineering

Two simple features are added for every base column (21 columns × 3 = 63 total):

| Feature type | Formula | Purpose |
|---|---|---|
| Original value | $x_t$ | Current state of each variable |
| Lag-1 difference | $x_t - x_{t-1}$ | How fast is each variable changing? |
| 6-step rolling mean | $\bar{x}_{t-6:t}$ | Short-term trend |

This gives the LSTM additional context about the direction and momentum of each variable, which is important for capturing the time-delayed dynamics of the flotation process.

---

## 5. Data Splitting and Scaling

The dataset is split **chronologically** to simulate real deployment:

| Split | Fraction | Samples | Period |
|-------|----------|---------|--------|
| Training | 70% | 2,867 | Past |
| Validation | 15% | 614 | Recent past |
| Test | 15% | 616 | Future (never seen during training) |

A `MinMaxScaler` is fitted **exclusively on the training set** and then applied to validation and test. This prevents data leakage — the model cannot implicitly learn the statistical range of future data.

---

## 6. LSTM Surrogate Model

### Architecture

```
Input: (batch, 24, 63)         — 24 hours of history, 63 features each
         ↓
   LSTM Layer 1  (hidden = 128)
         ↓
   LSTM Layer 2  (hidden = 128, dropout = 0.20)
         ↓
   Last hidden state  (128,)
         ↓
   Dropout (20%)  →  Linear (128→64)  →  ReLU  →  Linear (64→2)
         ↓
Output: (batch, 2)             — predicted % Iron, % Silica
```

**Why LSTM?** Standard networks treat each input independently. The flotation process has transport delays of up to several hours — an actuator change now affects the output 1–6 hours later. The LSTM's gating mechanism retains information across the 24-step window, making it suitable for this kind of temporal dependency.

**Why look-back = 24?** One full day of history covers the dominant process lags while keeping the input sequence manageable.

### Training

| Parameter | Value |
|---|---|
| Loss function | MSE |
| Optimiser | Adam (lr = 0.001, weight decay = 1e-4) |
| Scheduler | ReduceLROnPlateau (factor = 0.5, patience = 5) |
| Early stopping | Patience = 12 epochs |
| Max epochs | 100 |
| Batch size | 64 |
| Trainable parameters | ~270,000 |

### Results

| Model | R² Iron | RMSE Iron | R² Silica | RMSE Silica |
|-------|---------|-----------|-----------|-------------|
| Persistence baseline (y_{t+1} = y_t) | 0.4636 | 0.8327 | 0.6116 | 0.7538 |
| **LSTM (ours)** | **0.4924** | **0.8100** | **0.6374** | **0.7283** |

The LSTM outperforms the persistence baseline on both metrics. The modest absolute R² values reflect an intrinsic property of the dataset: the flotation process outputs have high short-term autocorrelation (AC(1) ≈ 0.75), meaning that much of the "signal" in y_{t+1} is simply the value y_t. A next-step prediction model on this data is fundamentally limited by this noise ceiling, which is why both the baseline and the LSTM achieve similar R² values.

It is worth noting that the persistence baseline is intentionally weak — a stronger comparator (e.g. ARIMA or a lag-only MLP) would more rigorously isolate the contribution of the control inputs. As a consequence, the improvement over baseline does not by itself prove that the model has learned the true actuator-to-output dynamics; the LSTM may partially be relying on y_t as a shortcut rather than capturing the full causal structure of the process. This is acceptable for the MPC use case, since what matters is multi-step rollout accuracy under changing u's, not single-step prediction — but it is an epistemic limitation to keep in mind when interpreting the results.

---

## 7. MPC Simulation (Notebook 2)

### What is MPC?

Model Predictive Control solves an optimisation problem at every timestep over a short finite window into the future, then applies only the first action (Receding Horizon Policy):

$$\min_{\mathbf{u}_{t}, \ldots, \mathbf{u}_{t+N_c-1}} J \quad \text{subject to} \quad \mathbf{u}_{\min} \leq \mathbf{u} \leq \mathbf{u}_{\max}$$

### Configuration

| Parameter | Value | Meaning |
|---|---|---|
| Prediction horizon $N_p$ | 8 steps | Look 8 hours ahead |
| Control horizon $N_c$ | 4 steps | Optimise 4 moves, hold the rest |
| Decision variables | $N_c \times 14 = 56$ | One vector per actuator per step |
| Solver | L-BFGS-B (scipy) | Gradient-based, supports box constraints |
| Setpoint Fe | 66.5% | Target iron purity |
| Setpoint Si | 1.5% | Maximum silica allowed |

### Cost Function

$$J = \underbrace{\sum_{k=1}^{N_p} \left[ 3.0 \cdot (\hat{y}^{Fe}_{t+k} - 66.5)^2 + 2.0 \cdot (\hat{y}^{Si}_{t+k} - 1.5)^2 \right]}_{\text{Tracking error}} + \underbrace{0.05 \cdot \sum_{k=0}^{N_c-1} \|\Delta \mathbf{u}_k\|^2}_{\text{Control effort}}$$

The tracking term penalises deviations from the setpoints. The effort term penalises large sudden changes in actuator values, protecting equipment and ensuring smooth operation.

The weights were chosen by manual inspection of the problem structure: the iron weight (3.0) is set higher than silica (2.0) because the iron setpoint has a larger absolute scale and its deviation is the primary economic objective; the effort weight (0.05) was kept small enough to allow meaningful actuator movement while preventing oscillation. A systematic tuning procedure (e.g. grid search over weight combinations) would be a natural extension but was not performed here.

### Receding Horizon Loop

At each of the 100 simulation steps:
1. **Forecast disturbances** — DVs are assumed constant over the horizon (persistence forecast)
2. **Optimise** — find the best 56-dimensional control vector using L-BFGS-B
3. **Apply** — use only `u*[0]`, the first of the 4 optimised steps
4. **Advance** — insert `u*[0]` into the state window and call the LSTM to get the next output
5. **Repeat** with updated measurements — this is the Receding Horizon Policy

### Simulation Results

| Metric | Value |
|---|---|
| Mean Fe over 100 steps | 66.03% (setpoint: 66.5%) |
| Mean Si over 100 steps | 1.31% (setpoint: 1.5%) |
| Constraint violations | **0 out of 1,400 checks** |
| Constraint satisfaction rate | **100%** |
| Average solve time per step | 0.22 s |

The MPC successfully steers the silica concentration below the target (1.31% < 1.5%) and keeps iron near the target. The 0 violations across 1,400 checks refers specifically to the **actuator box constraints** (u_min ≤ u ≤ u_max) — these are guaranteed by construction by the L-BFGS-B solver, which handles bound constraints exactly. The output setpoints (Fe ≥ 66.5%, Si ≤ 1.5%) are soft objectives encoded in the cost function, not hard constraints, so they are not counted here. The silica output target is met on average (1.31% < 1.5%), but the iron target is not fully reached (66.03% vs 66.5%), as discussed in the convergence note below. This distinction matters: the key advantage of MPC stated in Unit 7 — that it is *"the only method that fully considers action constraints"* — applies precisely to the actuator bounds, which are enforced perfectly throughout the simulation.

### Note on Convergence

The simulation converges quickly (by step ~20) to a stable operating point at Fe = 66.03%, Si = 1.31%, and stays there for the remaining 80 steps. This is expected behaviour: the optimiser finds a local minimum of the cost function and the system stabilises. It does not reach the iron setpoint of 66.5% exactly, indicating the cost function weights or the model's uncertainty around that region prevent full convergence. This is a known limitation of neural surrogates in MPC, discussed below.

---

## 8. Limitations

**1. Weak predictive baseline**
The persistence baseline (y_{t+1} = y_t) is a natural first comparator given the high autocorrelation, but it is too weak to demonstrate that the LSTM has learned actuator-to-output dynamics. A lag-only MLP or ARIMA model would be a more rigorous comparator. Related to this, the LSTM may partially be relying on y_t as a shortcut rather than on the control inputs, which is an open question not resolved by the current experiments.

**2. MPC convergence to a local minimum**
The simulation stabilises around step 20 and does not move further toward the iron setpoint of 66.5%. The L-BFGS-B solver finds a local optimum and warm-starts from it at every subsequent step. This would be partially addressed by multi-start initialisation or a global optimisation strategy.

**3. Disturbance forecast by persistence**
The feed properties (DVs) are assumed constant over the MPC horizon. In reality, ore feed varies. A dedicated DV forecast model would improve closed-loop performance.

**4. Rolling mean computed before split**
The 6-step rolling mean is applied to the full dataset before the train/val/test split. This means the first 5 rows of the validation set inherit values from the last 5 rows of training. The practical effect is negligible (5 out of 614 validation samples), but in a strict setting it should be corrected by applying the rolling mean only within each split.

---

## 9. Conclusion

This project demonstrates a complete Neural MPC pipeline for an industrial MIMO process, covering data preprocessing, surrogate model training, and closed-loop simulation with hard actuator constraints.

The main takeaway is that the approach is viable but comes with important caveats. The LSTM surrogate is not a high-fidelity model in the classical sense — its single-step R² is modest and partially explained by output autocorrelation rather than learned actuator dynamics. Its value lies specifically in multi-step rollout inside the MPC loop, where a persistence baseline would be useless. The MPC itself enforces actuator constraints perfectly, but converges to a local optimum that does not fully reach the iron setpoint, a known limitation of gradient-based solvers on non-linear neural surrogates.

If repeated, the most impactful improvements would be: (1) a stronger predictive baseline to better isolate what the LSTM actually learns, (2) a global or multi-start optimisation strategy to escape local minima in the MPC, and (3) a dedicated disturbance forecast model to replace the persistence assumption on the feed variables.

The project sits at the intersection of classical control engineering (MPC framework, receding horizon, actuator constraints) and machine learning (data-driven surrogate replacing intractable physics), which is the Learning-based MPC paradigm described in Unit 7 as the current state of the art for complex industrial control problems.
