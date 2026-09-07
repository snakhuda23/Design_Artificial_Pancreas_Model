# Percutaneous Intraperitoneal Insulin Pump — Simulation

Closed-loop glucose-insulin simulation for a percutaneous intraperitoneal (IP) insulin
pump, developed as part of an engineering design project. 

## Files

| File | Purpose |
|---|---|
| `glucose.py` | Glucose subsystem (Dalla Man model), per-patient parameterisation |
| `insulin_new.py` | Intraperitoneal insulin absorption subsystem (Schiavon model) |
| `meal_disturbances.py` | 3-compartment meal/carbohydrate absorption model |
| `controller_pid.py` | PID controller with anti-windup and basal feedforward |
| `patient.py` | Patient parameter sets (real, simglucose v0.2.11-derived) |
| `cgm_sensor_noise.py` | CGM sensor noise model (8.2% MARD) |
| `insulin_pump_delivery.py` | Meal bolus calculation and pump quantization (0.025 U) |
| `run_tests.py` | Runs all five test scenarios |
| `plot_graphs.py` | Generates report figures (Tests 1–5) |
| `error_calcs.py` | MSE/RMSE tracking error calculations |
| `main.py` | Entry point |

## Running

```bash
python run_tests.py      # run all scenarios, print results
python plot_graphs.py    # generate figures
python error_calcs.py    # compute tracking error metrics
```

## Test Scenarios

1. 12-hour fasting baseline
2. 24-hour 3-meal day
3. 3-meal day with exercise disturbance
4. Uncontrolled vs. closed-loop comparison
5. Patient variability (three independently-parameterised patients)

## Key References

- Schiavon, M. et al. (2021). "Modeling Intraperitoneal Insulin Absorption in Patients with Type 1 Diabetes." *Metabolites*, 11(9), 600.
- Huyett, L.M. et al. (2015). "Design and Evaluation of a Robust PID Controller for a Fully Implantable Artificial Pancreas." *Ind. Eng. Chem. Res.*, 54(42).
- Xie, J. (2018). Simglucose v0.2.1 (patient parameter dataset).
- Molano-Jiménez, A. (2017). "UVa/Padova T1DMS Dynamic Model Revision."
- Dalla Man, C et al. (2014). "The UVA/PADOVA Type 1 Diabetes Simulator." 17(3):751–756. doi: 10.1177/19322968221076559


## Status

Developed as part of an academic design project. Simulation only — not validated
against real patient data or intended for clinical use.
