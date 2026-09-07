# --- Meal bolus (feedforward carb-ratio dosing) -----------------------

ICR = 10.0              # g carbs per U insulin (placeholder; clinically
                         # individualised, typical range ~8-15 g/U)
BOLUS_DURATION_MIN = 5.0  # deliver each meal bolus over this many minutes


def meal_bolus_rate(t: float, meal_events: list) -> float:
    """
    Feedforward meal-bolus infusion rate (mU/min) at time t.

    Each meal in meal_events is (start_time_min, carbs_g, gut_duration_min).
    At meal_start, a bolus is sized from the standard carb-ratio formula
        bolus (U) = carbs (g) / ICR
    and delivered as a constant-rate burst over BOLUS_DURATION_MIN,
    independent of the PID -- the PID continues to handle basal and any
    correction on top of this.
    """
    rate = 0.0
    for meal_start, carbs_g, _gut_duration in meal_events:
        if meal_start <= t < (meal_start + BOLUS_DURATION_MIN):
            bolus_U = carbs_g / ICR
            bolus_mU = bolus_U * 1000.0
            rate += bolus_mU / BOLUS_DURATION_MIN
    return rate


# --- Pump resolution (discrete delivery steps) -------------------------

PUMP_STEP_U = 0.025                # minimum deliverable increment (U)
PUMP_STEP_mU = PUMP_STEP_U * 1000.0  # 25 mU


def quantize_to_pump_resolution(rate_mU_per_min: float, dt_min: float) -> float:
    """
    Rounds a commanded infusion rate to the nearest rate the real pump can
    actually deliver in one control interval, given its minimum step size
    of 0.025 U. The PID and bolus logic above both output a mathematically
    continuous rate; this is the last step before it reaches the "motor",
    modelling the leadscrew's discrete microstep resolution.
    """
    dose_mU = rate_mU_per_min * dt_min
    quantized_dose_mU = round(dose_mU / PUMP_STEP_mU) * PUMP_STEP_mU
    return quantized_dose_mU / dt_min

def personalized_Kp(r_basal_patient: float, tau_C: float = 60.0, bolus_TDI: float = 21.5) -> float:
    """
    Huyett et al. (2015) IMC tuning formula, evaluated using this
    patient's own real basal insulin need rather than a shared value:
        Kc = 0.023 * TDI / (tau_C + 11)   [U/h per mg/dL]
    tau_C = 40 min: the fastest-response setting the original paper
    itself selected. bolus_TDI: this design's daily bolus load
    (45+70+80+20 g carbs / ICR=10 g/U = 21.5 U/day), added to basal
    TDI to get each patient's total daily insulin.
    """
    TDI = r_basal_patient * 60 * 24 / 1000 + bolus_TDI
    Kc_Uh = 0.023 * TDI / (tau_C + 11)
    return Kc_Uh * 1000 / 60  # mU/min per mg/dL