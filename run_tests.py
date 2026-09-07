import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from insulin_new import (
    q_ip1_basal, q_ip2_basal, q_basal_l, q_basal_p,
    r_basal, v_i, derivatives as insulin_derivatives
)
from glucose import (
    G_p_basal, I_basal_prime, X_basal_L, X_basal,
    V_g, derivatives as glucose_derivatives, insulin_dependent_utilisation,
    setup_patient,
)
from meal_disturbances import meal_derivatives
from controller_pid import PIDcontroller
from patient import patient_types

from insulin_pump_delivery import meal_bolus_rate, quantize_to_pump_resolution

from cgm_sensor_noise import add_noise_from_cgm


def personalized_Kp(r_basal_patient: float, tau_C: float = 60.0, bolus_TDI: float = 21.5) -> float:
    """
    Huyett et al. (2015) IMC tuning formula, evaluated per-patient
    using their own real basal insulin need (mU/min), rather than a
    shared value across all patients.
    """
    TDI = r_basal_patient * 60 * 24 / 1000 + bolus_TDI
    Kc_Uh = 0.023 * TDI / (tau_C + 11)
    return Kc_Uh * 1000 / 60  # mU/min per mg/dL

def run_scenario(
    scenario_name: str,
    duration_min: float = 1440.0,
    meal_events: list = [],
    exercise_event: tuple = (0.0, 0.0, 1.0),
    closed_loop: bool = True,
    patient_name: str = "original",
):
    patient_params = patient_types[patient_name]
    setup_kwargs = {k: v for k, v in patient_params.items() if k != "r_basal"}
    patient = setup_patient(**setup_kwargs)

    dt = 1.0
    time_steps = np.arange(0.0, duration_min + dt, dt)
    n_steps = len(time_steps)

    state_meal = np.array([0.0, 0.0, 0.0], dtype=float)
    state_insulin = np.array([q_ip1_basal, q_ip2_basal, q_basal_l, q_basal_p], dtype=float)
    state_glucose = np.array([G_p_basal, patient.G_t_basal, I_basal_prime, X_basal_L, X_basal], dtype=float)

    patient_r_basal = patient_params["r_basal"]
    patient_Kp = personalized_Kp(patient_r_basal)
    controller = PIDcontroller(
        target_glucose=110.0, r_basal=patient_r_basal,
        Kp=patient_Kp, Ti=273.0, Td=23.5, sample_time_min=dt,
    )

    history_glucose_mmol = np.zeros(n_steps)
    history_infusion = np.zeros(n_steps)
    history_Ra = np.zeros(n_steps)

    ex_start, ex_dur, ex_int = exercise_event

    for k in range(n_steps):
        t = time_steps[k]

        glucose_mg_dl = state_glucose[0] / V_g
        history_glucose_mmol[k] = glucose_mg_dl / 18.0182
        cgm_reading_in_mg_dl = add_noise_from_cgm(glucose_mg_dl)
        plasma_insulin_mU_L = state_insulin[3] / v_i

        plasma_insulin_mU_L = state_insulin[3] / v_i

        if closed_loop:
            infusion_rate = controller.compute(cgm_reading_in_mg_dl)
            infusion_rate += meal_bolus_rate(t, meal_events)
        else:
            infusion_rate = patient_r_basal

        infusion_rate = quantize_to_pump_resolution(infusion_rate, dt)
        history_infusion[k] = infusion_rate

        current_Ra_total = 0.90 * 0.0570 * state_meal[2]
        current_Ra_per_kg = current_Ra_total / patient.BW
        history_Ra[k] = current_Ra_total

        if k == n_steps - 1:
            break

        t_span = (t, t + dt)

        sol_meal = solve_ivp(
            lambda t_i, y: meal_derivatives(t_i, y, meal_events),
            t_span, state_meal, method="LSODA"
        )
        state_meal = sol_meal.y[:, -1]

        sol_ins = solve_ivp(
            lambda t_i, y: insulin_derivatives(t_i, y, lambda _: infusion_rate),
            t_span, state_insulin, method="LSODA"
        )
        state_insulin = sol_ins.y[:, -1]

        exercise_on = ex_start <= t < (ex_start + ex_dur)

        def custom_glucose_derivs(t_i, y):
            dydt = glucose_derivatives(
                t_i, y,
                insulin_function=lambda _: plasma_insulin_mU_L,
                glucose_appearance_function=lambda _: current_Ra_per_kg,
                patient=patient,
            )
            if exercise_on:
                G_t, X = y[1], y[4]
                U_id_now = insulin_dependent_utilisation(G_t, X, patient.Vmx, patient.Km0, patient.Vm0)
                dydt[1] -= U_id_now * (ex_int - 1.0)
            return dydt

        sol_glu = solve_ivp(custom_glucose_derivs, t_span, state_glucose, method="LSODA")
        state_glucose = sol_glu.y[:, -1]

    tir = np.mean((history_glucose_mmol >= 3.9) & (history_glucose_mmol <= 10.0)) * 100.0
    tbr = np.mean(history_glucose_mmol < 3.9) * 100.0
    tar = np.mean(history_glucose_mmol > 10.0) * 100.0

    return {
        "name": scenario_name,
        "time_h": time_steps / 60.0,
        "glucose": history_glucose_mmol,
        "infusion": history_infusion,
        "Ra": history_Ra,
        "tir": tir,
        "tbr": tbr,
        "tar": tar,
        "peak": np.max(history_glucose_mmol),
        "nadir": np.min(history_glucose_mmol),
    }


def main():
    print("=================================================================")
    print("       RUNNING FULL ARTIFICIAL PANCREAS TEST PROTOCOL            ")
    print("=================================================================")

    meals_24h = [
        (60.0, 45.0, 15.0),
        (390.0, 70.0, 20.0),
        (750.0, 80.0, 25.0),
        (960.0, 20.0, 10.0),
    ]

    # --- Original four tests, all on the "original" patient (your
    # existing glucose.py parameters), so existing report numbers
    # stay valid ---
    res_fasting = run_scenario("Test 1: 12h Fasting Baseline", 720.0, meal_events=[], closed_loop=True)
    res_3meals = run_scenario("Test 2: 24h 3-Meal + Snack Trial", 1440.0, meal_events=meals_24h, closed_loop=True)
    res_exercise = run_scenario("Test 3: Exercise Disturbance", 1440.0, meal_events=meals_24h, exercise_event=(540.0, 45.0, 1.8), closed_loop=True)
    res_t1d_open = run_scenario("Test 4A: Uncontrolled Open-Loop", 1440.0, meal_events=meals_24h, closed_loop=False)
    res_t1d_closed = run_scenario("Test 4B: Closed-Loop AP", 1440.0, meal_events=meals_24h, closed_loop=True)

    all_tests = [res_fasting, res_3meals, res_exercise, res_t1d_open, res_t1d_closed]

    print("\n" + "=" * 95)
    print(f"{'Scenario / Test Name':<55} | {'TIR %':<7} | {'TBR %':<7} | {'Peak':<7} | {'Nadir':<7}")
    print("=" * 95)
    for r in all_tests:
        print(f"{r['name']:<55} | {r['tir']:<7.1f} | {r['tbr']:<7.1f} | {r['peak']:<7.2f} | {r['nadir']:<7.2f}")
    print("=" * 95 + "\n")

    # --- New Test 5: patient variability (sensitivity + body weight
    # together), 3-meal day across three real patients (does NOT
    # include "original") ---
    print("=" * 95)
    print("       TEST 5: PATIENT VARIABILITY (3-Meal Day)                  ")
    print("=" * 95)
    variability_results = {}
    variability_patients = ["adult_low_sens", "adult_normal", "adult_high_sens"]
    for name in variability_patients:
        variability_results[name] = run_scenario(
            f"Test 5: {name}", 1440.0, meal_events=meals_24h, closed_loop=True, patient_name=name
        )
    print(f"{'Patient':<25} | {'TIR %':<7} | {'TBR %':<7} | {'Peak':<7} | {'Nadir':<7}")
    print("-" * 95)
    for name, r in variability_results.items():
        print(f"{name:<25} | {r['tir']:<7.1f} | {r['tbr']:<7.1f} | {r['peak']:<7.2f} | {r['nadir']:<7.2f}")
    print("=" * 95 + "\n")

    # =========================================================================
    # 2x2 CLEAR DUAL-AXIS SUBPLOTS (unchanged from before)
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    ax1 = axs[0]
    ax1_ins = ax1.twinx()
    l1 = ax1.plot(res_fasting["time_h"], res_fasting["glucose"], color="tab:blue", lw=2.5, label="Glucose (mmol/L)")
    l2 = ax1_ins.plot(res_fasting["time_h"], res_fasting["infusion"], color="tab:purple", lw=1.8, linestyle="--", label="Pump Infusion (mU/min)")
    ax1.axhline(6.1, color="blue", linestyle=":", label="Basal Target (6.1 mmol/L)")
    ax1.set_title("Test 1: 12-Hour Fasting Baseline Stability", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (Hours)")
    ax1.set_ylabel("Glucose (mmol/L)", color="tab:blue", fontweight="bold")
    ax1_ins.set_ylabel("Pump Infusion (mU/min)", color="tab:purple", fontweight="bold")
    ax1.set_ylim(2.0, 12.0)
    ax1_ins.set_ylim(0.0, 200.0)
    ax1.grid(True)
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    ax2 = axs[1]
    ax2_ins = ax2.twinx()
    l1 = ax2.plot(res_3meals["time_h"], res_3meals["glucose"], color="tab:blue", lw=2.5, label="Glucose (mmol/L)")
    l2 = ax2_ins.plot(res_3meals["time_h"], res_3meals["infusion"], color="tab:purple", lw=1.8, linestyle="--", label="Pump Infusion (mU/min)")
    ax2.axhspan(3.9, 10.0, color="green", alpha=0.15, label="Safe Range (3.9-10.0)")
    ax2.set_title("Test 2: 24-Hour 3-Meal + Snack Trial", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (Hours)")
    ax2.set_ylabel("Glucose (mmol/L)", color="tab:blue", fontweight="bold")
    ax2_ins.set_ylabel("Pump Infusion (mU/min)", color="tab:purple", fontweight="bold")
    ax2.set_ylim(2.0, 13.0)
    ax2_ins.set_ylim(0.0, 220.0)
    ax2.grid(True)
    lines = l1 + l2
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right")

    ax3 = axs[2]
    ax3_ins = ax3.twinx()
    l1 = ax3.plot(res_exercise["time_h"], res_exercise["glucose"], color="tab:blue", lw=2.5, label="Glucose (mmol/L)")
    l2 = ax3_ins.plot(res_exercise["time_h"], res_exercise["infusion"], color="tab:purple", lw=1.8, linestyle="--", label="Pump Infusion (mU/min)")
    ax3.axvspan(9.0, 9.75, color="orange", alpha=0.25, label="45-min Exercise (15:00)")
    ax3.axhspan(3.9, 10.0, color="green", alpha=0.15)
    ax3.set_title("Test 3: Exercise Disturbance (Braking Response)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (Hours)")
    ax3.set_ylabel("Glucose (mmol/L)", color="tab:blue", fontweight="bold")
    ax3_ins.set_ylabel("Pump Infusion (mU/min)", color="tab:purple", fontweight="bold")
    ax3.set_ylim(2.0, 13.0)
    ax3_ins.set_ylim(0.0, 220.0)
    ax3.grid(True)
    lines = l1 + l2
    ax3.legend(lines, [l.get_label() for l in lines], loc="upper right")

    ax4 = axs[3]
    l1 = ax4.plot(res_t1d_open["time_h"], res_t1d_open["glucose"], color="tab:red", linestyle="--", lw=2.5, label="Uncontrolled (Fixed Basal)")
    l2 = ax4.plot(res_t1d_closed["time_h"], res_t1d_closed["glucose"], color="tab:blue", lw=2.5, label="Closed-Loop Artificial Pancreas")
    ax4.axhspan(3.9, 10.0, color="green", alpha=0.15, label="Target Safe Zone (3.9-10.0)")
    ax4.axhline(10.0, color="gray", linestyle=":", label="Hyperglycemia (10.0)")
    ax4.axhline(3.9, color="red", linestyle=":", label="Hypoglycemia (3.9)")
    ax4.set_title("Test 4: Zero Endogenous Insulin (C-Peptide Negative)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time (Hours)")
    ax4.set_ylabel("Glucose (mmol/L)", color="black", fontweight="bold")
    ax4.set_ylim(2.0, 25.0)
    ax4.grid(True)
    ax4.legend(loc="upper right")

    plt.tight_layout()

    # =========================================================================
    # NEW: Test 5 patient variability plot -- three patients overlaid
    # =========================================================================
    fig2, ax5 = plt.subplots(figsize=(11, 6))
    colors = {"adult_low_sens": "tab:red", "adult_normal": "tab:blue", "adult_high_sens": "tab:green"}
    for name, r in variability_results.items():
        ax5.plot(r["time_h"], r["glucose"], color=colors[name], lw=2.2, label=f"{name} (peak {r['peak']:.2f})")
    ax5.axhspan(3.9, 10.0, color="green", alpha=0.12, label="Safe Range (3.9-10.0)")
    ax5.set_title("Test 5: Patient Variability -- 3-Meal Day Across Insulin Sensitivities", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Time (Hours)")
    ax5.set_ylabel("Glucose (mmol/L)")
    ax5.grid(True)
    ax5.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()