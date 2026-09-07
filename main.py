import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Subsystem Imports
from insulin_new import (
    basal_insulin_p,
    q_basal_l,
    q_basal_p,
    r_basal,
    v_i,
    q_ip1_basal,
    q_ip2_basal,
    derivatives as insulin_derivatives,
)

from glucose import (
    G_p_basal,
    I_basal_prime,
    X_basal_L,
    X_basal,
    V_g,
    derivatives as glucose_derivatives,
    setup_patient,
)

from meal_disturbances import meal_derivatives

from controller_pid import PIDcontroller

from insulin_pump_delivery import meal_bolus_rate, quantize_to_pump_resolution

from patient import patient_types

from glucose import setup_patient 
patient = setup_patient(**patient_types["original"])

from cgm_sensor_noise import add_noise_from_cgm

def main():
    print("Closed-loop Intraperitoneal Insulin Delivery Simulation")
    print(f"Basal plasma insulin concentration: {basal_insulin_p:.2f} mU/L")
    print(f"Initial plasma insulin mass:        {q_basal_p:.2f} mU")
    print(f"Initial liver insulin mass:         {q_basal_l:.2f} mU")
    print(f"Basal continuous IP infusion:       {r_basal:.4f} mU/min\n")

    duration = 600.0  # 600 mins = 10 hours
    dt = 1.0          # 1-minute control & integration step
    time_steps = np.arange(0.0, duration + dt, dt)
    n_steps = len(time_steps)

    # -------------------------------------------------------------
    # INITIALIZE STATE VECTORS
    # -------------------------------------------------------------
    # Meal states: [Q_sto1, Q_sto2, Q_gut] (all starting at 0 mg)
    state_meal = np.array([0.0, 0.0, 0.0], dtype=float)

    # Insulin states: [Q_ip1, Q_ip2, q_l, q_p] (all starting at basal)
    state_insulin = np.array([q_ip1_basal, q_ip2_basal, q_basal_l, q_basal_p], dtype=float)

    # Glucose states: [G_p, G_t, I_prime, X_L, X] (all starting at basal)
    state_glucose = np.array([G_p_basal, patient.G_t_basal, I_basal_prime, X_basal_L, X_basal], dtype=float)

    # Meal event: 45g Carbs at t = 60 min, lasting 15 mins
    meal_disturbance_event = [(60.0, 45.0, 15.0)]

    # -------------------------------------------------------------
    # INITIALISE CONTROLLER HERE
    # -------------------------------------------------------------
    controller = PIDcontroller(
        target_glucose=110.0,
        r_basal=r_basal,
        sample_time_min=dt,
    )

    # Logging arrays for plotting
    history_glucose = np.zeros(n_steps)
    history_insulin_conc = np.zeros(n_steps)
    history_infusion = np.zeros(n_steps)
    history_Ra = np.zeros(n_steps)
    history_cgm_reading = np.zeros(n_steps)

    # -------------------------------------------------------------
    # MINUTE-BY-MINUTE CLOSED-LOOP SIMULATION
    # -------------------------------------------------------------
    print("Running closed-loop simulation...")
    for k in range(n_steps):
        t_current = time_steps[k]

        # 1. Read Current Blood Glucose (mg/dL) (assumes ideal glucose sensor)
        glucose_mg_dl = state_glucose[0] / V_g
        glucose_mmol_per_litre = glucose_mg_dl / 18.0182
        history_glucose[k] = glucose_mmol_per_litre

        # Simulated CGM reading -- what the controller actually gets,
        # with sensor noise applied
        cgm_reading_mg_dl = add_noise_from_cgm(glucose_mg_dl)
        history_cgm_reading[k] = cgm_reading_mg_dl / 18.0182

        # 2. Read Current Plasma Insulin Concentration (mU/L)
        plasma_insulin_mU_L = state_insulin[3] / v_i
        history_insulin_conc[k] = plasma_insulin_mU_L

        # 3. CONTROLLER COMPUTES IP PUMP INFUSION (mU/min)
        # 3a. PID: basal + reactive correction, based on current glucose
        infusion_rate = controller.compute(glucose_mg_dl)

        # 3b. Feedforward meal bolus, added on top of the PID output
        infusion_rate += meal_bolus_rate(t_current, meal_disturbance_event)

        # 3c. Quantise to the real pump's 0.025 U minimum step size
        infusion_rate = quantize_to_pump_resolution(infusion_rate, dt)

        history_infusion[k] = infusion_rate

        # 4. Calculate Current Meal Glucose Appearance Ra (mg/min)
        # Ra = f * k_abs * Q_gut
        current_Ra_total = 0.90 * 0.0570 * state_meal[2]
        current_Ra = current_Ra_total / patient.BW
        history_Ra[k] = current_Ra

        # Stop at the final time point
        if k == n_steps - 1:
            break

        # 5. Integrate ODEs across the 1-minute step [t, t + dt]
        t_span = (t_current, t_current + dt)

        # A. Integrate Meal Gastrointestinal ODEs
        sol_meal = solve_ivp(
            fun=lambda t, y: meal_derivatives(t, y, meal_disturbance_event),
            t_span=t_span,
            y0=state_meal,
            method="LSODA",
        )
        state_meal = sol_meal.y[:, -1]

        # B. Integrate Intraperitoneal Insulin Delivery ODEs
        sol_insulin = solve_ivp(
            fun=lambda t, y: insulin_derivatives(t, y, lambda t: infusion_rate),
            t_span=t_span,
            y0=state_insulin,
            method="LSODA",
        )
        state_insulin = sol_insulin.y[:, -1]

        # C. Integrate UVA/Padova Glucose ODEs
        sol_glucose = solve_ivp(
            lambda t_i, y: glucose_derivatives(
                min_time=t_i,
                state=y,
                insulin_function=lambda t: plasma_insulin_mU_L,
                glucose_appearance_function=lambda t: current_Ra,
                patient = patient,
            ),
            t_span=t_span,
            y0=state_glucose,
            method="LSODA",
        )
        state_glucose = sol_glucose.y[:, -1]

    # -------------------------------------------------------------
    # SIMULATION RESULTS & METRICS
    # -------------------------------------------------------------
    print("\nSim Results:")
    print(f"Fasting Glucose (0 min):      {history_glucose[0]:.2f} mmol/L")
    print(f"Glucose before meal (60 min): {history_glucose[60]:.2f} mmol/L")
    print(f"Peak Glucose After Meal:     {np.max(history_glucose):.2f} mmol/L")
    print(f"Minimum Glucose (Nadir):     {np.min(history_glucose):.2f} mmol/L")
    print(f"Final Glucose (600 min):      {history_glucose[-1]:.2f} mmol/L")

    # Time-in-Range (3.9 to 10 mmol/L)
    tir = np.mean((history_glucose >= 3.9) & (history_glucose <= 10.0)) * 100.0
    print(f"Time-in-Range (3.9–10.0 mmol/L): {tir:.1f} %")

    # -------------------------------------------------------------
    # PLOT ALL RESULTS
    # -------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. Meal Glucose Appearance Rate Ra(t)
    axs[0].plot(time_steps, history_Ra, color="tab:green", label="Glucose Appearance $R_a(t)$")
    axs[0].set_ylabel("$R_a$ (mg/min)")
    axs[0].set_title("Meal Disturbance: 45g Carbs at $t = 60$ min")
    axs[0].grid(True)
    axs[0].legend()

    # 2. Intraperitoneal Insulin Pump Delivery & Resulting Plasma Insulin
    axs[1].plot(time_steps, history_infusion, color="tab:purple", label="PID Pump Command (mU/min)")
    axs[1].plot(time_steps, history_insulin_conc, color="tab:orange", linestyle="--", label="Plasma [Insulin] (mU/L)")
    axs[1].set_ylabel("Insulin Rates")
    axs[1].set_title("IP Insulin Delivery & Plasma Response")
    axs[1].grid(True)
    axs[1].legend()

    # 3. Blood Glucose Response
    axs[2].plot(time_steps, history_glucose, color="tab:blue", linewidth=2, label="True Blood Glucose")
    axs[2].scatter(time_steps, history_cgm_reading, color="tab:gray", s=6, alpha=0.4, label="Noisy CGM Reading")
    axs[2].axhspan(3.9, 10.0, color="green", alpha=0.15, label="Target Safe Range (3.9-10.0 mmol/L)")
    axs[2].axhline(y=10.0, color="gray", linestyle="--", label="Hyperglycemia Limit (10.0 mmol/L)")
    axs[2].axhline(y=6.1, color="blue", linestyle=":", label="Basal Target (6.1 mmol/L)")
    axs[2].axhline(y=3.9, color="red", linestyle="--", label="Hypoglycemia Limit (3.9 mmol/L)")

    axs[2].set_ylim(2.0, 13.0)

    axs[2].set_xlabel("Time (min)")
    axs[2].set_ylabel("Glucose (mmol/L)")
    axs[2].set_title("Blood Glucose Response (Closed-Loop PID Control)")
    axs[2].grid(True)
    axs[2].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()