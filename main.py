import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from insulin_new import (
    basal_insulin_p,
    q_basal_l,
    q_basal_p,
    r_basal,
    perform_insulin_sim,
    plasma_insulin_concentration,
)

from glucose import(
     perform_glucose_sim,
     plasma_glucose_conc,
     constant_basal_insulin,
     zero_glucose_appearance,

)


def main():

    print("=== INTRAPERITONEAL INSULIN & GLUCOSE SIMULATION ===")

    print(
        f"Basal plasma insulin concentration: "
        f"{basal_insulin_p:.2f} mU/L"
    )

    print(
        f"Initial plasma insulin mass: "
        f"{q_basal_p:.2f} mU"
    )

    print(
        f"Initial liver insulin mass: "
        f"{q_basal_l:.2f} mU"
    )

    print(f"Basal continuous infusion: {r_basal:.4f} mU/min\n")


    #tests:
    def test_insulin_infusion(min_time: float) -> float:

        if 100 <= min_time < 200:
            return 0.0

        return r_basal

    # 1. Run IP Insulin Simulation
    duration = 600.0
    soln_insulin = perform_insulin_sim(
        compartment_infusion_function=test_insulin_infusion,
        min_duration=duration,
        output_interval_min=1.0,
    )

    time_insulin = soln_insulin.t
    q_ip1 = soln_insulin.y[0]
    q_ip2 = soln_insulin.y[1]
    q_liver = soln_insulin.y[2]
    plasma_mass = soln_insulin.y[3]
    plasma_conc = plasma_insulin_concentration(plasma_mass)

    # 2. Couple Insulin output to Glucose Simulation
    insulin_interpolator = interp1d(
        time_insulin,
        plasma_conc,
        kind="linear",
        fill_value="extrapolate",
    )

    soln_glucose = perform_glucose_sim(
        insulin_function=lambda t: float(insulin_interpolator(t)),
        glucose_appearance_function=zero_glucose_appearance,
        min_duration=duration,
        output_interval_min=1.0,
    )

    time_glucose = soln_glucose.t
    glucose_mg_dl = plasma_glucose_conc(soln_glucose.y[0])

    print("=== SIMULATION RESULTS ===")
    print(f"Glucose at 0 min:   {glucose_mg_dl[0]:.2f} mg/dL")
    print(f"Glucose at 100 min: {glucose_mg_dl[100]:.2f} mg/dL")
    print(f"Glucose at 200 min: {glucose_mg_dl[200]:.2f} mg/dL (after pause)")
    print(f"Glucose at 600 min: {glucose_mg_dl[-1]:.2f} mg/dL (recovered)")

    # 3. Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Subplot 1: IP Compartment Masses
    axs[0].plot(time_insulin, q_ip1, label="IP Comp 1 (mU)")
    axs[0].plot(time_insulin, q_ip2, label="IP Comp 2 (mU)")
    axs[0].plot(time_insulin, q_liver, label="Liver Insulin (mU)")
    axs[0].plot(time_insulin, plasma_mass, label="Plasma Insulin Mass (mU)")
    axs[0].set_ylabel("Insulin Mass (mU)")
    axs[0].set_title("Intraperitoneal Insulin Compartments")
    axs[0].grid(True)
    axs[0].legend()

    # Subplot 2: Plasma Insulin Concentration
    axs[1].plot(time_insulin, plasma_conc, color="tab:orange", label="Plasma [Insulin]")
    axs[1].set_ylabel("Insulin (mU/L)")
    axs[1].set_title("Plasma Insulin Concentration")
    axs[1].grid(True)
    axs[1].legend()

    # Subplot 3: Plasma Glucose Response
    axs[2].plot(time_glucose, glucose_mg_dl, color="tab:red", label="Blood Glucose (mg/dL)")
    axs[2].axhline(y=110.0, color="gray", linestyle="--", label="Target Basal (110 mg/dL)")
    axs[2].set_xlabel("Time (min)")
    axs[2].set_ylabel("Glucose (mg/dL)")
    axs[2].set_title("Blood Glucose Response")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def test_for_basal_only(min_time: float) -> float:
    return r_basal

if __name__ == "__main__":
    main()