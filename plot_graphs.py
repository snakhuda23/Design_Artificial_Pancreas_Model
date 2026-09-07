"""
plot_graphs.py

Generates five report-ready plots. Tests 1-3 show glucose with a
smoothed insulin infusion overlay (single-line scenarios, where
insulin's response is the point). Tests 4-5 remain glucose-only,
since insulin overlay would clutter their multi-line comparisons.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_tests import run_scenario

meals_24h = [
    (60.0, 45.0, 15.0),
    (390.0, 70.0, 20.0),
    (750.0, 80.0, 25.0),
    (960.0, 20.0, 10.0),
]


def smooth(arr, window=10):
    """Rolling average for DISPLAY only -- removes CGM-noise jitter
    while keeping real features (bolus spikes, basal shifts) visible."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def main():
    print("Running scenarios...")
    res_fasting = run_scenario("Test 1: 12h Fasting Baseline", 720.0, meal_events=[], closed_loop=True)
    res_3meals = run_scenario("Test 2: 24h 3-Meal + Snack Trial", 1440.0, meal_events=meals_24h, closed_loop=True)
    res_exercise = run_scenario("Test 3: Exercise Disturbance", 1440.0, meal_events=meals_24h, exercise_event=(540.0, 45.0, 1.8), closed_loop=True)
    res_t1d_open = run_scenario("Test 4A: Uncontrolled Open-Loop", 1440.0, meal_events=meals_24h, closed_loop=False)

    variability_patients = ["adult_low_sens", "adult_normal", "adult_high_sens"]
    variability_results = {
        n: run_scenario(f"Test 5: {n}", 1440.0, meal_events=meals_24h, closed_loop=True, patient_name=n)
        for n in variability_patients
    }

    print("Building figures...")

    # ---------------- Test 1: Fasting -- WITH insulin overlay ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res_fasting["time_h"], res_fasting["glucose"], color="tab:blue", lw=2.2, label="Glucose (mmol/L)")
    ax.axhspan(3.9, 10.0, color="green", alpha=0.12, label="Safe Range (3.9-10.0)")
    ax.axhline(6.1, color="blue", linestyle=":", lw=1.2, label="Basal Target (6.1 mmol/L)")
    ax_ins = ax.twinx()
    ax_ins.plot(res_fasting["time_h"], smooth(res_fasting["infusion"]), color="tab:purple", lw=1.6, linestyle="--", alpha=0.75, label="Insulin Infusion (mU/min)")
    ax_ins.set_ylabel("Insulin Infusion (mU/min)", color="tab:purple")
    ax.set_title("Test 1: 12-Hour Fasting Baseline Stability", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Glucose (mmol/L)")
    ax.grid(True, alpha=0.4)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax_ins.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, loc="upper right")
    plt.tight_layout(); plt.savefig("test1_fasting.png", dpi=140); plt.close(fig)
    print("Saved test1_fasting.png")

    # ---------------- Test 2: 3-Meal -- WITH insulin overlay ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res_3meals["time_h"], res_3meals["glucose"], color="tab:blue", lw=2.2, label="Glucose (mmol/L)")
    ax.axhspan(3.9, 10.0, color="green", alpha=0.12, label="Safe Range (3.9-10.0)")
    ax_ins = ax.twinx()
    ax_ins.plot(res_3meals["time_h"], smooth(res_3meals["infusion"]), color="tab:purple", lw=1.6, linestyle="--", alpha=0.75, label="Insulin Infusion (mU/min)")
    ax_ins.set_ylabel("Insulin Infusion (mU/min)", color="tab:purple")
    ax.set_title("Test 2: 24-Hour 3-Meal + Snack Trial", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Glucose (mmol/L)")
    ax.grid(True, alpha=0.4)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax_ins.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, loc="upper right")
    plt.tight_layout(); plt.savefig("test2_3meal.png", dpi=140); plt.close(fig)
    print("Saved test2_3meal.png")

    # ---------------- Test 3: Exercise -- WITH insulin overlay ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res_exercise["time_h"], res_exercise["glucose"], color="tab:blue", lw=2.2, label="Glucose (mmol/L)")
    ax.axvspan(9.0, 9.75, color="orange", alpha=0.25, label="45-min Exercise")
    ax.axhspan(3.9, 10.0, color="green", alpha=0.12, label="Safe Range (3.9-10.0)")
    ax_ins = ax.twinx()
    ax_ins.plot(res_exercise["time_h"], smooth(res_exercise["infusion"]), color="tab:purple", lw=1.6, linestyle="--", alpha=0.75, label="Insulin Infusion (mU/min)")
    ax_ins.set_ylabel("Insulin Infusion (mU/min)", color="tab:purple")
    ax.set_title("Test 3: Exercise Disturbance (Braking Response)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Glucose (mmol/L)")
    ax.grid(True, alpha=0.4)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax_ins.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, loc="upper right")
    plt.tight_layout(); plt.savefig("test3_exercise.png", dpi=140); plt.close(fig)
    print("Saved test3_exercise.png")

    # ---------------- Test 4: Uncontrolled vs Closed-loop -- glucose ONLY ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res_t1d_open["time_h"], res_t1d_open["glucose"], color="tab:red", linestyle="--", lw=2.2, label="Uncontrolled (Fixed Basal)")
    ax.plot(res_3meals["time_h"], res_3meals["glucose"], color="tab:blue", lw=2.2, label="Closed-Loop Artificial Pancreas")
    ax.axhspan(3.9, 10.0, color="green", alpha=0.12, label="Target Safe Zone (3.9-10.0)")
    ax.axhline(10.0, color="gray", linestyle=":", lw=1.2, label="Hyperglycemia (10.0)")
    ax.axhline(3.9, color="red", linestyle=":", lw=1.2, label="Hypoglycemia (3.9)")
    ax.set_title("Test 4: Zero Endogenous Insulin (C-Peptide Negative)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Glucose (mmol/L)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig("test4_uncontrolled_vs_closedloop.png", dpi=140); plt.close(fig)
    print("Saved test4_uncontrolled_vs_closedloop.png")

    # ---------------- Test 5: Patient Variability -- glucose ONLY ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"adult_low_sens": "tab:red", "adult_normal": "tab:blue", "adult_high_sens": "tab:green"}
    labels = {"adult_low_sens": "Low sensitivity", "adult_normal": "Normal sensitivity", "adult_high_sens": "High sensitivity"}
    for name, r in variability_results.items():
        ax.plot(r["time_h"], r["glucose"], color=colors[name], lw=2.0, label=f"{labels[name]} (peak {r['peak']:.2f})")
    ax.axhspan(3.9, 10.0, color="green", alpha=0.12, label="Safe Range (3.9-10.0)")
    ax.set_title("Test 5: Patient Variability -- 3-Meal Day Across Insulin Sensitivities", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Glucose (mmol/L)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig("test5_variability.png", dpi=140); plt.close(fig)
    print("Saved test5_variability.png")

    print("\nAll five figures generated successfully.")


if __name__ == "__main__":
    main()