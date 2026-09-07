import numpy as np
from scipy.integrate import solve_ivp

#Parameters were taken from the paper: "UVa/Padova T1DMS Dynamic Model Revision" by Molano-Jiménez et al.
#at url = https://0-ieeexplore-ieee-org.innopac.wits.ac.za/stamp/stamp.jsp?tp=&arnumber=8276390

k_max = 0.0558     # Maximum gastric emptying rate (1/min)
k_min = 0.0080     # Minimum gastric emptying rate (1/min)
k_abs = 0.0570     # Intestinal absorption rate constant (1/min)
k_gri = 0.0558     # Stomach grinding rate: solid -> liquid (1/min)
f = 0.90           # Fraction of ingested glucose absorbed into blood (bioavailability)

# Parameters for nonlinear gastric emptying function K_e(Q_sto)
b = 0.82
c = 0.00236
alpha = 0.00013    # 1/mg
beta = 0.05        # 1/mg
D_th = 23000.0     # Small meal threshold: 23g (23,000 mg)


def k_empt(Q_sto: float, D_n: float) -> float:
    """
    Computes the nonlinear stomach emptying rate k_empt(Q_sto)
    from Equations (11) and (16) in the paper.
    """
    if D_n <= 0.0:
        return k_min

    # Non-linear scaling factor K_e (Equation 11)
    K_e = (
        np.tanh(alpha * (Q_sto - b * D_n))
        - np.tanh(beta * (Q_sto - c * D_n))
        + 2.0
    )

    if D_n > D_th:
        # Standard meal emptying rate
        return k_min + ((k_max - k_min) / 2.0) * K_e
    else:
        # Small snack meal adjustment (Section II.C)
        return (D_n / D_th) * (k_min + ((k_max - k_min) / 2.0) * K_e)


def meal_derivatives(
    t: float,
    state: np.ndarray,
    meal_events: list,  # List of tuples: [(start_time_min, carbs_in_grams, duration_min)]
) -> np.ndarray:
    """
    Three-compartment differential equations for gut absorption (Equation 16):
    state = [Q_sto1, Q_sto2, Q_gut] (all in mg)
    """
    Q_sto1, Q_sto2, Q_gut = state

    # 1. Ingested glucose input rate IG(t) in mg/min (Equation 15)
    IG = 0.0
    D_n = 0.0
    for meal_start, carbs_g, duration in meal_events:
        carbs_mg = carbs_g * 1000.0
        if meal_start <= t < (meal_start + duration):
            IG += carbs_mg / duration
        if t >= meal_start:
            D_n += carbs_mg

    Q_sto = Q_sto1 + Q_sto2
    D_n_total = D_n + Q_sto

    # 2. Compute stomach emptying rate for current fullness
    k_e = k_empt(Q_sto, D_n_total)

    # 3. ODEs (Equation 16 in Appendix D)
    dQ_sto1_dt = -k_gri * Q_sto1 + IG
    dQ_sto2_dt = -k_e * Q_sto2 + k_gri * Q_sto1
    dQ_gut_dt = -k_abs * Q_gut + k_e * Q_sto2

    return np.array([dQ_sto1_dt, dQ_sto2_dt, dQ_gut_dt], dtype=float)


def simulate_meal_absorption(
    meal_events: list,
    min_duration: float = 600.0,
    output_interval_min: float = 1.0,
):
    """
    Solves the meal model and returns the rate of glucose appearance Ra(t) in mg/min.
    """
    initial_state = np.array([0.0, 0.0, 0.0], dtype=float)  # Fasting: gut is empty
    time_points = np.arange(0.0, min_duration + output_interval_min, output_interval_min)

    soln = solve_ivp(
        fun=lambda t, y: meal_derivatives(t, y, meal_events),
        t_span=(0.0, min_duration),
        y0=initial_state,
        t_eval=time_points,
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
        max_step = 1.0,
    )

    if not soln.success:
        raise RuntimeError(f"Meal absorption sim failed: {soln.message}")

    Q_gut = soln.y[2]
    # Glucose rate of appearance into bloodstream (Equation 16): Ra = f * k_abs * Q_gut (mg/min)
    R_a_profile = f * k_abs * Q_gut

    return soln.t, R_a_profile
