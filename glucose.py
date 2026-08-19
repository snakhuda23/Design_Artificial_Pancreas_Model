import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

#Parameters taken from UVa/Padova Adult T1DM parameter table 10 at url = https://noesis.uis.edu.co/server/api/core/bitstreams/f28c7546-9521-4c71-9786-74e9b43262bd/content?
#Model taken from "" by Molano-Jiménez et al. at url: https://0-ieeexplore-ieee-org.innopac.wits.ac.za/stamp/stamp.jsp?tp=&arnumber=8276390

V_g = 1.77 #dL/kg
G_b = 110.0 #mg/dL -> chosen initial basal glucose target or use 100 mg/dL from url = https://diabetesjournals.org/care/article/33/1/121/30006/Closed-Loop-Insulin-Delivery-Using-a-Subcutaneous
k_1 = 0.0588 #1/min
k_2 = 0.2132 #1/min
_BW = 79.79 #body weight in kg

F_cns = 1.0 #mg/kg/min

# Endogenous glucose production
kp1 = 4.6576        # mg/kg/min
kp2 = 0.0033       # 1/min
kp3 = 0.0083       # model-specific insulin sensitivity coefficient
ki = 0.0092        # 1/min

# Insulin-dependent glucose utilisation
Vm0 = 7.2283        # mg/kg/min
Vmx = 0.1276       # model-specific coefficient
Km0 = 218.8779       # mg/kg
p2U = 0.047       # 1/min
E = 0.0     #excretion

# Renal excretion
ke1 = 0.0007       # 1/min
ke2 = 269.0        # mg/kg

V_g_total = V_g * _BW
G_p_basal = G_b * V_g_total
insulin_basal_mU_per_L = 21.1  #mU/L

#Basal glucose 

Glucose_basal = G_b * V_g * _BW


insulin_basal_mU_per_L = 21.1

#Unit Conversions:

def insulin_umL_to_pmolL(insulin_mU_per_L : float) -> float:
    return insulin_mU_per_L * 6.0 #from literature

I_basal = insulin_umL_to_pmolL(insulin_basal_mU_per_L)
X_basal = 0.0
I_basal_prime = I_basal
X_basal_L = I_basal


# Insulin-stimulated glucose uptake 

def insulin_dependent_utilisation(
        G_t: float,
        X: float,
) -> float:

    # Maximum glucose utilisation

    Vm = Vm0 + Vmx * X

    # Km is converted to total glucose amount

    Km_total = Km0 * _BW

    # Total glucose utilisation

    U_id = (_BW * Vm * G_t / (Km_total + G_t))

    return U_id


# Calculate basal glucose 


def tissue_steady_state_equation(
        G_t: float,
) -> float:

    U_id = insulin_dependent_utilisation(
        G_t, X_basal,
    )

    return (-U_id + k_1 * G_p_basal - k_2 * G_t)


# Solve for basal tissue glucose

steady_state_soln = root_scalar(
    tissue_steady_state_equation,
    bracket=[0.0, 100000.0],
    method="brentq",
)


if not steady_state_soln.converged:
    raise RuntimeError(
        "Could not calculate basal tissue glucose"
    )


G_t_basal = steady_state_soln.root


# Basal glucose utilisation 

U_id_basal = insulin_dependent_utilisation(G_t_basal, X_basal,
)


# Insulin-independent glucose utilisation 

U_ii = F_cns * _BW

# Basal endogenous glucose production

# At steady state:
#
# dGp/dt = 0
#
# Therefore:
#
# EGP = Uii + Uid + k1*Gp - k2*Gt
#
# assuming:
#
# Ra = 0
# renal excretion = 0

EGP_basal = (U_ii + U_id_basal)


def derivatives(
        min_time: float,
        state: np.ndarray,
        insulin_function,
        glucose_appearance_function,
) -> np.ndarray:

    G_p, G_t, I_prime, X_L, X = state

    # Insulin input

    insulin_mU_L = insulin_function(
        min_time
    )

    I = insulin_umL_to_pmolL(
        insulin_mU_L
    )

    # Meal glucose appearance

    R_a = glucose_appearance_function(
        min_time
    )

    # Insulin action

    dI_prime_dt = (-ki * (I_prime - I))

    dX_L_dt = (-ki * (X_L - I_prime))

    dX_dt = (-p2U * X + p2U * (I - I_basal))

    # Glucose utilisation:

    U_id = insulin_dependent_utilisation(
        G_t,
        X,
    )

    U_ii = F_cns * _BW

    # Endogenous glucose production:

    EGP = (
        EGP_basal - kp2 * (G_p - G_p_basal) - kp3 * _BW * (X_L - X_basal_L)
    )

    # Prevent negative glucose production

    EGP = max(EGP, 0.0)

    # Renal excretion:

    # Renal excretion threshold
    if G_p > (ke2 * _BW):
        E = ke1 * (G_p - ke2 * _BW)
    else:
        E = 0.0

    # Plasma glucose:

    dG_p_dt = (EGP + R_a - U_ii - E - k_1 * G_p + k_2 * G_t
    )

    # Tissue glucose:

    dG_t_dt = (-U_id + k_1 * G_p - k_2 * G_t
    )

    return np.array(
        [
            dG_p_dt,
            dG_t_dt,
            dI_prime_dt,
            dX_L_dt,
            dX_dt,
        ],
        dtype=float,
    )

# Print basal configuration 

print()

print("BASAL GLUCOSE CONFIGURATION")

print(
    f"Body weight: "
    f"{_BW:.2f} kg"
)

print(
    f"Basal glucose concentration: "
    f"{G_b:.2f} mg/dL"
)

print(
    f"Basal plasma glucose: "
    f"{G_p_basal:.2f} mg"
)

print(
    f"Basal tissue glucose: "
    f"{G_t_basal:.2f} mg"
)

print(
    f"Basal insulin-independent utilisation: "
    f"{U_ii:.2f} mg/min"
)

print(
    f"Basal insulin-dependent utilisation: "
    f"{U_id_basal:.2f} mg/min"
)

print(
    f"Basal endogenous glucose production: "
    f"{EGP_basal:.2f} mg/min"
)


# Insulin action derivatives:

def insulin_action_derivatives(
        plasma_insulin_mU_L: float,
        I_prime: float,
        X_L: float,
        X: float,
):

    # Convert insulin concentration:

    I = insulin_umL_to_pmolL(
        plasma_insulin_mU_L
    )

    # Delayed insulin signal:

    dI_prime_dt = (-ki * (I_prime - I))

    # Delayed hepatic insulin action:

    dX_L_dt = (-ki * (X_L - I_prime))

    # Insulin action on glucose utilisation:

    dX_dt = (-p2U * X + p2U * (I - I_basal))

    return (
        dI_prime_dt,
        dX_L_dt,
        dX_dt,
    )

def plasma_glucose_conc(
        plasma_glucose_mass: np.ndarray,
) -> np.ndarray:

    return plasma_glucose_mass / V_g_total  #gives mg/dL

def constant_basal_insulin(
        min_time: float,
) -> float:

    return insulin_basal_mU_per_L

def zero_glucose_appearance(
        min_time: float,
) -> float:

    return 0.0

def perform_glucose_sim(
        insulin_function,
        glucose_appearance_function = zero_glucose_appearance,
        min_duration: float = 600.0,
        output_interval_min: float = 1.0,
):

        initial = np.array(
            [
                G_p_basal,
                G_t_basal,
                I_basal_prime,
                X_basal_L,
                X_basal,
            ],
            dtype=float,
        )

        time_points = np.arange(
            0.0,
            min_duration + output_interval_min,
            output_interval_min,
        )

        soln = solve_ivp(

            fun=lambda t, y: derivatives(
                min_time=t,
                state=y,
                insulin_function=insulin_function,
                glucose_appearance_function=glucose_appearance_function,
            ),

            t_span=(0.0, min_duration),

            y0=initial,

            t_eval=time_points,

            method="LSODA",

            rtol=1e-7,

            atol=1e-9,
        )

        if not soln.success:
            raise RuntimeError(
                f"Glucose sim failed: {soln.message}"
            )

        return soln