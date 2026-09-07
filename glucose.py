import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d


#Original patient parameters taken from UVa/Padova Adult T1DM parameter table 10 at url = https://noesis.uis.edu.co/server/api/core/bitstreams/f28c7546-9521-4c71-9786-74e9b43262bd/content?
#Model taken from "" by Molano-Jiménez et al. at url: https://0-ieeexplore-ieee-org.innopac.wits.ac.za/stamp/stamp.jsp?tp=&arnumber=8276390

V_g = 1.77
G_b = 110.0
F_cns = 1.0
ki = 0.0092
p2U = 0.047
insulin_basal_mU_per_L = 21.1

def insulin_umL_to_pmolL(insulin_mU_per_L: float) -> float:
    return insulin_mU_per_L * 6.0

I_basal = insulin_umL_to_pmolL(insulin_basal_mU_per_L)
X_basal = 0.0
I_basal_prime = I_basal
X_basal_L = I_basal
G_p_basal = G_b * V_g


def insulin_dependent_utilisation(G_t: float, X: float, Vmx: float, Km0: float, Vm0: float) -> float:
    Vm = Vm0 + Vmx * X
    return Vm * G_t / (Km0 + G_t)


class patientBasalState:
    def __init__(self, Vmx, Km0, BW, Vm0, k1, k2, kp1, kp2, kp3, p2u, ke1, ke2,
                 G_t_basal, U_id_basal, EGP_basal):
        self.Vmx, self.Km0, self.BW, self.Vm0 = Vmx, Km0, BW, Vm0
        self.k1, self.k2 = k1, k2
        self.kp1, self.kp2, self.kp3 = kp1, kp2, kp3
        self.p2u, self.ke1, self.ke2 = p2u, ke1, ke2
        self.G_t_basal, self.U_id_basal, self.EGP_basal = G_t_basal, U_id_basal, EGP_basal


def setup_patient(Vmx, Km0, BW, Vm0=7.2283, k1=0.0588, k2=0.2132,
                   kp1=4.6576, kp2=0.0033, kp3=0.0083,
                   p2u=0.047, ke1=0.0007, ke2=269.0) -> patientBasalState:
    def basal_eq(G_t):
        U_id = insulin_dependent_utilisation(G_t, X_basal, Vmx, Km0, Vm0)
        return -U_id + k1 * G_p_basal - k2 * G_t

    soln = root_scalar(basal_eq, bracket=[0.0, 1000.0], method="brentq")
    if not soln.converged:
        raise RuntimeError(f"Could not solve basal state for Vmx={Vmx}, Km0={Km0}")

    G_t_basal = soln.root
    U_id_basal = insulin_dependent_utilisation(G_t_basal, X_basal, Vmx, Km0, Vm0)
    EGP_basal = F_cns + k1 * G_p_basal - k2 * G_t_basal

    return patientBasalState(Vmx, Km0, BW, Vm0, k1, k2, kp1, kp2, kp3, p2u, ke1, ke2,
                              G_t_basal, U_id_basal, EGP_basal)


def derivatives(min_time, state, insulin_function, glucose_appearance_function, patient):
    G_p, G_t, I_prime, X_L, X = state
    I = insulin_umL_to_pmolL(insulin_function(min_time))
    R_a = glucose_appearance_function(min_time)

    dI_prime_dt = -ki * (I_prime - I)
    dX_L_dt = -ki * (X_L - I_prime)
    dX_dt = -patient.p2u * X + patient.p2u * (I - I_basal)

    U_id = insulin_dependent_utilisation(G_t, X, patient.Vmx, patient.Km0, patient.Vm0)
    EGP = max(patient.EGP_basal - patient.kp2 * (G_p - G_p_basal) - patient.kp3 * (X_L - X_basal_L), 0.0)
    E = patient.ke1 * (G_p - patient.ke2) if G_p > patient.ke2 else 0.0

    dG_p_dt = EGP + R_a - F_cns - E - patient.k1 * G_p + patient.k2 * G_t
    dG_t_dt = -U_id + patient.k1 * G_p - patient.k2 * G_t

    return np.array([dG_p_dt, dG_t_dt, dI_prime_dt, dX_L_dt, dX_dt], dtype=float)