import numpy as np
from scipy.integrate import solve_ivp

#Using the intraperiotoneal insulin delivery model 
#from url: https://pmc.ncbi.nlm.nih.gov/articles/PMC8465342/?utm_source=chatgpt.com#B13-metabolites-11-00600
#from Schiavon et al.

#The following are parameters from the above paper:

k_a1 = 0.010 #first IP compartment
k_a2 = 0.028 #second IP compartment 

v_i = 3.4 #volume of insulin distribution
m_1 = 0.15 #constant rate of insulin distribution from liver -> plasma
m_2 = 0.268 #volume of insulin distribution 

cl = 1.16 #post-hepatic insulin clearance
He_b = 0.59 #basal hepatic insulin extraction
a_i = 16e-5 #control of hepatic insulin

#Parameter that requires derivation: m_4

m_4 = (cl/v_i) - (He_b * m_2) #rate of insulin distribution from liver -> plasma

#Set up the initial conditions:

basal_insulin_p = 21.1

q_basal_p = basal_insulin_p * v_i
q_basal_l = ((m_2 + m_4)/m_1) * q_basal_p


def hepatic_insulin_extraction(q_l : float) -> float:
    #determines the value for the hepatic insulin extraction, HE(t):

    H_e = -a_i * (q_l - q_basal_l) + He_b
    return H_e

def liver_deg_rate(q_l : float) -> float:
    #determines the value for m3 = liver degradation rate
    H_e = hepatic_insulin_extraction(q_l)
    return (H_e/(1.0 - H_e)) * m_1


#Calculate basal steady state values:

m3_basal = liver_deg_rate(q_basal_l)

r_basal = (((m_1 + m3_basal) * q_basal_l - m_2 * q_basal_p))

q_ip1_basal = r_basal / (k_a1 + k_a2)
q_ip2_basal = q_ip1_basal

print(f"Basal liver degradation rate: {m3_basal:.6f}")
print(f"Basal insulin delivery rate: {r_basal:.6f} mU/min")
print(f"Basal IP compartment 1: {q_ip1_basal:.6f} mU")
print(f"Basal IP compartment 2: {q_ip2_basal:.6f} mU")

#Return basal insulin infusion:

def basal_insulin_infusion(min_time : float) -> float:
    return r_basal #needed to maintain basal state 


#Insulin model derivatives:

def derivatives(min_time : float, 
                compartment : np.ndarray, 
                infusion_func,) -> np.ndarray:

    Q_ip1, Q_ip2, q_l, q_p = compartment

    #insulin entering first IP compartment
    infusion_rate = infusion_func(min_time)

    #IP insulin absorption model 
    dq_ip1_dt = (-(k_a1 + k_a2) * Q_ip1 + infusion_rate)

    dq_ip2_dt = (-k_a2 * Q_ip2 + k_a2 * Q_ip1)
    
    #Insulin: IP -> Liver
    R_ai = (k_a1 * Q_ip1 + k_a2 * Q_ip2)

    #liver degradation:
    m_3 = liver_deg_rate(q_l)

    #liver + plasma insulin

    dq_l_dt = (-(m_1 + m_3) * q_l + m_2 * q_p + R_ai)
    dq_p_dt = (-(m_2 + m_4) * q_p + m_1 * q_l)

    return np.array([dq_ip1_dt, dq_ip2_dt, dq_l_dt, dq_p_dt], 
                    dtype = float,)



def perform_insulin_sim(
        compartment_infusion_function, 
        min_duration: float, 
        output_interval_min: float = 1.0,):

    initial = np.array([q_ip1_basal, q_ip2_basal, q_basal_l, q_basal_p,],
        dtype= float,)

    time_points = np.arange(
        0.0,
        min_duration + output_interval_min,
        output_interval_min,
    )

    soln = solve_ivp(
        fun = lambda min_time, compartment: derivatives( min_time = min_time, compartment = compartment, infusion_func = compartment_infusion_function,),
        t_span = (0.0, min_duration), 
        y0 = initial,
        t_eval = time_points, 
        method = "LSODA",
        rtol = 1e-7,
        atol = 1e-9,
        max_step = 1.0,
    )

    if not soln.success:
        raise RuntimeError(f"Sim failed: {soln.message}")
    return soln


def plasma_insulin_concentration(
    plasma_mass: np.ndarray,
) -> np.ndarray:
    
    #Convert plasma insulin mass (mU)
    #to plasma insulin concentration (mU/L).

    return plasma_mass / v_i




