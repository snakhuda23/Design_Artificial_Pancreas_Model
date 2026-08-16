from parameters import PatientParameters
import numpy as np
from scipy.integrate import solve_ivp

#Insulin equations taken from Paper: "Intraperitoneal Insulin
#Delivery: Evidence of a Physiological Route for Artificial 
#Pancreas From Compartmental Modeling" by Lo Presti et al. at url: 
#https://pmc.ncbi.nlm.nih.gov/articles/PMC10210098/

#For the initial virtual patient, we are assuming the worst case scenario
#for a type 1 diabetic patient, i.e. they do not produce any endogenous
#insulin and therefore, basal insulin concentrations are set to zero

#I_L = liver insulin
#I_P = plasma insulin

def insulin_derivatives( #Paper's model 4 equations:
        min_time, compartment, parameters, R_aIP, 
):
    I_L, I_P = compartment #where insulin gets delivered

    #calculate the derivative of liver insulin from paper:
    dI_L_dt = -(parameters.k_PL + parameters.k_0L)*I_L + (parameters.k_LP*I_P) + R_aIP

    #calculate the derivative of pancreas insulin from paper:
    dI_P_dt = ((parameters.k_PL * I_L) - ((parameters.k_0P + parameters.k_LP) * I_P))

    return np.array([dI_L_dt, dI_P_dt], dtype = float,)

    test_state = np.array([0.0, 0.0])

    test_derivatives = insulin_derivatives(
        min_time=30.0,
        compartment=test_state,
        parameters=parameters,
        R_aIP=test_for_insulin_delivery(30.0),
    )

    print("Derivatives at 30 min:", test_derivatives)



def perform_insulin_sim( #Solve model 4's equations over time
        parameters, initial_state, insulin_delivery_func, min_duration, output_interval_min = 1.0,
):
    time_points = np.arange(0.0, min_duration + output_interval_min, output_interval_min)

    soln = solve_ivp(fun = lambda time, compartment: insulin_derivatives(min_time = time, compartment = compartment, parameters = parameters, R_aIP = insulin_delivery_func(time),),

    t_span = (0.0, min_duration), y0 = initial_state, t_eval = time_points, method = "LSODA",  max_step=output_interval_min,)

    if not soln.success:
        raise RuntimeError(
            f"Insulin simulation failed: {soln.message}"
        )

    return soln


def test_for_insulin_delivery(min_time): #Creates a controlled test for the insulin pump
    return 1.0

def basal_insulin_delivery(min_time: float) -> float:
    #the basal insulin delivery from the pump
    basal_rate = 0.1
    return basal_rate