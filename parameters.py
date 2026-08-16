from dataclasses import dataclass


@dataclass(frozen=True)
class PatientParameters:

    #Rate constants taken from Paper: "Intraperitoneal Insulin
    #Delivery: Evidence of a Physiological Route for Artificial 
    #Pancreas From Compartmental Modeling" by Lo Presti et al. at url: 
    #https://pmc.ncbi.nlm.nih.gov/articles/PMC10210098/

    #all k constants are in min^-1 units

    k_PL = 0.008 #1/min  Rate of insulin transfer from Liver to Plasma
    k_LP = 0.322 #1/min  Rate of insulin transfer from Plasma to Liver

    k_0P = 0.119 #1/min  Rate of insulin clearance from Plasma
    k_0L = 0.008 #1/min  Rate of insulin clearance from Liver

    Vi = 0.045 #in L/kg; plasma distribution volume of insulin

@dataclass(frozen=True)
class IPinsulinParameters:

    pass