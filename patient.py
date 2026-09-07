#Patient parameters are taken from: Jinyu Xie. Simglucose v0.2.1 (2018) [Online]. Available: https://github.com/jxx123/simglucose. Accessed on: 08-26-2026

#Vg = volume distr. of glucose
#Vmx = peripheral insulin action use
#km0 = in mg/kg
#k2 = glucose kinetics constant
#kp1 = for 0 glucose and insulin- estimated EGP

patient_types = {
    "original": {
        "BW": 79.79, "Vmx": 0.1276, "Km0": 218.8779, "r_basal": 59.6976,
        "Vm0": 7.2283, "k1": 0.0588, "k2": 0.2132,
        "kp1": 4.6576, "kp2": 0.0033, "kp3": 0.0083,
        "p2u": 0.047, "ke1": 0.0007, "ke2": 269.0,
    },
    "adult_low_sens": {   # adult#009
        "BW": 74.604, "Vmx": 0.016839, "Km0": 246.44, "r_basal": 18.8969,
        "Vm0": 4.743293, "k1": 0.044387, "k2": 0.145980,
        "kp1": 4.336893, "kp2": 0.004180, "kp3": 0.009886,
        "p2u": 0.024128, "ke1": 0.0005, "ke2": 339.0,
    },
    "adult_normal": {     # adult#006 -- true population median (was adult#003, rank 8/10)
    "BW": 66.097, "Vmx": 0.045288, "Km0": 188.77, "r_basal": 28.7422,
    "Vm0": 2.118484, "k1": 0.073784, "k2": 0.070249,
    "kp1": 3.562132, "kp2": 0.002659, "kp3": 0.004659,
    "p2u": 0.053212, "ke1": 0.0005, "ke2": 339.0,
},
    "adult_high_sens": {  # adult#004
        "BW": 63.000, "Vmx": 0.108626, "Km0": 246.881887, "r_basal": 14.7972,
        "Vm0": 4.366596, "k1": 0.093573, "k2": 0.119406,
        "kp1": 4.335970, "kp2": 0.003088, "kp3": 0.005634,
        "p2u": 0.042194, "ke1": 0.0005, "ke2": 339.0,
    },
}

def get_patient(name):
    return patient_types[name]