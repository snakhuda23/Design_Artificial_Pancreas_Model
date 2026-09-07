#Cgm sensors do not report blood glucose without errors
#Dexcom ONE+ reports an adult MARD of 8.2% (url = https://dexcom.com.au/hcp-dexcom-one-plus)

import numpy as np

mard_cgm = 0.082

random_noise = np.random.default_rng(seed=42) #allows for same noise sequence to occur 

def add_noise_from_cgm(true_glucose_mg_dl: float) -> float:

    noise_std = true_glucose_mg_dl * mard_cgm
    noisy_reading = true_glucose_mg_dl + random_noise.normal(0.0, noise_std)
    return max(noisy_reading, 1.0)  # keep it physically positive