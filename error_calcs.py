import numpy as np
from run_tests import run_scenario

meals_24h = [(60.0, 45.0, 15.0), (390.0, 70.0, 20.0), (750.0, 80.0, 25.0), (960.0, 20.0, 10.0)]
target = 6.1  # glucose target = 6.1 mmol/L

# ---------------------------------------------------------------
# Part 1: Worked example -- hourly sample table for Test 1
# ---------------------------------------------------------------
r1 = run_scenario('t1', 720.0, meal_events=[], closed_loop=True)
g1 = r1['glucose']

print("=== Test 1: Hourly Sample Table (worked example) ===")
print("time_h | G(t) mmol/L | error | error^2")
for hour in range(0, 13):
    idx = hour * 60
    G_t = g1[idx]
    error = G_t - target
    sq_error = error**2
    print(f"{hour:6d} | {G_t:11.4f} | {error:7.4f} | {sq_error:.4f}")

# ---------------------------------------------------------------
# Part 2: Full MSE/RMSE table -- all scenarios
# ---------------------------------------------------------------
print()
print("=== Full MSE/RMSE Table (all scenarios) ===")
print(f"{'Scenario':<28} | {'MSE':<10} | {'RMSE':<10}")
print("-" * 55)

def report(name, glucose_array):
    mse = np.mean((glucose_array - target) ** 2)
    rmse = np.sqrt(mse)
    print(f"{name:<28} | {mse:<10.4f} | {rmse:<10.4f}")
    return mse, rmse

# Test 1 (already computed above)
report("Test 1: Fasting", g1)

# Test 2: 3-Meal Day
r2 = run_scenario('t2', 1440.0, meal_events=meals_24h, closed_loop=True)
report("Test 2: 3-Meal Day", r2['glucose'])

# Test 3: Exercise
r3 = run_scenario('t3', 1440.0, meal_events=meals_24h, exercise_event=(540.0, 45.0, 1.8), closed_loop=True)
report("Test 3: Exercise", r3['glucose'])

# Test 4: Uncontrolled vs Closed-loop
r4a = run_scenario('t4a', 1440.0, meal_events=meals_24h, closed_loop=False)
report("Test 4A: Uncontrolled", r4a['glucose'])
report("Test 4B: Closed-Loop", r2['glucose'])  # same scenario as Test 2

# Test 5: Patient Variability (corrected, independent patient parameters)
for name in ['adult_low_sens', 'adult_normal', 'adult_high_sens']:
    r5 = run_scenario(f't5_{name}', 1440.0, meal_events=meals_24h, closed_loop=True, patient_name=name)
    report(f"Test 5: {name}", r5['glucose'])