import matplotlib.pyplot as plt
import numpy as np

from parameters import PatientParameters
from insulin import perform_insulin_sim


def test_for_insulin_delivery(min_time):

    if min_time < 30.0:
        return 0.0

    elif min_time < 60.0:
        return 1.0

    elif min_time < 90.0:
        return 0.5

    else:
        return 0.0


def main():

    # Create the patient parameters
    parameters = PatientParameters()

    # Initial conditions:
    # [liver insulin, plasma insulin]
    initial = np.array(
        [0.0, 0.0],
        dtype=float,
    )

    print("Delivery at 0 min:", test_for_insulin_delivery(0))
    print("Delivery at 30 min:", test_for_insulin_delivery(30))
    print("Delivery at 60 min:", test_for_insulin_delivery(60))
    print("Delivery at 90 min:", test_for_insulin_delivery(90))

    # Run the Model 4 insulin simulation
    soln = perform_insulin_sim(
        parameters=parameters,
        initial_state=initial,
        insulin_delivery_func=test_for_insulin_delivery,
        min_duration=120.0,
        output_interval_min=1.0,
    )

    # Extract results
    time = soln.t

    liver_insulin = soln.y[0]
    plasma_insulin = soln.y[1]

    # Calculate the insulin delivery at every simulated time
    delivery = np.array(
        [
            test_for_insulin_delivery(t)
            for t in time
        ]
    )

    # --------------------------------------------------
    # GRAPH 1: INSULIN RESPONSE
    # --------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        time,
        liver_insulin,
        label="Liver insulin",
    )

    plt.plot(
        time,
        plasma_insulin,
        label="Plasma insulin",
    )

    plt.xlabel("Time (min)")
    plt.ylabel("Insulin state")
    plt.title("Model 4 Intraperitoneal Insulin Response")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --------------------------------------------------
    # GRAPH 2: INSULIN DELIVERY
    # --------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.step(
        time,
        delivery,
        where="post",
    )

    plt.xlabel("Time (min)")
    plt.ylabel("IP insulin delivery")
    plt.title("Test Intraperitoneal Insulin Delivery")

    plt.grid(True)
    plt.tight_layout()

    # Display both graphs
    plt.show()

    for t in [0, 30, 60, 90, 120]:

        index = int(t)

        print(
        f"Time = {t:3d} min | "
        f"Delivery = {delivery[index]:.2f} | "
        f"Liver = {liver_insulin[index]:.4f} | "
        f"Plasma = {plasma_insulin[index]:.4f}"
    )


if __name__ == "__main__":
    main()