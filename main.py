import matplotlib.pyplot as plt

from insulin_new import (
    basal_insulin_p,
    q_basal_l,
    q_basal_p,
    r_basal,
    perform_insulin_sim,
    basal_insulin_infusion,
    plasma_insulin_concentration,
)


def main():

    print(
        f"Basal plasma insulin concentration: "
        f"{basal_insulin_p:.2f} mU/L"
    )

    print(
        f"Initial plasma insulin mass: "
        f"{q_basal_p:.2f} mU"
    )

    print(
        f"Initial liver insulin mass: "
        f"{q_basal_l:.2f} mU"
    )

    #test for increased insulin infusion
    def increased_insulin_infusion(min_time: float) -> float:

        if 100 <= min_time < 200:
            return 2.0 * r_basal

        return r_basal

    #test for reduced insulin infusion
    def reduced_insulin_infusion(min_time: float) -> float:

         if 100 <= min_time < 200:
              return 0.5 * r_basal

         return r_basal

    #test for no insulin infusion
    def no_insulin_infusion(min_time: float) -> float:

         if 100 <= min_time < 200:
              return 0.0

         return r_basal

    print(f"Infusion at 0 min: {no_insulin_infusion(0):.6f} mU/min")
    print(f"Infusion at 100 min: {no_insulin_infusion(100):.6f} mU/min")
    print(f"Infusion at 150 min: {no_insulin_infusion(150):.6f} mU/min")
    print(f"Infusion at 200 min: {no_insulin_infusion(200):.6f} mU/min")

    #Perform simulation:

    soln = perform_insulin_sim(
        compartment_infusion_function = no_insulin_infusion,
        min_duration = 600.0,
        output_interval_min = 1.0,
    )


    #Get results:
    time = soln.t

    q_ip1 = soln.y[0]
    q_ip2 = soln.y[1]
    q_liver = soln.y[2]
    plasma_mass = soln.y[3]

    print()
    print("IP compartment 1:")
    print(f"0 min:   {q_ip1[0]:.4f} mU")
    print(f"100 min: {q_ip1[100]:.4f} mU")
    print(f"150 min: {q_ip1[150]:.4f} mU")
    print(f"200 min: {q_ip1[200]:.4f} mU")
    print(f"300 min: {q_ip1[300]:.4f} mU")

    print()
    print("IP compartment 2:")
    print(f"0 min:   {q_ip2[0]:.4f} mU")
    print(f"100 min: {q_ip2[100]:.4f} mU")
    print(f"150 min: {q_ip2[150]:.4f} mU")
    print(f"200 min: {q_ip2[200]:.4f} mU")
    print(f"300 min: {q_ip2[300]:.4f} mU")


    # Convert plasma insulin mass to concentration

    plasma_concentration = plasma_insulin_concentration(
        soln.y[3]
    )


    print()
    print(f"Plasma insulin at 0 min: {plasma_concentration[0]:.4f} mU/L")
    print(f"Plasma insulin at 100 min: {plasma_concentration[100]:.4f} mU/L")
    print(f"Plasma insulin at 150 min: {plasma_concentration[150]:.4f} mU/L")
    print(f"Plasma insulin at 200 min: {plasma_concentration[200]:.4f} mU/L")
    print(f"Plasma insulin at 300 min: {plasma_concentration[300]:.4f} mU/L")
    print(f"Plasma insulin at 600 min: {plasma_concentration[600]:.4f} mU/L")


    #Output results

    print()

    print(
        f"Initial plasma insulin: "
        f"{plasma_concentration[0]:.12f} mU/L"
    )

    print(
        f"Final plasma insulin: "
        f"{plasma_concentration[-1]:.12f} mU/L"
    )

    print(
        f"Initial liver insulin: "
        f"{q_liver[0]:.12f} mU"
    )

    print(
        f"Final liver insulin: "
        f"{q_liver[-1]:.12f} mU"
    )


    #Plot insulin compartments

    plt.figure(figsize=(9, 5))

    plt.plot(
        time,
        q_ip1,
        label="IP compartment 1",
    )

    plt.plot(
        time,
        q_ip2,
        label="IP compartment 2",
    )

    plt.plot(
        time,
        q_liver,
        label="Liver insulin",
    )

    plt.plot(
        time,
        plasma_mass,
        label="Plasma insulin",
    )

    plt.xlabel("Time (min)")
    plt.ylabel("Insulin mass (mU)")
    plt.title("Intraperitoneal Insulin Model")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


    #Plot [plasma insulin]

    plt.figure(figsize=(9, 5))

    plt.plot(
        time,
        plasma_concentration,
        label="Plasma insulin concentration",
    )

    plt.xlabel("Time (min)")
    plt.ylabel("Plasma insulin (mU/L)")
    plt.title("Plasma Insulin Concentration")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    main()