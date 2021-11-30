import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



pressure = 50 #atm
d2_mass = 2.014 * 2 #u, times 2 because there are 2 deuteriums
u_to_kg = 1.660540199E-27
radius = 0.17971422120457106 #mm
temperature = 293
volume = (3/4) * np.pi * radius**3
boltzmann = 1.38064852e-23
density = (pressure * d2_mass * u_to_kg) / ((boltzmann) * (temperature))
density_without_atm = density * 101325



diameter_data = pd.read_csv("../data/capsuledata.csv")
diameter_data.sort_values(by=["Time delay (ns)"], inplace=True)

print (diameter_data)

# plt.plot(diameter_data["Time delay (ns)"], diameter_data["Diameter of capsule (um)"])
# plt.title("Ungrouped")
# plt.xlabel("Time delay (ns)")
# plt.ylabel("Diameter of capsule (um)")
# plt.show()

grouped_data = diameter_data.groupby(["Time delay (ns)"]).mean()

print (grouped_data)

grouped_data.to_csv("../data/grouped_capsuledata.csv")
grouped_data = pd.read_csv("../data/grouped_capsuledata.csv")

grouped_data["volume (um^3)"] = (4/3) * np.pi * (grouped_data["Diameter of capsule (um)"]/2)**3
grouped_data["density (um^-3)"] = 100/grouped_data["volume (um^3)"] # assuming 1 as the mass, since it will stay constant
initial_density = density_without_atm
gamma = 5/3
grouped_data["delta_density"] = grouped_data["density (um^-3)"]/initial_density
grouped_data["delta_pressure"] = (
    (
        (gamma + 1) * grouped_data["delta_density"] - (gamma - 1)
    )
    /
    (
        (gamma + 1) - (gamma - 1) * grouped_data["delta_density"]
    )
)

# plt.plot(grouped_data["Time delay (ns)"], grouped_data["Diameter of capsule (um)"], label="Diameter of capsule (um)")
# plt.plot(grouped_data["Time delay (ns)"], grouped_data["density (um^-3)"], label="density (um^-3)")
# plt.legend(loc="best")
# plt.title("Grouped")
# plt.xlabel("Time delay (ns)")
# plt.ylabel("Diameter of capsule (um)")
# plt.show()

print (grouped_data)

plt.plot(grouped_data["Time delay (ns)"], grouped_data["density (um^-3)"], label="density (um^-3)")
plt.plot(grouped_data["Time delay (ns)"], grouped_data["delta_pressure"], label="delta_pressure")
plt.legend(loc="best")
# plt.title("Grouped")
plt.xlabel("Time delay (ns)")
plt.ylabel("density (um^-3)")
plt.show()
