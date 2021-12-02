import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



pressure = 50 #atm
d2_mass = 2.014 * 2 #u, times 2 because there are 2 deuteriums
u_to_kg = 1.660540199E-27
radius = 0.17971422120457106 #mm
temperature = 293
def get_volume(radius):
    return(4/3) * np.pi * radius**3
def get_pressure(n, T, V, R=8.31446261815324):
    return (n*T*R) / V
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

grouped_data["volume (um^3)"] = get_volume(grouped_data["Diameter of capsule (um)"]/2)
initial_density = density_without_atm
gamma = 5/3
initial_volume_0 = get_volume(189) # +- 39 microns
print ("volume: "+str(initial_volume_0))
initial_pressure_0 = 50
grouped_data["delta_density"] = grouped_data["volume (um^3)"] / initial_volume_0
grouped_data["density (kg / m^3)"] = initial_density * grouped_data["delta_density"]
grouped_data["delta_pressure"] = (
    (
        (gamma + 1) * grouped_data["delta_density"] - (gamma - 1)
    )
    /
    (
        (gamma + 1) - (gamma - 1) * grouped_data["delta_density"]
    )
)
grouped_data["pressure"] = grouped_data["delta_pressure"] * initial_pressure_0

# error bars
grouped_data["diameter error pctg"] = grouped_data["Error on the diameter of the capsule (um)"] / grouped_data["Diameter of capsule (um)"]
grouped_data["volume errors"] = grouped_data["diameter error pctg"] * grouped_data["volume (um^3)"]
grouped_data["pressure errors"] = grouped_data["diameter error pctg"] * grouped_data["pressure"]












# plt.plot(grouped_data["Time delay (ns)"], grouped_data["Diameter of capsule (um)"], label="Diameter of capsule (um)")
# plt.plot(grouped_data["Time delay (ns)"], grouped_data["density (um^-3)"], label="density (um^-3)")
# plt.legend(loc="best")
# plt.title("Grouped")
# plt.xlabel("Time delay (ns)")
# plt.ylabel("Diameter of capsule (um)")
# plt.show()

print (grouped_data["volume errors"])
print (initial_density)

# plt.plot(grouped_data["Time delay (ns)"], grouped_data["delta_density"], label="density ratio")
# plt.plot(grouped_data["Time delay (ns)"], grouped_data["delta_pressure"], label="pressure ratio")
# plt.legend(loc="best")
# plt.title("These ratios are with respect to the initial values")
# plt.xlabel("Time delay (ns)")
# plt.ylabel("ratio")
# plt.show()


plt.plot(grouped_data["Time delay (ns)"], grouped_data["pressure"], label="pressure (atm)")
plt.plot(grouped_data["Time delay (ns)"], grouped_data["density (kg / m^3)"], label="density (kg / m^3)")
plt.errorbar(grouped_data["Time delay (ns)"], grouped_data["pressure"], yerr=grouped_data["pressure errors"], fmt='o')
plt.legend(loc="best")
plt.xlabel("Time delay (ns)")
#plt.ylabel("ratio")
plt.show()