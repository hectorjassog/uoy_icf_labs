import numpy as np
import matplotlib.pyplot as plt
import icf
import pandas as pd
import scipy.interpolate as interpolate
from scipy.optimize import curve_fit

def calculate_pixels():
    """
        delta tao =  temporal resolution
        v = sweep speed = 63 ps/mm
        d = width of the slit in front of the photocathode = 0.25 mm
        M = instrument magnification = 1.24
        delta s = spatial resolution = 150 micron meter
    """
    d = 0.25
    M = 1.24
    delta_s = 0.15 # changed to mm
    v = 63

    delta_tao = (np.sqrt((d*M)**2 + delta_s**2))/(1/v)
    # print (delta_tao)

    pixels_to_mm = 0.06


    number_of_pixels = 1 / (v * pixels_to_mm / delta_tao)
    # print (number_of_pixels)

def plot_the_data(X, Y, title):
    plt.plot(X, Y)
    plt.title(title)
    plt.show()
    return True

original_xdata, original_ydata = icf.load_2col("../images/data/lineout_2.csv")
#xdata = reversed_arr = arr[::-1]
#plot_the_data(original_xdata, original_ydata, "original")
rearranged_xdata = original_xdata[::-1]



energy_transitions = pd.read_csv("../images/data/energy_transitions.csv")
energy_transitions["difference_between_past_point"] = energy_transitions["ev"].diff()
# print (energy_transitions.head(13))
energy_transitions = energy_transitions[energy_transitions["x_value"].notna()]
energy_transitions = energy_transitions.drop(["difference_between_past_point"], axis=1)
# print (energy_transitions[["ev", "x_value"]])

peakEnergy = energy_transitions["ev"]
peakCentre = energy_transitions["x_value"]



# f_fit = interpolate.interp1d(peakCentre, peakEnergy, kind="linear")
# energy_fitted = f_fit(peakCentre)

def f(x, *params):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]

    return a + x*b + c*x**2 + d*x**3
guess = [0, 0, 0, 0]
popt_dispersion, pcov_dispersion = curve_fit(f, peakCentre, peakEnergy, p0=guess)
energy_fitted = f(rearranged_xdata, *popt_dispersion)

# plt.plot(peakCentre, peakEnergy, 'o', label="points")
# plt.plot(rearranged_xdata, energy_fitted, label="fit")
# plt.legend(loc="best")
# plt.xlabel("peackCentre")
# plt.ylabel("peackEnrgy")
# plt.show()

# plt.plot(energy_fitted,original_ydata)
# for peak in peakEnergy:
#     plt.axvline(peak, color='r')
# plt.plot()
# plt.xlabel("energy")
# plt.show()
# y_data_shifted = f(rearranged_xdata, *popt)

# plt.plot(rearranged_xdata, original_ydata, label='Data')
# plt.plot(rearranged_xdata, y_data_shifted, label="Interpolation")
# plt.show()
# print ("min: {}, max: {}".format(np.min(energy_fitted), np.max(energy_fitted)))

files_and_interpolations = pd.DataFrame(
    {
        "file":["photocathode.dat", "reflectivity.dat", "xray1318.dat"],
        "function":["C(E)", "R(E)", "T(E)"]
    })

energy_spectrum_data = pd.DataFrame({"Energy(eV)": energy_fitted, "spectrum": original_ydata})
energy_spectrum_data = energy_spectrum_data[energy_spectrum_data["Energy(eV)"] >= 3034]

for idx, row in files_and_interpolations.iterrows():
    file = row["file"]
    # print (file)
    correction_data_x, correction_data_y = icf.load_2col("../images/corrections/"+file)
    interpolation_func = interpolate.interp1d(correction_data_x, correction_data_y, kind="linear")
    corrected_spectrum = interpolation_func(energy_spectrum_data["Energy(eV)"])
    energy_spectrum_data[row["function"]] = corrected_spectrum

#     plt.plot(energy_spectrum_data["Energy(eV)"], corrected_spectrum, label=row["function"])

# print (energy_spectrum_data)
# plt.legend(loc="best")
# plt.show()

energy_spectrum_data["corrections_combined"] = energy_spectrum_data["C(E)"] * energy_spectrum_data["R(E)"] * energy_spectrum_data["T(E)"]

energy_spectrum_data["corrected_wavelength"] = energy_spectrum_data["spectrum"] / energy_spectrum_data["corrections_combined"]

# plt.plot(energy_spectrum_data["Energy(eV)"], energy_spectrum_data["spectrum"], label="original spectrum")
# plt.plot(energy_spectrum_data["Energy(eV)"], energy_spectrum_data["corrected_wavelength"], label="corrected spectrum")
# plt.legend(loc="best")
# plt.xlabel("Energy(eV)")
# #plt.ylabel("")
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(energy_spectrum_data["Energy(eV)"], energy_spectrum_data["spectrum"], label="original spectrum")
# ax.legend(loc="best")
# ax.set_xlabel("Energy(eV)")
# ax2=ax.twinx()
# ax2.plot(energy_spectrum_data["Energy(eV)"], energy_spectrum_data["corrected_wavelength"], label="corrected spectrum", color="r")
# ax2.legend(loc="best")
# plt.show()

def gaussian(x, *params):

    A = params[0]
    x0 = params[1]
    c = params[2]

    return A*np.exp(-(x-x0)**2 / (2*c**2))


def multiple_gaussians(x, *params):
    y0 = params[9]
    G1 = gaussian(x, *params[0:3])
    G2 = gaussian(x, *params[3:6])
    G3 = gaussian(x, *params[6:9])

    return y0 + G1 + G2 + G3


guesses = [4300, 3685, 80,
    1800, 3875, 50,
    3000, 3940, 60, 
    500
]


He_B = 3683.7
Ly_B = 3935.6
He_alpha = 3139.3
# plt.plot(energy_spectrum_data["Energy(eV)"], energy_spectrum_data["corrected_wavelength"], label="corrected spectrum")
# plt.axvline(He_B, color='r', label="He-B")
# plt.axvline(Ly_B, color='r', label="Ly-B")
# plt.legend(loc="best")
# plt.xlabel("Energy(eV)")
# plt.show()

energy_spectrum_data = energy_spectrum_data[energy_spectrum_data["Energy(eV)"] >= 3500]
energy_spectrum_data = energy_spectrum_data[energy_spectrum_data["Energy(eV)"] <= 4100]
ranges = [
    [0, 3400, 1, 0, 3400, 1, 0, 3400, 1, 0],
    [np.inf, 4500, 200, np.inf, 4500, 200, np.inf, 4500, 200, 2000]
]
popt, pcov = curve_fit(multiple_gaussians, energy_spectrum_data["Energy(eV)"], energy_spectrum_data["corrected_wavelength"], guesses, bounds=ranges)
fit_data = multiple_gaussians(energy_spectrum_data["Energy(eV)"], *popt)
# print (popt, pcov)
plt.plot(energy_spectrum_data["Energy(eV)"], energy_spectrum_data["corrected_wavelength"], label="corrected spectrum")
plt.plot(energy_spectrum_data["Energy(eV)"], fit_data, label="fit")
plt.axvline(He_B, color='r', label="He-B")
plt.axvline(Ly_B, color='r', label="Ly-B")
# plt.axvline(He_alpha, color='r', label="He-alpha")
plt.legend(loc="best")
plt.xlabel("Energy(eV)")
plt.show()

def get_delta_x(c, n=2):
    return np.sqrt(2) * c * (np.log(2))**(1/n)

def R(source_s=100, ssc=150, film=1, dig=60):
    return np.sqrt(source_s**2 + ssc**2 + film**2 + dig**2)

def get_fwhm(C, C_cov):
    print (C, C_cov)
    C_error = np.sqrt(C_cov)
    delta_x = get_delta_x(C)
    C_error_prct = C_error / C
    delta_x_error = delta_x * C_error_prct
    diameter = delta_x * 2
    diameter_error = delta_x_error * 2
    # need to add more parameters... psf?
    print ("R" + str(R()))
    R_2_value = (R() * (200/6000))**2
    print ("R2: " + str(R_2_value))
    core_diameter = np.sqrt((diameter**2) - (R_2_value))    
    core_diameter_error = np.sqrt((diameter_error**2))
    print (core_diameter_error)
    return core_diameter, core_diameter_error
print ("i")
fwhm_df = pd.DataFrame({
    "C": [popt[2], popt[5], popt[8]],
    "C_cov": [pcov[2][2], pcov[5][5], pcov[8][8]],
})

fwhm_df["S_FWHM"], fwhm_df["S_FWHM_error"] = get_fwhm(fwhm_df["C"], fwhm_df["C_cov"])

print (fwhm_df)
for idx, row in fwhm_df.iterrows():
    
    print ("The value of FWHM is: {} +- {}".format(row["S_FWHM"], np.sqrt(row["S_FWHM"])))
