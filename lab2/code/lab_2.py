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
    print (delta_tao)

    pixels_to_mm = 0.06


    number_of_pixels = 1 / (v * pixels_to_mm / delta_tao)
    print (number_of_pixels)

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
print (energy_transitions.head(13))
energy_transitions = energy_transitions[energy_transitions["x_value"].notna()]
energy_transitions = energy_transitions.drop(["difference_between_past_point"], axis=1)
print (energy_transitions[["ev", "x_value"]])

peakEnergy = energy_transitions["ev"]
peakCentre = energy_transitions["x_value"]

plt.plot(peakCentre, peakEnergy, 'o', label="points")

# f_fit = interpolate.interp1d(peakCentre, peakEnergy, kind="linear")
# energy_fitted = f_fit(peakCentre)

def f(x, *params):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]

    return a + x*b + c*x**2 + d*x**3
guess = [0, 0, 0, 0]
popt, pcov = curve_fit(f, peakCentre, peakEnergy, p0=guess)
energy_fitted = f(rearranged_xdata, *popt)

plt.plot(rearranged_xdata, energy_fitted, label="fit")
plt.legend(loc="best")
plt.xlabel("peackCentre")
plt.ylabel("peackEnrgy")
plt.show()

plt.plot(energy_fitted,original_ydata)
for peak in peakEnergy:
    plt.axvline(peak, color='r')
plt.plot()
plt.xlabel("energy")
plt.show()
# y_data_shifted = f(rearranged_xdata, *popt)

# plt.plot(rearranged_xdata, original_ydata, label='Data')
# plt.plot(rearranged_xdata, y_data_shifted, label="Interpolation")
# plt.show()
print ("min: {}, max: {}".format(np.min(energy_fitted), np.max(energy_fitted)))







# df[df['EPS'].notna()]

# plot_the_data(rearranged_xdata, original_ydata, "rearranged")


# peaks = find_peaks(original_ydata, height = 1, threshold = 1, distance = 1)
# peaks_pos = rearranged_xdata[peaks[0]] 
# height = peaks[1]['peak_heights']
# print (peaks)
# plt.plot(peaks_pos, height, marker = 'D', label="peaks")
# plt.plot(rearranged_xdata, original_ydata, label="data")
# plt.legend(loc="best")
# plt.show()