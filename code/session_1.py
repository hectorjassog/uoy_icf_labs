import icf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def gaussian(x, *params):
    print (params)

    A = params[0]
    x0 = params[1]
    c = params[2]
    y0 = params[3]

    return y0 + A*np.exp(-(x-x0)**2 / (2*c*c))

file_names = ["vertical", "horizontal", "diagonal_top_left_to_bottom_right", "diagonal_top_right_to_bottom_left"]

for file_name in file_names:
    # xdata, ydata = icf.load_2col("../data/session1/{0}.csv".format(file_name))
    xdata, ydata = icf.load_2col("../data/session1/{0}.csv".format(file_name))

    xdata *= 60.0/1000.0


    if file_name == "diagonal_top_left_to_bottom_right":
        # xdata = xdata[:]
        pass

    # plt.plot(xdata, ydata)
    # plt.title("Lineout: {}".format(file_name))
    # plt.xlabel("Position - mm")
    # plt.ylabel("Brightness units")
    # plt.show()




    guess = [5000, 1, 1, 28000]

    print ("Our initial guess is: {0}".format(guess))


    popt, pcov = curve_fit(gaussian, xdata, ydata, p0=guess)

    for i in range(len(popt)):
        print ("Parameter {0}: {1} +/- {2}".format(i, popt[i], np.sqrt(pcov[i][i])))

    print ("Fit parameters : {0}".format(popt))
    print ("Fit standard deviations : {0}".format(np.sqrt(np.diag(pcov))))

    yfit = gaussian(xdata, *popt)

    icf.fit_plot(xdata, ydata, yfit)

