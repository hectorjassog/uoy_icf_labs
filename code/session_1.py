import icf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def gaussian(x, *params):

    A = params[0]
    x0 = params[1]
    c = params[2]
    y0 = params[3]

    return y0 + A*np.exp(-(x-x0)**2 / (2*c*c))

def super_gaussian(x, *params):
    if n == 0:
        print ("n not set")
    A = params[0]
    x0 = params[1]
    c = params[2]
    y0 = params[3]

    return y0 + A*np.exp(-((x-x0)/(np.sqrt(2)*c))**n)

n = 0
ns = [2, 4, 6, 8]
guess = [5000, 1, 0.5, 28000]
print ("Our initial guess is: {0}".format(guess))
file_names = ["vertical", "horizontal", "diagonal_top_left_to_bottom_right", "diagonal_top_right_to_bottom_left"]
for n_value in ns:
    n = n_value

    mse_values = []
    mse_values_sg = []


    for file_name in file_names:
        if file_name == "vertical":
            # xdata, ydata = icf.load_2col("../data/session1/{0}.csv".format(file_name))
            xdata, ydata = icf.load_2col("../data/session1/{0}.csv".format(file_name))

            xdata *= 60.0/1000.0

            # plt.plot(xdata, ydata)
            # plt.title("Lineout: {}".format(file_name))
            # plt.xlabel("Position - mm")
            # plt.ylabel("Brightness units")
            # plt.show()


            popt, pcov = curve_fit(gaussian, xdata, ydata, p0=guess)
            popt_sg, pcov_sg = curve_fit(super_gaussian, xdata, ydata, p0=guess)

            # for i in range(len(popt)):
            #     print ("Parameter {0}: {1} +/- {2}".format(i, popt[i], np.sqrt(pcov[i][i])))

            # print ("Fit parameters gaussian: {0}".format(popt))
            # print ("Fit standard deviations gaussian: {0}".format(np.sqrt(np.diag(pcov))))

            # for i in range(len(popt_sg)):
            #     print ("Parameter {0}: {1} +/- {2}".format(i, popt_sg[i], np.sqrt(pcov_sg[i][i])))

            # print ("Fit parameters super gaussian: {0}".format(popt_sg))
            # print ("Fit standard deviations super gaussian: {0}".format(np.sqrt(np.diag(pcov_sg))))

            yfit = gaussian(xdata, *popt)
            yfit_sg = super_gaussian(xdata, *popt_sg)

            mse = np.mean((ydata - yfit)**2)
            mse_values.append(mse)
            mse_sg = np.mean((ydata - yfit_sg)**2)
            mse_values_sg.append(mse_sg)


            # print ("MSE for super gaussian {} with n {}")

            plt.plot(xdata, ydata, label="data")
            plt.plot(xdata, yfit, label="gaussian")
            plt.plot(xdata, yfit_sg, label="super gaussian")
            plt.legend(loc="best")
            plt.show()
        else: 
            pass
    
    print ("For the super gaussian with n = {0}, the MSE = {1}".format(n_value, np.mean(mse_values_sg)))
    # break
    # icf.fit_plot(xdata, ydata, yfit)

