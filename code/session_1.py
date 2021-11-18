import icf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def gaussian(x, *params):

    A = params[0]
    x0 = params[1]
    c = params[2]
    y0 = params[3]

    return y0 + A*np.exp(-(x-x0)**2 / (2*c**2))

def super_gaussian(x, *params):
    if n == 0:
        print ("n not set")
    A = params[0]
    x0 = params[1]
    c = params[2]
    y0 = params[3]

    return y0 + A*np.exp(-((x-x0)/(np.sqrt(2)*c))**n)

n = 0
ns = [4, 6]
guess = [5000, 1, 0.5, 28000]
print ("Our initial guess is: {0}".format(guess))
file_names = ["vertical", "horizontal", "diagonal_top_left_to_bottom_right", "diagonal_top_right_to_bottom_left"]
mse_values = []
calculated_mse_values_sg = []
# matrix to store the values to be plotted
mse_for_n_file_name = np.zeros((len(ns), len(file_names)))
fig, axis = plt.subplots(len(ns), len(file_names))
for idxn, n_value in enumerate(ns):



    n = n_value

    mse_values_sg = []


    for idxf, file_name in enumerate(file_names):
        if 1==1:    
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

            # the MSE for the gaussian doesn't really change with respect to N, but it is easier to have it in the forloop. 
            # we will just add the same values several times and get the mean, so it won't affect the result.
            mse = np.mean((ydata - yfit)**2)
            mse_values.append(mse)
            mse_sg = np.mean((ydata - yfit_sg)**2)
            mse_values_sg.append(mse_sg)


            print ("For super gaussian with n {0} and for file {1} is: {2}".format(n, file_name, mse_sg))
            mse_for_n_file_name[idxn][idxf] = mse_sg

            # plt.plot(n_value, mse_values_sg[0], label=str(file_name))
            # print (file_name)
            # print (mse_values_sg[0])

            axis[idxn, idxf].plot(xdata, ydata, label="data")
            axis[idxn, idxf].plot(xdata, yfit_sg, label="super gaussian")
            axis[idxn, idxf].legend(loc="best")
            axis[idxn, idxf].set_title(file_name)
            axis[idxn, idxf].set_ylabel("MSE for n={0}".format(n_value))

            # plt.plot(xdata, ydata, label="data")
            # plt.plot(xdata, yfit_sg, label="super gaussian")
            # plt.legend(loc="best")
            # plt.show()
        else: 
            pass
    n_mse_sg = np.mean(mse_values_sg)
    print ("For the super gaussian with n = {0}, the average MSE = {1}".format(n_value, n_mse_sg))
    calculated_mse_values_sg.append(n_mse_sg)

print ("For the gaussian, the MSE = {1}".format(n_value, np.mean(mse_values)))
plt.show()

# plt.plot(ns, mse_for_n_file_name)
# plt.xlabel("Values of n")
# plt.ylabel("Mean Squared Error (MSE)")
# plt.show()
# icf.fit_plot(xdata, ydata, yfit)

