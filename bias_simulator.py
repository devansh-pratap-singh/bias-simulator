import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Objective Function for OLS
def obj_func(params, y, x):
    beta = params[:2]
    sigma = params[2]
    y_hat = x @ beta
    residual = y - y_hat
    obj = np.sum(residual ** 2) / (2 * (sigma ** 2))
    obj += len(y) / 2 * np.log(2 * np.pi * (sigma ** 2))
    return obj

#Set Parameters
alpha = 2
rho = 0.5
sims = 1000
rho_hat = np.empty(sims)
np.random.seed(123)
n_values = np.arange(100, 1100, 100)
bias_values = np.zeros(len(n_values))

#Simulating for different values of n
for k in range(len(n_values)):
    n = n_values[k]
    bias = np.zeros(sims)
    #Simulating data and model estimation for each simulation
    for i in range(sims):
        #Generate Data
        et = np.random.normal(size = n)
        xt = np.empty(n)
        xt[0] = alpha + et[0] #initial value of x
        for j in range(1, n):
            xt[j] = alpha + (rho * xt[j-1]) + et[j]
        #Estimate model using OLS
        x = np.vstack((np.ones(n - 1), xt[:-1])).T
        y = xt[1:]
        init_params = np.zeros(3)
        init_params[2] = 0.1 #set initial value of sigma to small positive number to avoid "divide by zero" error
        res = minimize(obj_func, init_params, args = (y, x))
        beta_hat = res.x[:2]
        rho_hat[i] = beta_hat[1]

        #Computing bias for rho
        bias[i] = np.mean(rho_hat) - rho
    bias_values[k] = np.mean(bias)
print(bias_values)
#We can see that we have a negative bias but the bias decrease as we increase the sample size