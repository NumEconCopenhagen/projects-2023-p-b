
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters
        par.eta = 1
        par.sigma = 2
        par.b = 0.5

        #Solution
        sol.W = np.zeros(1)
        sol.L =np.zeros(1)

   
    def calc_union_utility(self,W):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. Optimal price
        P = par.sigma/(par.sigma-1)*W

        # a. Optimal production
        L = min((par.sigma/(par.sigma-1)*W)**-par.sigma,1)

        # a. Labor demand
        

        # store optimal labor value in solution namespace
        self.sol.L[0] = L

        return (W-par.b)*L**par.eta

    def solve(self,do_print=False):
        """ solve model continously """
        #Objective function set to minus utility
        obj = lambda x: - self.calc_union_utility(x[0])  
        #Bounds for choice variables  
        bounds = [(0,100)]
        #Initial guess for the optimizer
        guess = [10]
        #Minimizing the objective function (maximize utility)
        result = optimize.minimize(obj, guess, method='Nelder-Mead',bounds=bounds)
        opt = SimpleNamespace()

        opt.W = result.x[0]
        opt.L = self.sol.L[0]


        return opt