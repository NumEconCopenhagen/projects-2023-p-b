
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
        par.beta = 1

        #Solution
        sol.W = np.zeros(1)
        sol.L =np.zeros(1)

   
    def calc_union_utility(self,W):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. Optimal price
        P = par.sigma/(par.sigma-1)*W

        # b. Optimal production
        Y = min(P**(-par.sigma),1)

        # c. Labor demand
        L = Y

        # store optimal labor value in solution namespace
        self.sol.L[0] = L

        return (W-par.b)*L**par.eta

    def solve(self,do_print=False, extension=False):
        """ Solve model """
        if extension:
            obj = lambda x: - self.extension(x[0])
        else:
        #Objective function set to minus utility
            obj = lambda x: - self.calc_union_utility(x[0])  
        #Bounds for choice variables  
        bounds = [(0.01,np.inf)]
        #Initial guess for the optimizer
        guess = [10]
        #Minimizing the objective function (maximize utility)
        result = optimize.minimize(obj, guess, method='Nelder-Mead',bounds=bounds)
        opt = SimpleNamespace()

        opt.W = result.x[0]
        opt.L = self.sol.L[0]


        return opt
    
    def extension(self,W):

        #Define profit as function of wage:
        par = self.par

        # a. Optimal price
        P = par.sigma/(par.sigma-1)*W

        # b. Optimal production
        Y = min(P**(-par.sigma),1)

        # c. Labor demand
        L = Y        

        profit_w = P**-par.sigma*(P-W)
        
        union_w = (W-par.b)*min(L,1)**par.eta

        # store optimal labor value in solution namespace
        self.sol.L[0] = L        

        return union_w**par.beta*profit_w**(1-par.beta)
    
    def solve_extension(self, do_print=False):
        #Objective function set to minus utility
        obj = lambda x: - self.extension(x[0])  
        #Bounds for choice variables  
        bounds = [(0.001,np.inf)]
        #Initial guess for the optimizer
        guess = [10]
        #Minimizing the objective function (maximize utility)
        result = optimize.minimize(obj, guess, method='Nelder-Mead',bounds=bounds)
        opt = SimpleNamespace()

        opt.W = result.x[0]
        opt.L = self.sol.L[0]

        return opt
