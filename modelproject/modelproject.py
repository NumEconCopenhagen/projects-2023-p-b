
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt
import ipywidgets as widgets

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
    
def plot(a = 2, b = 2, X = 0.4):
        model = HouseholdSpecializationModelClass()
     # Update model parameters
        model.par.sigma= a
        model.par.eta = b
        model.par.b = X   

        opt = model.solve()
        wage_opt = opt.W
        labor_opt = opt.L

        list_Labor = np.linspace(0.1, 1.2, 20)
        list_Wage = (list_Labor**(1/-model.par.sigma))*(model.par.sigma-1)/model.par.sigma 

        plt.plot(list_Labor, list_Wage)
        plt.axvline(x=1, color='r', linestyle='--') # add a vertical line at L=1

        # Add a horizontal line at the wage and labor values obtained from model.solve()
        plt.axhline(y=wage_opt, color='g', linestyle='--')
        plt.text(labor_opt-0.65, wage_opt+0.05, 'W = {:.2f}, L = {:.2f}'.format(wage_opt, labor_opt), fontsize=10, color='g')

        # Add a label to the vertical line
        plt.text(0.95, 1.2, 'L$^S$', rotation=0, fontsize=10)

        # Set the axis labels and title
        plt.xlabel('Labor (L)')
        plt.ylabel('Wage (W)')
        plt.title('Wage as a Function of Labor')

        plt.show()
    
def plot_interact():
    widgets.interact(plot,
                
                 a=widgets.FloatSlider(
                     description="a", min=1, max=5, step=0.25, value=1),
                 b=widgets.FloatSlider(
                     description="b", min=1, max=5, step=0.25, value=1),
                 x0=widgets.FloatSlider(
                     description="X", min=1, max=50, step=0.5, value=20),
                 c2=widgets.FloatSlider(
                     description="c2", min=0, max=5, step=0.1, value=0)

    );
