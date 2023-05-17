
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import warnings
import pandas as pd 
import matplotlib.pyplot as plt
import ipywidgets as widgets
import seaborn as sns


class UnionModel:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters
        par.eta = 1
        par.sigma = 2
        par.b = 0.5

        #For extension
        par.beta = 1

        #Solution
        sol.W = np.zeros(1)
        sol.L =np.zeros(1)

   
    def calc_union_utility(self,W):
        """ calculate utility """

        par = self.par


        # a. Optimal price for firm
        P = par.sigma/(par.sigma-1)*W

        # b. Optimal production for firm
        Y = min(P**(-par.sigma),1)

        # c. Labor demand by firm
        L = Y

        # Store optimal labor value in solution namespace
        self.sol.L[0] = L
        
        # Return union utility
        return (W-par.b)*L**par.eta

    def extension(self,W):
        """ extension """
        par = self.par

        # a. Optimal price
        P = par.sigma/(par.sigma-1)*W

        # b. Optimal production
        Y = min(P**(-par.sigma),1)

        # c. Labor demand
        L = Y        

        #Define profit as function of wage:
        profit_w = P**-par.sigma*(P-W)
        
        #Define union utility as function of wage:
        union_w = (W-par.b)*L**par.eta

        # store optimal labor value in solution namespace
        self.sol.L[0] = L        

        #Return nash product
        return union_w**par.beta*profit_w**(1-par.beta)

    def solve(self,do_print=False, extension=False):
        """ Solve model with and without extension """
        if extension:
            obj = lambda x: - self.extension(x[0]) #Objective function is nash product for extension
        else:
        #Objective function set to minus utility
            obj = lambda x: - self.calc_union_utility(x[0]) #Objective function is union utilty if no extension 
        #Bounds for choice variables  
        bounds = [(0.0001,np.inf)]
        #Initial guess for the optimizer
        guess = [10]
        #Minimizing the objective function (maximize utility)
        result = optimize.minimize(obj, guess, method='Nelder-Mead',bounds=bounds)
        opt = SimpleNamespace()

        opt.W = result.x[0]
        opt.L = self.sol.L[0]

        #Return optimal wage and labor employment
        return opt
    
    def plot(self, a = 2, b = 2, c = 0.4, d=0.5, extension=False):
     # Update model parameters
        par = self.par

        #Set model parameters
        par.sigma= a
        par.eta = b
        par.b = c
        par.beta = d  

        #Create the demand for labor
        list_Labor = np.linspace(0.01, 1.2, 30)
        list_Wage = (list_Labor**(1/-par.sigma))*(par.sigma-1)/par.sigma

        #Solve the model with or without extension
        self.solve(extension=extension)
        opt = self.solve(extension=extension)
        #Store results
        wage_opt = opt.W
        labor_opt = opt.L

        #Plot demand curve
        plt.plot(list_Labor, list_Wage, label='Demand for labor')
        plt.axvline(x=1, color='r', linestyle='--', label='Labor supply') # add a vertical line at L=1

        # Add a horizontal line at the wage value in equlibrium
        plt.axhline(y=wage_opt, color='g', linestyle='--', label='Wage set by union')
        plt.text(0.5, 1.2, 'W = {:.2f}, L = {:.2f}'.format(wage_opt, labor_opt), fontsize=10, color='g')

        # Set the axis labels and title
        plt.xlabel('Labor (L)')
        plt.ylabel('Wage (W)')
        plt.title('Union model equlibrium')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        plt.show()
    
    def plot_interact(self):
        warnings.filterwarnings("ignore")
        #Make the above plot interactive
        widgets.interact(self.plot,
            
            a=widgets.FloatSlider(
                description="sigma", min=1.2, max=5, step=0.1, value=2),
            b=widgets.FloatSlider(
                 description="eta", min=0.8, max=5, step=0.1, value=1),
            c=widgets.FloatSlider(
                description="b", min=0.25, max=1, step=0.05, value=0.5),
            d=widgets.FloatSlider(
                description="beta", min=0, max=1, step=0.01, value=0.5), 


   ) ;
