
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

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5

        #We add a parameter for the model extension in 5.

        par.theta = 0

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = np.fmin(HM,HF)
        elif par.sigma == 1:
            H = (HM+1e-10)**(1-par.alpha)*(HF+1e-10)**par.alpha
        #To avoid errors, 0.000000001 is added in multiple places in the code below
        else:
            H = ((1-par.alpha)*(HM+0.000000001)**((par.sigma-1)/(par.sigma))+(par.alpha)*(HF+0.000000001)**((par.sigma-1)/(par.sigma)))**((par.sigma)/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*((TM)**epsilon_/epsilon_+(TF-HF*par.theta)**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    
    def solve(self,do_print=False):
        """ solve model continously """
        #Objective function set to minus utility
        obj = lambda x: - self.calc_utility(x[0], x[1], x[2], x[3])  
        #Bounds for choice variables  
        bounds = [(0,24)]*4
        #Initial guess for the optimizer
        guess = [4]*4
        #Minimizing the objective function (maximize utility)
        result = optimize.minimize(obj, guess, method='Nelder-Mead',bounds=bounds)
        opt = SimpleNamespace()
        #Storing optimal choice variables
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]

        return opt

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol
        #Looping through female wage vector and solve model with continous solver for each entry
        for n, i in enumerate(par.wF_vec) :
            par.wF = i
            out = self.solve()
            #Storing results
            sol.LM_vec[n] = out.LM
            sol.LF_vec[n] = out.LF
            sol.HM_vec[n] = out.HM
            sol.HF_vec[n] = out.HF
        

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/(sol.HM_vec+1e-10))
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        #Define objective function to be minimized
        def objective(x, self):
            par = self.par
            sol=self.sol
            #Set variables
            par.alpha = x[0]
            par.sigma = x[1]
            #Call the solver for vector of female wages
            self.solve_wF_vec()
            #Run regression with results from the above solver
            self.run_regression()
            #Return objective to minimized (squared deviation from data)
            return (0.4-sol.beta0)**2+(-0.1-sol.beta1)**2
        #Set guess for alpha and sigma
        guess = [.5]*2
        #Bound alpha to (0,1) and sigma to (0,10)
        bounds = [(0,1), (0,10)]
        #Minimize objective function
        result = optimize.minimize(objective, guess, args = (self), method = 'Nelder-Mead', bounds=bounds)
    
    #The following function is identical to the function above, except it only optimize with respect to sigma (alpha fixed)
    def estimatev2(self,sigma=None):
        """ estimate alpha and sigma """
        def objective(x, self):
            par = self.par
            sol=self.sol
            par.sigma = x[0]
            self.solve_wF_vec()
            self.run_regression()
            return (0.4-sol.beta0)**2+(-0.1-sol.beta1)**2
        guess = [.1]
        bounds = [(0,10)]
        result = optimize.minimize(objective, guess, args = (self), method = 'Nelder-Mead', bounds=bounds)

    #The following function is identical to the function above, except it optimize with respect to theta and sigma (alpha still fixed)  
    def estimatev3(self,wM=None,sigma=None):
        """ estimate alpha and sigma """
        def objective(x, self):
            par = self.par
            sol=self.sol
            par.theta = x[0]
            par.sigma = x[1]
            self.solve_wF_vec()
            self.run_regression()
            return (0.4-sol.beta0)**2+(-0.1-sol.beta1)**2
        guess = [(1.5)]*2
        bounds = [(0,10)]*2
        result = optimize.minimize(objective, guess, args = (self), method = 'Nelder-Mead', bounds=bounds)