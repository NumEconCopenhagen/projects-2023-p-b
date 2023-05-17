#Set imports:
import numpy as np
import pandas as pd
import pandas_datareader #API used to retrieve data form FRED
import statsmodels.api as sm # used for linear regression
import ipywidgets as widgets
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})


def interactive(merged):

    #Plot variables for comparison:
    def _plot_timeseries(dataframe, variable, years, BRS_data= True):
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1,1,1)
        
        dataframe.loc[:,['Years']] = pd.to_numeric(dataframe['Years'])
        I = (dataframe['Years'] >= years[0]) & (dataframe['Years'] <= years[1])
        
        x = dataframe.loc[I,'Years']
        if BRS_data == True:
            variable = [i.__contains__(variable) for i in dataframe.columns.to_list()]
        y = dataframe.loc[I,variable]
        n_col = len(y.shape)
        if n_col  !=1:
            color = ['orange', '#1f77b4']
            for i in range(n_col):
                ax.plot(x,y[y.columns[i]], color = color[i])
        else:
            color = '#1f77b4'
            ax.plot(x,y, color = color)
            
        
        ax.set_xticks(list(range(years[0], years[1] + 1, 10)))   
        plt.title('Comparing variables')
        plt.show()


    def plot_timeseries(dataframe):
        widgets.interact(_plot_timeseries, 
        dataframe = widgets.fixed(dataframe),
        variable = widgets.Dropdown(
            description='variable', 
            options=['I/K','P/K','Q'], 
            value='I/K'),
        years=widgets.IntRangeSlider(
            description="years",
            min=1954,
            max=2020,
            value=[1954, 2020],
            continuous_update=False,
        opt = widgets.Checkbox(
            value=False,
            description= 'BRS_data',
            disabled=False,
            indent=False)    
        )           
            
    ); 

    plot_timeseries(merged)