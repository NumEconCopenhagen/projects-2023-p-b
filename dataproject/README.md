# Data analysis project

Our project is titled **Tobins q - data project**. In the proejct we replicate the main analysis in our bachelor thesis "Tobin's Q - Quotable or Questionable?". Buidling on the work of Blanchard, Rhee and Summers (1993) and looking at the U.S non-financial sector, we investigate whether mispricing in the financial markets has important implications for the real economy. We find that the elasticity of investment with respect to market valuation is more than 5 times higher when changes in market valuation reflect changed fundamentals, compared to when they do not.

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).

We apply the **following datasets**:

1. Data from FRED using an API 
1. BRS data.xlsx. Previously available from: https://economics.mit.edu/faculty/blanchar/papers. Link has experied since we did our BA. We only use the data for comparission and our analysis does not depend on it.

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install matplotlib-venn``

The notebook leans on code provided in dataproject.py.