import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model():
    def __init__(self, Y, X):
        """Multiple regression.

        :param Y: Dependent (explained) variable
        :type Y: class "pandas.core.series.Series" or "pandas.core.frame.DataFrame"
        
        :param X: Independent (explaining) variable(s). 
        :type X: class "pandas.core.series.Series" or "pandas.core.frame.DataFrame"
        """
        self._init_normal_distribution()
        self.N = len(Y)
        keys = pd.Series(range(0, self.N))
        
        # Reassign new index in case Y or X are coming from slices, which causes "matrices not aligned error"
        self.Y = pd.Series(pd.DataFrame(Y).set_index(keys=keys).iloc[:,0])
        self.X = pd.DataFrame(X).set_index(keys=keys)

    def regression(self, showCorrelation=True):
        """Compute and print multiple regression results.
        
        :param showCorrelation: Set False if you don't need to check multicollinearity, defaults to True.
        :type showCorrelation: bool
        """
        
        print("Regression starts... \n")
        
        Y = self.Y
        X = self.X
        k = len(self.X.columns) # Number of independent variables
        N = self.N
        
        # Add constant vector and rearrange X's order to compute coefficient.
        X["_constant_"] = 1
        cols=list(X.columns)
        cols = [cols[-1]] + cols[:-1]
        X = X[cols]
        self.X = X 

        # Compute multiple regression.
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
        
        # Compute and generate Y_hat as: pandas.core.series.Series
        Y_hat = {}
        for coef, column in zip(coefficients, X.columns):
            Y_hat[column] = coef * X[column]
        Y_hat = pd.DataFrame(Y_hat)
        Y_hat = Y_hat.T.sum()
        
        # Compute resudual and Adjusted R squared.
        residual = Y - Y_hat
        R2 = 1 - ((N-1) / (N-k-1)) * ((residual**2).sum() / ((Y - Y.mean())**2).sum())
        
        # Compute V_hat
        V_hat = (np.linalg.inv((1/N * X.T @ X))) @ \
                    (1/(N-k-1) * X.T @ np.diag(residual**2) @ X) @ \
                    (np.linalg.inv((1/N) * X.T @ X))
        # V_hat shapes 3 x 3 symmetric matrix. To get V for each coefficient, extract diagonal matrix:
        V_hat = np.diag(V_hat)
        
        # Standard error of coefficients
        stdErr = ((1/N) * V_hat) ** (1/2)
        
        # t-value
        t_values = coefficients / stdErr
        
        # p-value
        n = len(self.pdf)
        p_values = []
        for t_value in t_values:
            p_value = 0.0
            for idx in range(n//2, -1, -1):
                if abs(t_value) < abs(self.sndx[idx]):
                    p_value = self.cdf[idx-1] * 2
                    break
            p_values.append(p_value)
        
        # Clearly visualize the data
        df = pd.DataFrame(
            {
                "Name": X.columns,
                "Coef": coefficients, 
                "Std Err": stdErr, 
                "t-value": t_values, 
                "p-value": p_values, 
            }
        )
        
        print(f"Explained variable: {self.Y.name}\n")
        print(f"Adjusted R-squared: {round(R2, 4)}\n")
        print("Two-tailed t-test results:\n")
        print(df, '\n')
        
        if showCorrelation:
            print("\nCorrelation between independent variables:\n")
            print(X.drop(columns="_constant_").corr())
        
    
    def _init_normal_distribution(self):
        """Compute the following values and make them be available as class instances.
        This method is called in __init__ function.
        
        self.sndx   # Stands for: Standard Normal Distribution's x
        self.cdf    # Stands for: Cummulative Density Function
        self.pdf    # Stands for: Probability Density Function
        """
        n = 5000
        mu = 0
        sigma = 1
        
        x = np.linspace(-5, 5, n)
        
        pdf = (np.exp((-(x - mu) ** 2) / 2 * (sigma ** 2))) / (sigma * (1/2 ** (2 * np.pi)))
        
        cdf = []
        for idx in np.arange(n):
            cdf.append((pdf[:idx] / pdf.sum()).sum())
        cdf = np.array(cdf)

        self.sndx = x 
        self.cdf = cdf
        self.pdf = pdf
        
    def visualizeDistributions(self):
        n = len(self.pdf)
        with plt.style.context("dark_background"):
            plt.figure(figsize=(24, 6))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            
            def plot(title, place, x, y):
                plt.subplot(place)
                plt.title(title, fontsize=30)
                plt.plot(x, y)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.grid(alpha=0.5)
                
            plot("SND", 121, self.sndx, self.pdf / n )
            plot("CDF", 122, self.sndx, self.cdf)
            
            plt.show()

