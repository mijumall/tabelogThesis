import numpy as np
import pandas as pd

from regression import Model

df = pd.read_csv("tabelog_without_outlier.csv")
Y = df["p2-p0"]
X = df["if_new"]

m = Model(Y, X)
m.regression()

print()
print("-------------------------")
print()

Y2 = df["p1-p0"]
X2 = df["if_new"]

m2 = Model(Y2, X2)
m2.regression()
