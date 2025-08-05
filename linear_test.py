# Import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in California housing dataset.
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df['MedHouseVal'] = housing.target

print(housing_df.head())

print("================================================")

# Make histograms bigger
plt.figure(figsize=(16, 12))
housing_df.hist(figsize=(16, 12), bins=30)
plt.suptitle('California Housing Dataset - Histograms', fontsize=16, y=0.95)
plt.tight_layout()
plt.show()

# Make density plots bigger
plt.figure(figsize=(16, 12))
housing_df.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(16, 12))
plt.suptitle('California Housing Dataset - Density Plots', fontsize=16, y=0.95)
plt.tight_layout()
plt.show()