import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv('data/final-data/finalize_dataCombine.csv')
dataframe.head()

dataframe.dropna(inplace = True)
dataframe.head()

dataframe.info()

dataframe.describe()

#no null values as we already handled it during data collection
sns.heatmap(dataframe.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# independent features
X=dataframe.iloc[:,:-1]
# dependent features
y=dataframe.iloc[:,-1]


print(X)
print(y)