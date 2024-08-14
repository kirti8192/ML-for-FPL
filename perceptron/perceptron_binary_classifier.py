import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

# read data from csv 
raw_data = pd.read_csv('datasets/mushroom_cleaned.csv')

# split dataframe into training data and classification
train_data = raw_data.iloc[:,:-1]
label_data = raw_data.iloc[:,-1]

# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(train_data, c=label_data, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)
plt.show()