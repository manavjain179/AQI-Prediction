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

#multivariate analysis - to compare various features with each other
sns.pairplot(dataframe)

dataframe.corr()

# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (21,21))
sns.heatmap(dataframe.corr(), annot = True, cmap = "RdYlGn")

plt.show()

## function to store error values

def store_errors(model_name, mae, mse, rmse):
    List_errors = []
    List_errors.append(model_name)
    List_errors.append(mae)
    List_errors.append(mse)
    List_errors.append(rmse)
    return List_errors

"""###Linear Regression"""

from sklearn.model_selection import train_test_split
Xtrain, Xtst, ytrain, ytst = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

model1 = LinearRegression()
model1.fit(Xtrain,ytrain)

model1.coef_

model1.intercept_

model1.score(Xtrain, ytrain)

model1.score(Xtst, ytst)

coeff_df = pd.DataFrame(model1.coef_,X.columns,columns=['Coefficient'])
coeff_df

model1_prediction=model1.predict(Xtst)

sns.displot(ytst-model1_prediction)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(ytst, model1_prediction))
print('MSE:', metrics.mean_squared_error(ytst, model1_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model1_prediction)))

model_errors = []
mae = metrics.mean_absolute_error(ytst, model1_prediction)
mse = metrics.mean_squared_error(ytst, model1_prediction)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model1_prediction))

model_errors.append(store_errors('Linear Regression', mae, mse, rmse))

model_errors

"""###Decision Tree"""

from sklearn.tree import DecisionTreeRegressor
model2=DecisionTreeRegressor()
model2.fit(Xtrain,ytrain)

model2.score(Xtrain, ytrain)

model2.score(Xtst, ytst)

model2_prediction=model2.predict(Xtst)

sns.displot(ytst-model2_prediction)

plt.scatter(ytst,model2_prediction)

DecisionTreeRegressor()

parameters={
 "splitter"    : ["best","random"] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_samples_leaf" : [ 1,2,3,4,5 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
 "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]
}

from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(model2,param_grid=parameters,scoring='neg_mean_squared_error',n_jobs=-1,cv=10,verbose=3)
grid_search.fit(X,y)

grid_search.best_params_

grid_search.best_score_

model2_prediction_2 =  grid_search.predict(Xtst)

sns.displot(ytst-model2_prediction_2)

mae = metrics.mean_absolute_error(ytst, model2_prediction_2)
mse = metrics.mean_squared_error(ytst, model2_prediction_2)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model2_prediction_2))
model_errors.append(store_errors('Decision Tree', mae, mse, rmse))

print('MAE:', metrics.mean_absolute_error(ytst, model2_prediction_2))
print('MSE:', metrics.mean_squared_error(ytst, model2_prediction_2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model2_prediction_2)))

"""###Random Forest"""

from sklearn.ensemble import RandomForestRegressor
model3 = RandomForestRegressor()
model3.fit(Xtrain, ytrain)

model3.score(Xtrain, ytrain)

model3.score(Xtst, ytst)

model3_prediction = model3.predict(Xtst)

sns.displot(ytst-model3_prediction)

plt.scatter(ytst,model3_prediction)

RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]

rf_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

model3_2 = RandomForestRegressor()

random_search = RandomizedSearchCV(estimator = model3_2, param_distributions = rf_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=3, random_state=None)

random_search.fit(Xtrain,ytrain)

random_search.best_params_

random_search.best_score_

model3_prediction_2 = random_search.predict(Xtst)

sns.displot(ytst-model3_prediction_2)

plt.scatter(ytst,model3_prediction_2)

mae = metrics.mean_absolute_error(ytst, model3_prediction_2)
mse = metrics.mean_squared_error(ytst, model3_prediction_2)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model3_prediction_2))
model_errors.append(store_errors('Random Forest', mae, mse, rmse))

print('MAE:', metrics.mean_absolute_error(ytst, model3_prediction_2))
print('MSE:', metrics.mean_squared_error(ytst, model3_prediction_2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model3_prediction_2)))



"""### XGBoost Boosting"""

import xgboost as xg_bst

model4 = xg_bst.XGBRegressor()
model4.fit(Xtrain,ytrain)

model4.score(Xtrain,ytrain)

model4.score(Xtst,ytst)

from sklearn.model_selection import cross_val_score
results = cross_val_score(model4,X,y,cv=6)

results.mean()

"""### Model Evaluation"""

model4_prediction = model4.predict(Xtst)

sns.displot(ytst-model4_prediction)

plt.scatter(ytst, model4_prediction)

### HyperParamter Tunning
xg_bst.XGBRegressor()

from sklearn.model_selection import RandomizedSearchCV
no_of_tree = [int(x) for x in np.linspace(start = 100, stop = 1100, num=20)]
print(no_of_tree)

# No of decision trees
no_of_tree = [int(x) for x in np.linspace(start = 100, stop = 1100, num=20)]
# Learning Rate
Learning_Rate = ['0.05','0.1','0.15','0.3','0.6','0.65']
# Max Level of a tree
max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
#Subssample parameter values
sampling=[0.7,0.6,0.8]
# Minimum child weight parameters
weight_minimum=[3,4,5,6,7]

# Create the random grid
grid_parameters = {'n_estimators': no_of_tree,
               'learning_rate': Learning_Rate,
               'max_depth': max_depth,
               'subsample': sampling,
               'min_child_weight': weight_minimum }

print(grid_parameters)

model4_prediction1= xg_bst.XGBRegressor()
model4_prediction2 = RandomizedSearchCV(estimator = model4_prediction1, param_distributions = grid_parameters, scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)

model4_prediction2.fit(Xtrain,ytrain)

model4_prediction2.best_params_

model4_prediction2.best_score_

model4_predictions3=model4_prediction2.predict(Xtst)

sns.displot(ytst-model4_predictions3)

plt.scatter(ytst,model4_predictions3)

from sklearn import metrics

mae = metrics.mean_absolute_error(ytst, model4_predictions3)
mse = metrics.mean_squared_error(ytst, model4_predictions3)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model4_predictions3))
model_errors.append(store_errors('XGBoost', mae, mse, rmse))

print('MAE:', metrics.mean_absolute_error(ytst, model4_predictions3))
print('MSE:', metrics.mean_squared_error(ytst, model4_predictions3))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model4_predictions3)))



"""### Artificial Neural Networks"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

## since problem is a linear regression output should be linear activation and 2 hidden layer to train arbitrary functions

model5 = Sequential()

# Input Layer
model5.add(Dense(128, kernel_initializer='normal',input_dim = Xtrain.shape[1], activation='relu'))

# 2 Hidden Layers with relu activation
model5.add(Dense(256, kernel_initializer='normal',activation='relu'))
model5.add(Dense(256, kernel_initializer='normal',activation='relu'))

# Output Layer
model5.add(Dense(1, kernel_initializer='normal',activation='linear'))


model5.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model5.summary()

# Training Set
model5_prediction = model5.fit(Xtrain, ytrain, validation_split=0.2, batch_size = 10, epochs=100 )

model5_prediction2 = model5.predict(Xtst)

sns.displot(ytst.values.reshape(-1,1)-model5_prediction2)

plt.scatter(ytst,model5_prediction2)

mae = metrics.mean_absolute_error(ytst,model5_prediction2)
mse = metrics.mean_squared_error(ytst,model5_prediction2)
rmse = np.sqrt(metrics.mean_squared_error(ytst,model5_prediction2))
model_errors.append(store_errors('ANN', mae, mse, rmse))

print('MAE:', metrics.mean_absolute_error(ytst,model5_prediction2))
print('MSE:', metrics.mean_squared_error(ytst,model5_prediction2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst,model5_prediction2)))



#model_errors

model_names = [row[0] for row in model_errors]
mae_values = [row[1] for row in model_errors]
rmse_values = [row[3] for row in model_errors]

fig, axis = plt.subplots()
axis.plot(model_names, mae_values, color='red', marker='o', label='MAE')
axis.plot(model_names, rmse_values, color='blue', marker='^', label='RMSE')

axis.set_title('Error Values for Models')
axis.set_xlabel(' - Model Name - ')
axis.set_ylabel('Error Value')
axis.legend()
axis.grid(True)
plt.show()

model_names = [row[0] for row in model_errors]
mse_values = [row[2] for row in model_errors]

fig, axis = plt.subplots()
axis.plot(model_names, mse_values, color='green', marker='s', label='MSE')

axis.set_title('MSE Values for Models')
axis.set_xlabel('- Model Name -')
axis.set_ylabel('Error Value')
axis.legend()
axis.grid(True)
plt.show()





