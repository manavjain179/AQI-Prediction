import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv('data/final-data/finalize_dataCombine.csv')
dataframe.head()
dataframe.dropna(inplace = True)
dataframe.head()

## since T and TM are highly correlated we remove 'TM' and evaluate the model

dataframe.drop(columns =['TM'],inplace=True)

# independent features
X=dataframe.iloc[:,:-1]
# dependent features
y=dataframe.iloc[:,-1]

X.head()

## function to store error values

def store_errors(model_name, mae, mse, rmse):
    List_errors = []
    List_errors.append(model_name)
    List_errors.append(mae)
    List_errors.append(mse)
    List_errors.append(rmse)
    return List_errors

model_errors = []

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

mae = metrics.mean_absolute_error(ytst, model1_prediction)
mse = metrics.mean_squared_error(ytst, model1_prediction)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model1_prediction))
model_errors.append(store_errors('Linear Regression 1', mae, mse, rmse))

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

print('MAE:', metrics.mean_absolute_error(ytst, model2_prediction_2))
print('MSE:', metrics.mean_squared_error(ytst, model2_prediction_2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model2_prediction_2)))

mae = metrics.mean_absolute_error(ytst, model2_prediction_2)
mse = metrics.mean_squared_error(ytst, model2_prediction_2)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model2_prediction_2))
model_errors.append(store_errors('Decision Tree 1', mae, mse, rmse))

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

print('MAE:', metrics.mean_absolute_error(ytst, model3_prediction_2))
print('MSE:', metrics.mean_squared_error(ytst, model3_prediction_2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model3_prediction_2)))

mae = metrics.mean_absolute_error(ytst, model3_prediction_2)
mse = metrics.mean_squared_error(ytst, model3_prediction_2)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model3_prediction_2))
model_errors.append(store_errors('Random Forest 1', mae, mse, rmse))

import xgboost as xg_bst
from sklearn.model_selection import RandomizedSearchCV

model4 = xg_bst.XGBRegressor()
model4.fit(Xtrain,ytrain)
model4.score(Xtrain,ytrain)
model4.score(Xtst,ytst)

from sklearn.model_selection import cross_val_score
results = cross_val_score(model4,X,y,cv=6)

model4_prediction = model4.predict(Xtst)
plt.scatter(ytst, model4_prediction)

### HyperParamter Tunning
xg_bst.XGBRegressor()

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

model4_prediction1= xg_bst.XGBRegressor()
model4_prediction2 = RandomizedSearchCV(estimator = model4_prediction1, param_distributions = grid_parameters, scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)
model4_prediction2.fit(Xtrain,ytrain)

print(model4_prediction2.best_params_)
model4_predictions3=model4_prediction2.predict(Xtst)
sns.displot(ytst-model4_predictions3)
plt.scatter(ytst,model4_predictions3)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(ytst, model4_predictions3))
print('MSE:', metrics.mean_squared_error(ytst, model4_predictions3))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model4_predictions3)))

mae = metrics.mean_absolute_error(ytst, model4_predictions3)
mse = metrics.mean_squared_error(ytst, model4_predictions3)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model4_predictions3))
model_errors.append(store_errors('XGBoost 1', mae, mse, rmse))

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

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(ytst,model5_prediction2))
print('MSE:', metrics.mean_squared_error(ytst,model5_prediction2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst,model5_prediction2)))

mae = metrics.mean_absolute_error(ytst,model5_prediction2)
mse = metrics.mean_squared_error(ytst,model5_prediction2)
rmse = np.sqrt(metrics.mean_squared_error(ytst,model5_prediction2))
model_errors.append(store_errors('ANN 1 2hidden', mae, mse, rmse))

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

## since problem is a linear regression output should be linear activation and 2 hidden layer to train arbitrary functions
model5 = Sequential()

# Input Layer
model5.add(Dense(128, kernel_initializer='normal',input_dim = Xtrain.shape[1], activation='relu'))

# 3 Hidden Layers with relu activation
model5.add(Dense(256, kernel_initializer='normal',activation='relu'))
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

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(ytst,model5_prediction2))
print('MSE:', metrics.mean_squared_error(ytst,model5_prediction2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst,model5_prediction2)))
  
mae = metrics.mean_absolute_error(ytst,model5_prediction2)
mse = metrics.mean_squared_error(ytst,model5_prediction2)
rmse = np.sqrt(metrics.mean_squared_error(ytst,model5_prediction2))
model_errors.append(store_errors('ANN 1 3hidden', mae, mse, rmse))



## Removing 2nd Correlated Value

dataframe.drop(columns =['Tm'],inplace=True)

# independent features
X=dataframe.iloc[:,:-1]
# dependent features
y=dataframe.iloc[:,-1]

X.head()

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

mae = metrics.mean_absolute_error(ytst, model1_prediction)
mse = metrics.mean_squared_error(ytst, model1_prediction)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model1_prediction))
model_errors.append(store_errors('Linear Regression 2', mae, mse, rmse))

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

print('MAE:', metrics.mean_absolute_error(ytst, model2_prediction_2))
print('MSE:', metrics.mean_squared_error(ytst, model2_prediction_2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model2_prediction_2)))

mae = metrics.mean_absolute_error(ytst, model2_prediction_2)
mse = metrics.mean_squared_error(ytst, model2_prediction_2)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model2_prediction_2))
model_errors.append(store_errors('Decision Tree 2', mae, mse, rmse))

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

print('MAE:', metrics.mean_absolute_error(ytst, model3_prediction_2))
print('MSE:', metrics.mean_squared_error(ytst, model3_prediction_2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model3_prediction_2)))

mae = metrics.mean_absolute_error(ytst, model3_prediction_2)
mse = metrics.mean_squared_error(ytst, model3_prediction_2)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model3_prediction_2))
model_errors.append(store_errors('Random Forest 2', mae, mse, rmse))

import xgboost as xg_bst
model4 = xg_bst.XGBRegressor()
model4.fit(Xtrain,ytrain)
model4.score(Xtrain,ytrain)
model4.score(Xtst,ytst)
model4_prediction = model4.predict(Xtst)
sns.displot(ytst-model4_prediction)
plt.scatter(ytst, model4_prediction)
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
print('MAE:', metrics.mean_absolute_error(ytst, model4_predictions3))
print('MSE:', metrics.mean_squared_error(ytst, model4_predictions3))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst, model4_predictions3)))

mae = metrics.mean_absolute_error(ytst, model4_predictions3)
mse = metrics.mean_squared_error(ytst, model4_predictions3)
rmse = np.sqrt(metrics.mean_squared_error(ytst, model4_predictions3))
model_errors.append(store_errors('XGBoost 2', mae, mse, rmse))

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

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(ytst,model5_prediction2))
print('MSE:', metrics.mean_squared_error(ytst,model5_prediction2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(ytst,model5_prediction2)))

mae = metrics.mean_absolute_error(ytst,model5_prediction2)
mse = metrics.mean_squared_error(ytst,model5_prediction2)
rmse = np.sqrt(metrics.mean_squared_error(ytst,model5_prediction2))
model_errors.append(store_errors('ANN 2', mae, mse, rmse))





model_1_feature_removal = [['ANN 2hidden', 38.356754663139064, 3210.4131207633577, 56.66050759359077],
 ['ANN 3hidden', 40.02788254151313, 3596.3862450987494, 59.9698778146058],
 ['Linear Regression',
  41.653709462961004,
  3265.740761752423,
  57.14666011021487],
 ['Decision Tree', 48.95063560914849, 4330.974835278704, 65.81014234355297],
 ['Random Forest', 36.85554940536202, 2743.261379801006, 52.3761527777767],
 ['XGBoost', 36.08265074117614, 2625.027147885373, 51.235018765346155]]

model_2_feature_removal = [['Linear Regression',
  42.363914920503966,
  3373.560246485142,
  58.082357446001986],
 ['Decision Tree', 44.2847155112832, 3865.759890071924, 62.17523534392068],
 ['Random Forest', 36.8358922320288, 2726.061030542887, 52.21169438490659],
 ['XGBoost', 35.44559548608807, 2610.2440925692854, 51.09054797679592],
 ['ANN', 40.6934540290616, 3395.7156894613363, 58.27276970816932]]

model_errors

import matplotlib.pyplot as plt

# Get the MAE values for each model
model_names = [model[0] for model in model_errors]
mae_values = [model[1] for model in model_errors]

# Set up the plot
fig, ax = plt.subplots()

# Plot the MAE values for each model
ax.plot(range(len(mae_values)), mae_values, color='blue',marker='o',label='MAE')

# Add x-axis labels
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=90)

# Add legend and title
ax.legend()
ax.set_title('MAE values for all models')

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Extract the model names and error values from the list
model_names = [model[0] for model in model_errors]
mae_values = [model[1] for model in model_errors]
rmse_values = [model[3] for model in model_errors]

# Set up the plot
fig, ax = plt.subplots()

# Plot the MAE values
ax.plot(range(len(mae_values)), mae_values, color='green',marker='o', label='MAE')

# Plot the RMSE values
ax.plot(range(len(rmse_values)), rmse_values, color='blue',marker='^', label='RMSE')

# Add x-axis labels
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')

# Add legend and title
ax.legend()
ax.set_title('MAE and RMSE values for all models')

# Show the plot
plt.show()

# Filter out the models
models = [model[0] for model in model_errors]
mse_values = [model[2] for model in model_errors]

# Set up the plot
fig, ax = plt.subplots()

# Plot the MSE values
ax.bar(models, mse_values, color='purple')

# Add labels and title
ax.set_xlabel('Models')
ax.set_ylabel('MSE values')
ax.set_title('Comparison of Mean Squared Error (MSE) values for different models')

# Rotate the x-axis labels
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()



model_1_feature_removal

import matplotlib.pyplot as plt

# Get the MAE and RMSE values for each model
model_names = [model[0] for model in model_1_feature_removal]
mae_values = [model[1] for model in model_1_feature_removal]
rmse_values = [model[3] for model in model_1_feature_removal]

# Set up the plot
fig, ax = plt.subplots()

# Plot the MAE values
ax.plot(range(len(mae_values)), mae_values, color='purple', marker='*', label='MAE')

# Plot the RMSE values
ax.plot(range(len(rmse_values)), rmse_values, color='green', marker = 'o',label='RMSE')

# Add x-axis labels
ax.set_xticks(range(len(model_1_feature_removal)))
ax.set_xticklabels(model_names, rotation=45, ha='right')

# Add legend and title
ax.legend()
ax.set_title('MAE and RMSE values for Model after removing 1 correlated feature')

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Get the MAE and RMSE values for each model
model_names = [model[0] for model in model_2_feature_removal]
mae_values = [model[1] for model in model_2_feature_removal]
rmse_values = [model[3] for model in model_2_feature_removal]

# Set up the plot
fig, ax = plt.subplots()

# Plot the MAE values
ax.plot(range(len(mae_values)), mae_values, color='purple', marker='*', label='MAE')

# Plot the RMSE values
ax.plot(range(len(rmse_values)), rmse_values, color='green', marker = 'o',label='RMSE')

# Add x-axis labels
ax.set_xticks(range(len(model_2_feature_removal)))
ax.set_xticklabels(model_names, rotation=45, ha='right')

# Add legend and title
ax.legend()
ax.set_title('MAE and RMSE values for Model after removing 2 correlated feature')

# Show the plot
plt.show()

