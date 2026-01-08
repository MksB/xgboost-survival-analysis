## ML Classification Regression

# supervised learning:: learn to map, required labeled data
# unsupervised learning:: teach an algorithm, required unlabeled data

# Classification and regression decision tree models...
# CART :: Classification and regression trees.

# Ensemble modeling:: ml technique combins multiple models
# Bagging:: bootstrap aggreation, sampling original dataset
# Boosting:: iterative technique that focuses on sequentially improving models

# XGBoosts Quick Start Guide...
import pandas as pd
import numpy as np

from sklearn import datasets
irisarray = datasets.load_iris()

print(irisarray)

irisdata = pd.DataFrame(
    np.c_[irisarray['data'],irisarray['target']],
    columns=irisarray['feature_names']+['Species']
)
irisdata['Species']=irisdata['Species'].astype(int)

# Exploring datasets making graphs
import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(
    irisdata,x="Species",discrete=True,
    hue="Species",shrink=0.8,palette="Greys"
    )
plt.show()

fig,axes = plt.subplots(2,2,figsize=(7,7))
sns.boxplot(ax=axes[0,0],data=irisdata,x="Species",y="sepal length (cm)",
            palette="Greys",hue="Species")
sns.boxplot(ax=axes[0,1],data=irisdata,x="Species",y="petal length (cm)",
            palette="Greys",hue="Species")
sns.boxplot(ax=axes[1,1],data=irisdata,x="Species",y="sepal width (cm)",
            palette="Greys",hue="Species")
sns.boxplot(ax=axes[1,1],data=irisdata,x="Species",y="petal width (cm)",
            palette="Greys",hue="Species")

sns.set_theme(
    rc={"axes.facecolor":"efefef",
        "figure.facecolor":"efefef"})
graphxy=sns.pairplot(irisdata,
                     hue="Species",
                     palette="Greys")
graphxy.add_legend()
plt.show()

from sklearn.model_selection import train_test_split
training_data, testing_data = train_test_split(
    irisdata,test_size=0.2,random_state=17)

training_data.shape


# Setting up and training XGBoost
X_train = training_data[[
    'sepal length (cm)','sepal width (cm)',
    'petal length (cm)','petal width (cm)']]
X_train.head()

y_train = training_data[['Species']]
y_train.head()

X_test = testing_data[[
    'sepal length (cm)','sepal width (cm)',
    'petal length (cm)','petal width (cm)'
    ]]
y_test = testing_data[['Species']]

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
iris_classifier = xgb.XGBClassifier(eval_metric="auc")
iris_classifier.fit(
    X_train,y_train,
    eval_set=[(X_test,y_test),(X_train,y_train)])

y_score = iris_classifier.predict(X_test)

X_example = np.array([4.5,3.0,1.5,0.25])
X_example = X_example.reshape(1,4)
y_example = iris_classifier.predict(X_example)
print(y_example)

print(classification_report(y_test,y_score))

conf=confusion_matrix(y_test,y_score)
print('Confusion matrix \n', conf)

ConfusionMatrixDisplay.from_predictions(
    y_test,y_score,cmap="Greys")


### 3 Demystifiying the XGBoost Paper
# paper: 1603.02754
#
# Examining paper XGBoost: A Scalable Tree Boosting System at a high level
# Exploring the features and benefits of XGBoost
# Understanding the XGBoot algorithm
# Comparing XGBoost with other enemble decision tree nodel...
#
# CART (classification and regression tree)
#   building a decision tree, minimizing a loss function...
#
# XGBoosts algorithm...
#
#
 
### 4 Adding on to the Quick Start - Switching out the Dataset with
# a Housing Data Case Study

# Data download
import pandas as pd
import numpy as np
from sklearn import datasets
housingX, housingy = datasets.fetch_california_housing(
    return_X_y = True, as_frame = True)

# Exploring the dataset by making graphs
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_palette("dark:grey")
sns.displot(housingy,kind="hist")

# Look for relationships in the X data:
graphx = sns.pairplot(housingX)

# Look for correlations between the target y parameter and the input X parameterx:
housingxy = pd.concat([housingy,housingX],axis=1)
graphxy = sns.pairplot(housingxy)

# Preparing data for predictive modeling
# Split the housing data into training and test DataFrames:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    housingX,housingy,test_size=0.2,random_state=17)

# Verify the split has occurred correctly:
X_train.shape
X_test.shape
y_train.shape
y_test.shape
X_test.head()


# XGBoost predictive model settings and model training
# Train the XGBoost regression model
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
housevalue_regressor = xgb.XGBRFRegressor()

# prediction using XGBoost
y_score = housevalue_regressor.predict(X_test)

# prediction (inference) based on example measurements:
X_example = np.array([12.5,10,9,3,4000,3,33.98,-118.45])
X_example = X_example.reshape(1,8)
y_example = housevalue_regressor.predict(X_example)
print(y_example)

# Test the effectiveness of the model by calculating the fit parameter R2 value.
predicter_r2 = r2_score(y_true=y_test, y_pred=y_score)
print(predicter_r2)

# Test the effectiveness of the model by calculating the fit parameter RMSE value
predicter_rmse = mean_squared_error(
    y_true = y_test, y_pred = y_score, squared=False)
print(predicter_rmse)

sns.regplot(
    x=y_test,y=y_score,
    scatter_kws={"color": "grey"},
    line_kws={"color": "black"}
    )
residuals = y_test - y_score
print(residuals)

X_testResiduals = pd.concat([X_test,residuals], axis=1)
X_testResiduals.head()

fig.axes = plt.subplots(
    nrows=2, ncols=4, figsize=(24,16), sharey=True)
axes[0,0].scatter(x=X_testResiduals["MedInc"],
                  y=X_testResiduals["MedHouseVal"],
                  alpha=0.5, color="grey")
axes[0,0].set_title("MedInc")

axes[0,1].scatter(x=X_testResiduals["HouseAge"],
                  y=X_testResiduals["MedHouseVal"],
                  alpha=0.5, color="grey")
axes[0,1].set_title("HouseAge")

axes[0,2].scatter(x=X_testResiduals["AveRooms"],
                  y=X_testResiduals["MedHouseVal"],
                  alpha=0.5, color="grey")
axes[0,2].set_title("AveRoom")

axes[0,3].scatter(x=X_testResiduals["AveBerms"],
                  y=X_testResiduals["MedHouseVal"],
                  alpha=0.5, color="grey")
axes[0,2].set_title("AveBedrms")

axes[1,0].scatter(x=X_testResiduals["Population"],
                  y=X_testResiduals["MedHouseVal"],
                  alpha=0.5, color="grey")
axes[1,0].set_title("Population")

axes[1,1].scatter(x=X_testResiduals["AveOccup"],
                  y=X_testResiduals["MedHouseVal"],
                  alpha=0.5, color="grey")
axes[1,1].set_title("AveOccup")

axes[1,2].scatter(x=X_testResiduals["Latitude"],
                  y=X_testResiduals["MedHouseVal"],
                  alpha=0.5, color="grey")
axes[1,2].set_title("Latitude")

axes[1,3].scatter(x=X_testResiduals["Longitude"],
                  y=X_testResiduals["MedHouseVal"],
                  alpha=0.5, color="grey")
axes[1,3].set_title("Longitude")


## Classification and Regression Trees, Ensembles, and Deep
## Learning Models - Whats Best for Your Data
#
# Comparing models with the housing dataset
import pandas as pd
import numpy as np

# load the California housing dataset from scikit-learn:
from sklearn import datasets
housingX, housingy = datasets.fetch_california_housing(
    return_X_y = True, as_frame=True)

# Prepare the data for modeling...
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    housingX, housingy, test_size=0.2, random_state=17)


# Comparing XGBoost to linear regression...
#
# Perform a linear fit on the data...
from sklearn.linear_model import LinearRegression
housing_linear_regression = LinearRegression().fit(
    X_train, y_train)

# Check the RMSE for the linear model....
from sklearn.metrics import root_mean_squared_error
housing_linreg_ypred = housing_linear_regression.predict(X_test)

housing_linreg_rmse = root_mean_squared_error(y_true=y_test,
                                              y_pred=housing_linreg_ypred)

# Check the R^2 value for the linear model...
housing_linreg_r2 = housing_linear_regression.score(
    X_test, y_test)
print("Linear regression Rsquared is {0:.2f}".format(
    housing_linreg_r2))


# Plot the predicted values compared to the actuals....
import matplotlib.pyplot as plt
import seaborn as sns
sns.regplot(x=y_test, y=housing_linreg_ypred,
            scatter_kws={"color": "grey"},
            line_kws={"color": "black"})

# Compare XGBoost to CART
#
# Fit a regression tree model using CART:
from sklearn.tree import DecisionTreeRegressor
housing_CART = DecisionTreeRegressor()
housing_CART_regression = housing_CART.fit(X_train,y_train)

# predict
housing_cart_ypred = housing_CART_regression.predict(X_test)

# Check the RMSE value for the CART model...
housing_cart_rmse = root_mean_squared_error(
    y_true=y_test, y_pred=housing_cart_ypred)
print("CART RMSE is {0:.2f}".format(housing_cart_rmse))

# Calculate the R^2 value for the CART model:
housing_cart_r2 = housing_CART_regression.score(X_test, y_test)
print("Rsquared is {0:.2f}".format(housing_cart_r2))

# plot the predicted values compard to the actuals...
sns.regplot(x=y_test,y=housing_cart_ypred,
            scatter_kws={"color": "grey"},
            line_kws={"color": "black"})


# When to use CART
from sklearn import tree
tree.plot_tree(housing_CART)

# Comparing XGBoost to gradient boosting and random forest models...
#
# fit a regression model using gradient-boosted tree:
# 
from sklearn.ensemble import GradientBoostingRegressor
housing_gbt = GradientBoostingRegressor(
    random_state=17, max_depth=6)
housing_gbt_regression = housing_gbt.fit(X_train,y_train)

# predict the results for the X_test dataset
housing_gbt_ypred = housing_gbt_regression.predict(X_test)

# calculate the RMSE value for the predicted y values:
housing_gbt_rmse = root_mean_squared_error(y_true=y_test,
                                           y_pred=housing_gbt_ypred)
print("Gradient boosting regressor RMSE is {0:.2f}".format(housing_gbt_rmse))

# Calculate the R^2  value for the predicted y values.
housing_gbt_r2 = housing_gbt_regression.score(
    X_test, y_test)
print("Gradient boosting regressor Rsquared is {0:.2f}".format(housing_gbt_r2))

# plot the predicted values compared to the actuals...
sns.regplot(x=y_test, y=housing_gbt_ypred,
            scatter_kws={"color": "grey"}, line_kws={"color": "black"})

# Fit a regression model using random forest
from sklearn.ensemble import RandomForestRegressor
housing_rf = RandomForestRegressor(random_state=17)
housing_rf_regression = housing_rf.fit(X_train,y_train)

# predict the results for the X_test dataset:
housing_rf_ypred = housing_rf_regression.predict(X_test)

# Calculate the RMSE for the predicted y values:
housing_rf_rmse = root_mean_squared_error(
    y_true=y_test,y_pred=housing_rf_ypred)
print("Random Forest RMSE is {0:.2f}".format(housing_rf_rmse))

# perform a calculate
housing_rf_r2 = housing_rf_regression.score(X_test,y_test)
print("Random forest Rsquared is {0:.2f}".format(housing_gbt_r2))

# plot the predicted values vompared to the actuals
sns.regplot(x=y_test, y=housing_rf_ypred,
            scatter_kws={"color": "grey"}
            line_kws={"color": "black"})


########
#
# Chapter 6
# Date Cleaning, Imbalanced Data, and Other Data Problems
#
# Data-cleaning methods
#
# pandas DataFrame
import pandas as pd
import numpy as np
testdf = pd.DataFrame(np.random.randn(5250,3), columns=list("ABC"))
# test category
testdf.insert(len(testdf.columns), "category","Category A")
# value row 3000 to 4499
testdf.loc[3000:4499,["category"]] = "CategoryB"
testdf.loc[4500:5250,["category"]] = "Cat C"
# create Date columns and polulate
testdf.insert(len(testdf.columns),"Date",np.random.choice(
    pd.date_range('2022-10-01','2024-11-30'),len(testdf)))

# 4-pseudi sigma filter for removing outliers from continuous data...
#
# 4-pseudo sigma filter:
def pseudosigmafiilter(datafram,parameter):
    mean = dataframe[parameter].mean()
    stdev = dataframe[parameter].std()
    lowerfiltervalue = mean - (4*stdev)
    upperfiltervalue = mean + (4*stdev)
    print(mean)
    print(stdev)
    print(lowerfiltervalue)
    print(upperfiltervalue)

# filter the data by creating
dataframe["filter_" + parameter] = np.where(
    ((dataframe[parameter] > lowerfiltervalue) & (
        dataframe[parameter] < upperfiltervalue)),
    dataframe[parameter], np.NaN)
return dataframe

# filter works by injection a large value into
testdf.at[0,"A"] = 11
pseudosigmafilter(testdf, "A")

# Standardizing continuous data
from sklearn import preprocessing
continuous = testdf[["A","B","C"]]
# fit StandardScaler
standardized = preprocessing.StandardScaler().fit(continuous)
# apply the standardization
standardized = standardized.transform(continuous)
print(standardized)
# pandas DataFrame
standardizeddf = pd.DataFrame(standardized,columns=["A","B","C"])
# concatenate DataFrame
standardizedtestdf = pd.concat([testdf, standardizeddf],axis=1)
# only center the data
centered = preprocessing.StandardScaler(with_std=False).fit(continuous)

# Normalizing  continuous data...
from sklearn import preprocessing
# create function
def normalizecolumn(dataframe, paramter):
    colarray = np.array(dataframe[paramter])
    normalizedarray = preprocessing.normalize([colarray]).tollist()
    normalizedarray = np.swapaxes(normalizedarray,0,1)
    dataframe["normalize_"+paramter] = normalizedarray
    return dataframe
normalizecolumn(testdf,"B")

# Correcting spelling in a column
cleancategory(testdf,"category")

# Reformatting date data....
testdf["Date_new"].dt.strftime("%d-%b-%Y")
testdf["Data_US"] = testdf["Date"].dt.strftime("%m/%d/%Y")

# Handling imbalanced data
# Correting imbalanced data by sampling the dataset
def subsamplecategory(oldataframe, category, nsamples):
    newdataframe = olddataframe.groupby(category).apply(lambda s: s.sample(nsamples))
    return newdataframe
