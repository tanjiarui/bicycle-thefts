import pandas as pd, numpy as np, shap, ppscore as pps, matplotlib.pyplot as plt, seaborn as sns, pandas_profiling as profile
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance as plot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from lofo import LOFOImportance, Dataset, plot_importance

drop_column = ['Bike Model', 'X', 'Y', 'Index', 'Event Unique Id', 'Occurrence Date', 'City', 'Location Type', 'Neighbourhood', 'Lat', 'Long', 'Object Id']
data = pd.read_csv('Bicycle_Thefts.csv')
data.drop(drop_column, axis=1, inplace=True)
data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True) # Neighbourhood is identical with Hood ID
report = profile.ProfileReport(data)
report.to_file('EDA')
data['Occurrence Time'] = pd.to_datetime(data['Occurrence Time']).dt.time # change data type
data['Bike Colour'].fillna('other', inplace=True) # fill nan value
data['Cost of Bike'].replace(0, np.nan, inplace=True) # zero is also invalid
unknown_make = ['UK', 'UNKNOWN MAKE', 'UNKNOWN', 'NONE', 'NO', 'UNKNOWNN', 'UNKONWN', 'UNKOWN', 'UNKNONW', '-', 'UNKNOW', 'NO NAME', '?'] # all typos stand for known
giant = data['Bike Make'][data['Bike Make'].str.contains('giant',case=False,na=False)].unique().tolist() # alias of giant
giant.append('GI')
data['Bike Make'].replace(giant, 'GIANT', inplace=True)
data['Bike Make'].replace('OT', 'OTHER', inplace=True)
data['Bike Make'].replace(unknown_make, np.nan, inplace=True)
encoder = preprocessing.LabelEncoder()
data['Bike Type'] = encoder.fit_transform(data['Bike Type']) # only numerical values for KNNImputer
data['Bike Make'] = pd.Series(encoder.fit_transform(data['Bike Make'][data['Bike Make'].notna()]),index=data['Bike Make'][data['Bike Make'].notna()].index) # only numerical values for KNNImputer
data[['Bike Type', 'Bike Speed', 'Cost of Bike']] = KNNImputer().fit_transform(data[['Bike Type', 'Bike Speed', 'Cost of Bike']])
data[['Bike Type', 'Bike Speed', 'Bike Make']] = KNNImputer().fit_transform(data[['Bike Type', 'Bike Speed', 'Bike Make']])
# convert cost to cost tier
low = data['Cost of Bike'].quantile(.25)
average = data['Cost of Bike'].quantile(.5)
high = data['Cost of Bike'].quantile(.75)
data['cost tier'] = np.where(data['Cost of Bike'] <= low, 'low', np.where((data['Cost of Bike'] > low) & (data['Cost of Bike'] <= average), 'average', np.where((data['Cost of Bike'] > average) & (data['Cost of Bike'] <= high), 'high', 'luxury')))
data['Status'].replace('STOLEN', 0, inplace=True)
data['Status'].replace(['UNKNOWN','RECOVERED'], 1, inplace=True)
# encoding categorical features
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
for col in categorical_cols:
	data[col] = encoder.fit_transform(data[col])
X, Y = data.drop('Status', axis=1), data['Status']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

# optional. it is used for determining non-linear correlations
matrix = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
plt.show()

# evaluate on xgboost
def plot_features(booster, figsize):
	fig, ax = plt.subplots(1,1,figsize=figsize)
	return plot(booster=booster, ax=ax)

model = XGBRegressor()
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)
plot_features(model, (14,14))
plt.show()

# SHAP explanation by xgboost
explainer = shap.TreeExplainer(model, x_train)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train)
shap.summary_plot(shap_values, x_train, plot_type='bar')

# SHAP explanation by logistic regression
model = LogisticRegression().fit(X, Y)
x_summary = shap.kmeans(X, 10)
explainer = shap.KernelExplainer(model.predict_proba, x_summary)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

# SHAP explanation by random forest
model = RandomForestClassifier().fit(X, Y)
explainer = shap.TreeExplainer(model, X)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

# SHAP explanation by lightgbm
model = LGBMRegressor().fit(X, Y)
explainer = shap.TreeExplainer(model, X)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type='bar')

dataset = Dataset(df=data, target='Status', features=[col for col in data.columns if col != 'Status'])
# evaluate on knn
knn_importance = LOFOImportance(dataset, scoring='f1', model=KNeighborsClassifier())
knn_importance = knn_importance.get_importance()
plot_importance(knn_importance, figsize=(14, 14))
plt.show()
# evaluate on logistic regression
logistic_importance = LOFOImportance(dataset, scoring='f1', model=LogisticRegression())
logistic_importance = logistic_importance.get_importance()
plot_importance(logistic_importance, figsize=(14, 14))
plt.show()
# evaluate on random forest
rf_importance = LOFOImportance(dataset, scoring='f1', model=RandomForestClassifier())
rf_importance = rf_importance.get_importance()
plot_importance(rf_importance, figsize=(14, 14))
plt.show()
# evaluate on lightgbm
lightgbm_importance = LOFOImportance(dataset, scoring='f1')
lightgbm_importance = lightgbm_importance.get_importance()
plot_importance(lightgbm_importance, figsize=(14, 14))
plt.show()