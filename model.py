import pandas as pd, numpy as np, catboost as cb
from sklearn.impute import KNNImputer
from imblearn.over_sampling import ADASYN
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, classification_report

# preprocessing
drop_column = ['Bike Model', 'X', 'Y', 'Index', 'Event Unique Id', 'Occurrence Date', 'City', 'Location Type', 'Neighbourhood', 'Lat', 'Long', 'Object Id', 'Season']
data = pd.read_csv('Bicycle_Thefts.csv')
data.drop(drop_column, axis=1, inplace=True)
data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True) # Neighbourhood is identical with Hood ID
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
# low = data['Cost of Bike'].quantile(.25)
# average = data['Cost of Bike'].quantile(.5)
# high = data['Cost of Bike'].quantile(.75)
# data['cost tier'] = np.where(data['Cost of Bike'] <= low, 'low', np.where((data['Cost of Bike'] > low) & (data['Cost of Bike'] <= average), 'average', np.where((data['Cost of Bike'] > average) & (data['Cost of Bike'] <= high), 'high', 'luxury')))
data['Status'].replace('STOLEN', 0, inplace=True)
data['Status'].replace(['UNKNOWN', 'RECOVERED'], 1, inplace=True)
# encoding categorical features
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
for col in categorical_cols:
	data[col] = encoder.fit_transform(data[col])

# todo try other year split
# data earlier than 2019 are split as training set
train_x, train_y = data[data['Occurrence Year'] < 2019].drop('Status', axis=1), data[data['Occurrence Year'] < 2019]['Status']
train_x, train_y = ADASYN().fit_resample(train_x, train_y)
test_x, test_y = data[data['Occurrence Year'] == 2019].drop('Status', axis=1), data[data['Occurrence Year'] == 2019]['Status']
test_x, test_y = ADASYN().fit_resample(test_x, test_y)

# modeling
boost = cb.CatBoostClassifier()
boost.fit(train_x, train_y, cat_features=categorical_cols)
predict = boost.predict(test_x)
print(classification_report(test_y, predict))
fpr, tpr, thresholds = roc_curve(test_y, predict)
print('auc is %f' % auc(fpr, tpr))

# cross validation
X, Y = data.drop('Status', axis=1), data['Status']
X, Y = ADASYN().fit_resample(X, Y)
score = cross_val_score(boost, X, Y, cv=5, scoring='f1')
# f1 score on cross validation: [0.8787948 0.94090382 0.91838545 0.98004695 0.79670277]
print('f1 score on cross validation: ' + str(score))