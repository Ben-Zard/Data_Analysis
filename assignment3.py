

files.upload()

import  pandas  as pd
test_df = pd.read_csv('test(1).csv')
train_df = pd.read_csv('train(1).csv')

train_df = train_df.drop(['id'],axis = 1)
test_df = test_df.drop(['id'],axis = 1)

train_df.loc[train_df['diagnosis'] == 'M',"diagnosis"] = 1
train_df.loc[train_df['diagnosis'] == 'B',"diagnosis"] = 0

test_df.loc[test_df['diagnosis'] == 'M',"diagnosis"] = 1
test_df.loc[test_df['diagnosis'] == 'B',"diagnosis"] = 0

features = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
            'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se',
            'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se',
            'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
            'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']

y_train = train_df["diagnosis"]
y_test = test_df["diagnosis"]

y_train = y_train.astype('int')
y_test = y_test.astype('int')

X_train = train_df[features]
X_test = test_df[features]

print(train_df.head())

#min-max
min_vals = X_train[features].min() #can add how much ever columns
max_vals = X_test[features].max() #can add how much ever columns

X_train = (X_train - min_vals) / (max_vals - min_vals)

#z-score
X_train = (X_train - X_train.mean()) /(X_train.std())
X_test = (X_test - X_train.mean()) / (X_train.std())

#no preprocessing 
from sklearn.tree import DecisionTreeClassifier
base_cls = DecisionTreeClassifier(min_weight_fraction_leaf = .2)

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(base_estimator = base_cls, n_estimators = 15, random_state = 5,learning_rate = .5)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier( n_estimators = 45, random_state = 1,max_depth = 20 , min_samples_leaf = 2)

#Bagging classifier
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator = base_cls, n_estimators = 50, random_state = 3,verbose = 1.5,n_jobs =2)

#prediction
model.fit(X_train, y_train)

prediction = model.predict(X_test)

#f-score,#recall,#precision,#Accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))

import seaborn as sns
sns.swarmplot(data=prediction)

#Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,prediction)
print(score)

#precision
from sklearn.metrics import precision_score
score2 =precision_score(y_test,prediction, average= None)
print(score2)