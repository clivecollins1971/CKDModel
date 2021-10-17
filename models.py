import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

dataset = pd.read_csv('CKD_raw2.csv', skipinitialspace=True, na_values='?')
dataset['class'] = dataset['class'].replace(to_replace=['notckd', 'ckd'], value=[0, 1])
NumList = []
CatList = []

# Drop rbc, rbcc, hemo  and ane good proxy for red cell count and rbcc has alot of missing values
DropList = ['rbc', 'rbcc', 'pcv', 'su']

# add numerical and categorical columns to separate lists and exclude the response variable 'class'
for col in dataset.columns:
    if col != 'class':
        if dataset.dtypes[col] == np.object:
            CatList.append(col)
        else:
            NumList.append(col)

y = dataset['class'].values
X = dataset.drop('class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Pipeline to transform numerical attributes. This will replace median of each column with missing values and
# standardize attributes
num_transfs = [('impute', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())]
num_pipeline = Pipeline(num_transfs)

# Pipeline to transform categorical attributes. This will replace most frequent of each column with missing values
# and assign numbers for each categories.
cat_transfs = [('impute', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder())]
cat_pipeline = Pipeline(cat_transfs)

# The complete pipeline to transform entire dataframes
all_transfs = [('numeric', num_pipeline, NumList), ('categorical', cat_pipeline, CatList), ('drops', 'drop', DropList)]
full_pipeline = ColumnTransformer(all_transfs, remainder='passthrough')

# Transform the dataset
X_train_transformed = full_pipeline.fit_transform(X_train)
X_test_transformed = full_pipeline.fit_transform(X_test)

# Create Logistic Model
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X_train_transformed, y_train)

# Instantiate a SVM Classifier
svm_model = SVC(C=10, gamma='scale', kernel='rbf')
# Fit the model on tranformed training data
svm_model.fit(X_train_transformed, y_train)

# Save the models
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))

pickle.dump(logistic_model, open('logistic_model.pkl', 'wb'))

# Save the pipeline
pickle.dump(full_pipeline, open('pipeline.pkl', 'wb'))
