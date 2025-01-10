import os
import opendatasets as od
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
data_dir=r'C:\Users\BIT\.cache\kagglehub\datasets\fedesoriano\stroke-prediction-dataset\versions\1'
train_csv=data_dir+r"\healthcare-dataset-stroke-data.csv"
raw_df=pd.read_csv(train_csv)
data = raw_df[raw_df['gender'] != 'Other']
raw_df['gender'] = raw_df['gender'].map({'Male': 1, 'Female': 0})
raw_df['ever_married'] = raw_df['ever_married'].map({'Yes': 1, 'No': 0})
train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
input_col=list(train_df.columns)[1:-1]
target_col='stroke'
train_input=train_df[input_col].copy()
train_target=train_df[target_col].copy()
val_inputs = val_df[input_col].copy()
val_target = val_df[target_col].copy()
test_input = test_df[input_col].copy()
test_target = test_df[target_col].copy()
numeric_col=train_input.select_dtypes(include=np.number).columns.tolist()
catagorical_col=train_input.select_dtypes('object').columns.tolist()
imputer= SimpleImputer(strategy='mean')
imputer.fit(raw_df[numeric_col])
train_input[numeric_col]=imputer.transform(train_input[numeric_col])
val_inputs[numeric_col]=imputer.transform(val_inputs[numeric_col])
test_input[numeric_col]=imputer.transform(test_input[numeric_col])
scaler= MinMaxScaler(feature_range=(0,1))
scaler.fit(raw_df[numeric_col])
train_input[numeric_col]=scaler.transform(train_input[numeric_col])
val_inputs[numeric_col]=scaler.transform(val_inputs[numeric_col])
test_input[numeric_col]=scaler.transform(test_input[numeric_col])
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoder.fit(raw_df[catagorical_col])
raw_df=raw_df[catagorical_col].fillna('unknown')
encoded_cols= list(encoder.get_feature_names_out(catagorical_col))
train_input[encoded_cols]=encoder.transform(train_input[catagorical_col])
val_inputs[encoded_cols]=encoder.transform(val_inputs[catagorical_col])
test_input[encoded_cols]=encoder.transform(test_input[catagorical_col])
x_train=train_input[numeric_col + encoded_cols]
x_val=val_inputs[numeric_col + encoded_cols]
x_test=test_input[numeric_col + encoded_cols]
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20, 30], 
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],    
    'class_weight': ['balanced', 'balanced_subsample'],  
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(x_train,train_target)
best_rf = grid_search.best_estimator_
train_pred = best_rf.predict(x_train)
joblib.dump(best_rf, 'stroke_prediction_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
print("Model and preprocessing pipeline saved successfully.")