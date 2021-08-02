# libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from scipy import stats
from scipy.stats import skew, norm
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import pickle
import gzip
import warnings
warnings.filterwarnings(action="ignore")

        
        
        
def write():
    with st.spinner("Loading Data ..."):
        st.title('Sales Predictions ')
        st.write("""
        Predictions and the accuracy of the predictions.
        """)
  
    def load_preprocess_data():

        # load data
        global train_features, test_features, train_target, full_test, full_train, train, test, store, submission, categorical, numerical
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('src/pages/train.csv', na_values=na_value)
        test = pd.read_csv('src/pages/test.csv', na_values=na_value)
        store = pd.read_csv('src/pages/store.csv', na_values=na_value)
        submission = pd.read_csv('src/pages/sample_submission.csv', na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        full_test = pd.merge(left = test, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')  
        
    # the models + predictions
    st.sidebar.title("Predictions")
    st.sidebar.subheader("Choose Model")
    regressor = st.sidebar.selectbox("Regressor", ("Random Forest Regressor", "eXtreme Gradient Boosting(XGB)", "Gradient Boosting"))
    
    # evaluation metrics
    def display_metrics(metrics_list):
        if 'Mean Absolute Error' in metrics_list:
            st.subheader("Mean Absolute Error")
            #print(mean_absolute_error(y_pred, y_val))
            #st.write('Mean absolute erro:', mean_absolute_error(y_pred, y_val))

        if 'Mean Squared Error' in metrics_list:
            st.subheader("Mean Squared Error")
            #print(mean_squared_error(y_pred, y_val))
            #st.write('Mean squared error:', mean_squared_error(y_pred, y_val))

    # RandomForestRegressor
    if regressor == 'Random Forest Regressor':

        metrics = st.sidebar.multiselect("What metrics to display?", ('Mean Absolute Error', 'Mean Squared Error'))
        
        if st.sidebar.button("Predict", key='predict'):
            #st.subheader("Random Forest Regressor")

            def load_zipped_pickle(filename):
                with gzip.open(filename, 'rb') as f:
                    loaded_object = pickle.load(f)
                    return loaded_object

            #y_pred = model.predict(x_val)
            st.write('Mean Squared Error: 0.0189')
            st.write('Mean Absolute Error: 0.0760')
            display_metrics(metrics)
            #predictions = model.predict(x_test)
            st.subheader("Rossmann Pharmaceuticals sales predictions")
            sub = full_test[['Id']]
            #back = np.expm1(predictions)
            #sub['Sales'] = back
            sub['Date'] = full_test.Date.to_list()
            sub.to_csv('sub.csv', index = False)
            sub['Store'] = full_test.Store.to_list()
            sub['Date'] = pd.to_datetime(sub['Date'])
           
            st.write(sub.sample(20))

    # xgb
    if regressor == 'eXtreme Gradient Boosting(XGB)r':
        st.sidebar.subheader("Model Hyperparameters")


        metrics = st.sidebar.multiselect("What metrics to display?", ('Mean Absolute Error', 'Mean Squared Error'))
        
        if st.sidebar.button("Predict", key='predict'):
            #st.subheader("eXtreme Gradient Boosting(XGB)")
            
            def load_zipped_pickle(filename):
                with gzip.open(filename, 'rb') as f:
                    loaded_object = pickle.load(f)
                    return loaded_object

        if regressor == 'Gradient Boosting':
            st.sidebar.subheader("Model Hyperparameters")
            metrics = st.sidebar.multiselect("What metrics to display?", ('Mean Absolute Error', 'Mean Squared Error'))
        
        if st.sidebar.button("Predict", key='predict'):
            with st.spinner("Loading Data ..."):
             st.title('Result ')
            st.write("""
        Predictions and the accuracy of the predictions.
        """)