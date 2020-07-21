#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle

st.write("""
# Predicting Red Wine Quality deployed using streamlit
For this project, I made use of Kaggle's Red Wine Quality dataset to build a logistic regression model to predict whether a particular wine is "good quality" or not based on physicochemical qualities. The objective of this project is to experiment with various classification methods to determine which model yields the highest accuracy and also determine the features which are most indicative of a good quality wine.
""")

st.sidebar.header('Wine Input Features')

# Functionalize model fittting
import pickle

def user_input_features():
    V1 = st.sidebar.slider('Fixed Acidity', 4.6, 16.0, 8.0)
    V2 = st.sidebar.slider('Volatile Acidity', 0.1, 1.6, 0.5)
    V3 = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.3)
    V4 = st.sidebar.slider('Residual Sugar', 0.9, 15.5, 2.5)
    V5 = st.sidebar.slider('Chlorides', 0.0, 0.7, 0.1)
    V6 = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 16.0)
    V7 = st.sidebar.slider('Total Sulfur Dioxide', 1.0, 289.0, 46.0)
    V8 = st.sidebar.slider('Density', 0.9, 1.1, 0.99)
    V9 = st.sidebar.slider('pH', 2.7, 4.0, 3.3)
    V10 = st.sidebar.slider('Sulphates', 0.33, 2.0, 0.6)
    V11 = st.sidebar.slider('Alchohol', 8.4, 14.9, 10.4)
    data = {'Fixed Acidity': V1,
           'Volatile Acidity': V2,
           'Citric Acid': V3,
           'Residual Sugar': V4,
           'Chlorides': V5,
           'Free Sulfur Dioxide': V6,
           'Total Sulfur Dioxide': V7,
           'Density': V8,
           'pH': V9,
           'Sulphates': V10,
           'Alchohol': V11
           }

    features= pd.DataFrame(data, index=[0])
    return features
df =  user_input_features()
st.subheader('Wine Input Features')
st.write(df)

filename = 'LogisticRegression'
model = pickle.load(open(filename,'rb'))
preprocess = pickle.load(open('preprocess_scale','rb'))
pred = model.predict(preprocess.transform(df))
st.subheader('Prediction - 1 denotes good quality wine, 0 denotes bad quality wine')
st.write(pred)
