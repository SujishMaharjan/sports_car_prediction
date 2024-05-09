#importing libraries and dependencies
import numpy as np
import pandas as pd
import streamlit as st

import pickle
import datetime
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

st.title('Sports Car Price Prediction')

#loading the cleaned_data 
df = pd.read_csv("cleaned_sports_car_data.csv")
df.head()

#splitting the dataset into train and test sets
X = df.drop(columns='Price (in USD)',axis=1)
y=df['Price (in USD)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#training the model using decision Tree algorithm
tree_model = DecisionTreeRegressor()
# Fit the model to the training data
tree_model.fit(X_train, y_train)

#Getting from the user
engine_size = st.text_input("Engine size")
horse_power = st.text_input("Horsepower")
torque = st.text_input("Torque (lb-ft)")
mph_time = st.text_input("0-60 MPH Time (seconds)")

if st.button("Submit"):
    data = {'Engine Size (L)': [engine_size],
        'Horsepower':[horse_power],
        'Torque (lb-ft)':[torque],
        '0-60 MPH Time (seconds)':[mph_time]}
    user_df = pd.DataFrame(data)
    predicted_price = tree_model.predict(user_df)
    st.write("predicted price = ",predicted_price[0])



df = pd.read_csv('cleaned_sports_car_data.csv',encoding='latin1')
# df = df.sample(frac = 1)
df

# seems there is certain problem in pickle file there is an 
# error in prediction