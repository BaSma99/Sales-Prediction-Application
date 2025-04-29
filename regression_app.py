import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io

st.title('Sales Prediction Application')

file = st.sidebar.file_uploader('Please, Upload a CSV File:', type = ['csv'])

if file is not None:
   # image = Image.open(file)
    #st.image(image, caption = 'Uploaded Image', use_column_width= True)
    data = pd.read_csv(file)
    st.write('Uploaded Data', data.head())

    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    st.write('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    st.write('R2 Score:', r2_score(y_test, y_pred))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    st.pyplot(fig)


st.write(data.describe())
st.sidebar.write('Please, Enter Input Data to Predict Sales:')

def user_input():
    TV = st.sidebar.slider('TV', 0.0, 300.0, 150.0)
    Radio = st.sidebar.slider('Radio', 0.0, 50.0, 25.0)
    Newspaper = st.sidebar.slider('Newspaper', 0.0, 120.0, 60.0)

    data = {
        'TV' : TV,
        'Radio' : Radio,
        'Newspaper' : Newspaper
    }

    features = pd.DataFrame([data])

    return features

inputs = user_input()

if st.sidebar.button('Predict Sales'):
    prediction = model.predict(inputs)
    st.sidebar.write('Sales Prediction: ', prediction)
