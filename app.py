import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200).fit(X, y)

st.title("🌸 Iris Flower Prediction")

features = [
    ('Sepal Length (cm)', 4.0, 8.0, 5.1),
    ('Sepal Width (cm)', 2.0, 4.5, 3.5),
    ('Petal Length (cm)', 1.0, 7.0, 1.4),
    ('Petal Width (cm)', 0.1, 2.5, 0.2)
]

inputs = [st.slider(label, min_value=low, max_value=high, value=default)
          for label, low, high, default in [(f[0], f[1], f[2], f[3]) for f in features]]

if st.button('Predict'):
    prediction = model.predict([inputs])[0]
    st.success(f" Predicted Iris Species: **{iris.target_names[prediction]}**")

