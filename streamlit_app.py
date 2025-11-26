import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import streamlit.components.v1 as components
import looker_sdk
import matplotlib.pyplot as plt


st.sidebar.markdown("<h1 style='color: lightsalmon;'>Happy Place</h1>", unsafe_allow_html=True)
st.sidebar.write("Your journey to joy")

st.image("OpenAI.jpeg", use_container_width=True)
st.markdown("<h1 style='color: lightsalmon; text-align: center;'>Happy Place</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: lightpink; text-align: center;'>Notice what you need. Nurture how you feel.</p>", unsafe_allow_html=True)

df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")
### Intro Page
page = st.sidebar.selectbox("Select Page",["Introduction","Data Viz","Prediction", "Crystal Ball",])
##Introduction Page
if page == "Introduction":
    st.image("hugol-halpingston-4OyLq2yN9u0-unsplash.jpg", use_container_width=True)
    st.markdown("<h2 style='color: lightpink; text-align: center;'>Introduction</p>", unsafe_allow_html=True)


    st.subheader("Our Mission")
    st.write("Sometimes you don’t even know why you’re feeling low, only that something needs to change. This app helps you pause, reflect, and uncover the patterns beneath your mood by organizing your daily habits—sleep, social media use, movement, and more—into clear, meaningful insights. As you start to see how your choices shape your well-being, you realize you have far more control than you thought.")
    st.markdown("<p style='color: lavender; '>Take a moment, search inward, and let your data gently guide you toward a happier, more intentional life!</p>", unsafe_allow_html=True)
    
    st.subheader("Data Set")
    st.markdown("<p style='color: lavender; '>Look here and see for yourself what we use to inform our student success visualizations and predictions!</p>", unsafe_allow_html=True)

    st.markdown("##### Data Preview")

    rows = st.slider("Select a number of rows",5,20,5)
    
    st.dataframe(df.head(rows))

    st.markdown("##### Missing Values")

    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum()==0:
        st.success("No missing values found")
    else:
        st.warning("This data has some missing values")

    st.markdown("#### 🧠 Statistical Summary")

    if st.button("Click Here to Generate Statistical Summary!"):
        st.dataframe(df.describe())
## Business Problem Presentation
## Data Summary Presentation
