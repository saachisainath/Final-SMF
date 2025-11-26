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


st.sidebar.title("<h1 style='color: lightsalmon; text-align: center;'>Happy Place</h1>", unsafe_allow_html=True)
st.sidebar.write("Your roadmap to your happiest self!")

st.image("OpenAI.jpeg", use_container_width=True)
st.markdown("<h1 style='color: lightsalmon; text-align: center;'>Happy Place</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: lightpink; text-align: center;'>Notice what you need. Nurture how you feel.</p>", unsafe_allow_html=True)

df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")
### Intro Page
## Business Problem Presentation
## Data Summary Presentation
