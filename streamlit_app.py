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


st.sidebar.title("Happy Place")
st.sidebar.write("Notice what you need. Nurture how you feel")

st.image("
### Intro Page
## Business Problem Presentation
## Data Summary Presentation
