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


st.sidebar.markdown("<h1 style='color: lightsalmon; text-align: center; '>🌻 Happy Place 🌻</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: lightpink; text-align: center; '>Your journey to joy!</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: lightsalmon; text-align: center; '>💐🌺🌷🌻🪷🪻🌸</h1>", unsafe_allow_html=True)

st.image("OpenAI.jpeg", use_container_width=True)
st.markdown("<h1 style='color: lightsalmon; text-align: center;'>🌻 Happy Place 🌻</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: lightpink; text-align: center;'>Notice what you need. Nurture how you feel.</p>", unsafe_allow_html=True)

df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")
### Intro Page
page = st.sidebar.selectbox("Select Page",["🌺 Introduction","🪻 Data Viz","🌸 Prediction", "💐 The Garden",])
##Introduction Page
if page == "🌺 Introduction":

    st.image("hugol-halpingston-4OyLq2yN9u0-unsplash.jpg", use_container_width=True)
    st.markdown("<h2 style='color: lightpink; text-align: center;'>Introduction</p>", unsafe_allow_html=True)


    st.subheader("🌸 Our Mission")
    st.markdown("<p style='color: lightpink; '>Sometimes you don’t even know why you’re feeling low, only that something needs to change. This app helps you pause, reflect, and uncover the patterns beneath your mood by organizing your daily habits—sleep, social media use, movement, and more—into clear, meaningful insights. As you start to see how your choices shape your well-being, you realize you have far more control than you thought.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: lightpink; '>Take a moment, search inward, and let your data gently guide you toward a happier, more intentional life!</p>", unsafe_allow_html=True)


    st.subheader("🌺 Look Inward")
    st.markdown("<p style='color: lightpink; '>Happiness is less complicated than you may think. Let's break down the factors that contribute to your mental health.</p>", unsafe_allow_html=True)

    if st.button("Sleep Quality"):
        st.markdown("<p style='color: hotpink; '>Quality sleep is essential for feeling energized, happy, and mentally clear. When you sleep well, your mood improves, your focus sharpens, and you’re better able to enjoy life’s moments and handle challenges with resilience.</p>", unsafe_allow_html=True)

    if st.button("Stress Level"):
        st.markdown("<p style='color: salmon; '>Managing stress effectively helps you feel calmer, more balanced, and emotionally strong. Lower stress allows you to be present, enjoy positive experiences, and maintain a greater sense of overall happiness.</p>", unsafe_allow_html=True)

    if st.button("Exercise Frequency"):
        st.markdown("<p style='color: fuchsia; '>Regular physical activity boosts endorphins and other “feel-good” chemicals that lift your mood and increase energy. Exercise also improves confidence, sleep, and mental clarity, all of which contribute to a happier, more vibrant life.</p>", unsafe_allow_html=True)

    if st.button("Social Media Use"):
        st.markdown("<p style='color: coral; '>Using social media mindfully can strengthen relationships, spark joy, and provide positive connection. However, excessive or passive scrolling may lead to comparison, anxiety, or feelings of isolation, so balance is key to maintaining happiness.</p>", unsafe_allow_html=True)

    if st.button("Screen Time"):
        st.markdown("<p style='color: orange; '>Mindful screen time can be a source of learning, entertainment, and connection, supporting your overall well-being. Too much passive screen use, however, can interfere with sleep, reduce social interaction, and lower your mood, so balancing online and offline life helps maximize happiness.</p>", unsafe_allow_html=True)

    st.subheader("🌻 Data Set")

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

    st.markdown("#### Statistical Summary")

    if st.button("Click Here to Generate Statistical Summary!"):
        st.dataframe(df.describe())


if page == "💐 The Garden":
    if st.button("Flower Shower"):
        st.markdown("""
        <style>
        /* Make sure container can show positioned elements */
        .stApp {
            overflow: hidden;
        }
        
        /* Full-screen layer for flowers */
        .falling-flowers {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;   /* UI still clickable */
            z-index: 0;
        }
        
        /* Each flower */
        .flower {
            position: absolute;
            top: -10%;
            font-size: 40px;
            animation-name: fall;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
            opacity: 0.9;
        }
        
        /* Animation */
        @keyframes fall {
            0% {
                transform: translateY(-100px) rotate(0deg);
                opacity: 0.6;
            }
            100% {
                transform: translateY(120vh) rotate(360deg);
                opacity: 1;
            }
        }
        </style>
        
        <div class="falling-flowers">
            <span class="flower" style="left: 10%; animation-duration: 10s; animation-delay: 0s;">🌸</span>
            <span class="flower" style="left: 20%; animation-duration: 14s; animation-delay: 2s;">🌺</span>
            <span class="flower" style="left: 30%; animation-duration: 12s; animation-delay: 1s;">🌼</span>
            <span class="flower" style="left: 40%; animation-duration: 9s;  animation-delay: 3s;">🌻</span>
            <span class="flower" style="left: 50%; animation-duration: 11s; animation-delay: 1s;">🌸</span>
            <span class="flower" style="left: 60%; animation-duration: 13s; animation-delay: .5s;">🌺</span>
            <span class="flower" style="left: 70%; animation-duration: 15s; animation-delay: 2.5s;">🌼</span>
            <span class="flower" style="left: 80%; animation-duration: 10s; animation-delay: 1.5s;">🌻</span>
            <span class="flower" style="left: 90%; animation-duration: 16s; animation-delay: 3s;">🌸</span>
        </div>
        """, unsafe_allow_html=True)


## Business Problem Presentation



## Data Summary Presentation
