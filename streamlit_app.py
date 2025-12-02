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


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer




st.sidebar.markdown("<h1 style='color: lightsalmon; text-align: center; '>🌻 Happy Place 🌻</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: hotpink; text-align: center; '>Your journey to joy!</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: lightsalmon; text-align: center; '>💐🌺🌷🌻🪷🪻🌸</h1>", unsafe_allow_html=True)

st.image("OpenAI.jpeg", use_container_width=True)
st.markdown("<h1 style='color: lightsalmon; text-align: center;'>🌻 Happy Place 🌻</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: hotpink; text-align: center;'>Notice what you need. Nurture how you feel.</p>", unsafe_allow_html=True)

df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")
### Intro Page
page = st.sidebar.selectbox("Select Page",["🌺 Introduction","🪻 Data Visualization","🌸 Modeling & Prediction", "💐 The Garden", ])
##Introduction Page
if page == "🌺 Introduction":

    st.image("hugol-halpingston-4OyLq2yN9u0-unsplash.jpg", use_container_width=True)
    st.markdown("<h2 style='color: hotpink; text-align: center;'>Introduction</p>", unsafe_allow_html=True)


    st.subheader("🌸 Our Mission")
    st.markdown("<p style='color: hotpink; '>Sometimes you don’t even know why you’re feeling low, only that something needs to change. This app helps you pause, reflect, and uncover the patterns beneath your mood by organizing your daily habits—sleep, social media use, movement, and more—into clear, meaningful insights. As you start to see how your choices shape your well-being, you realize you have far more control than you thought.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: hotpink; '>Take a moment, search inward, and let your data gently guide you toward a happier, more intentional life!</p>", unsafe_allow_html=True)


    st.subheader("🌺 Look Inward")
    st.markdown("<p style='color: hotpink; '>Happiness is less complicated than you may think. Let's break down the factors that contribute to your mental health.</p>", unsafe_allow_html=True)

    if st.button("Sleep Quality"):
        st.markdown("<p style='color: deeppink; '>Quality sleep is essential for feeling energized, happy, and mentally clear. When you sleep well, your mood improves, your focus sharpens, and you’re better able to enjoy life’s moments and handle challenges with resilience.</p>", unsafe_allow_html=True)

    if st.button("Stress Level"):
        st.markdown("<p style='color: salmon; '>Managing stress effectively helps you feel calmer, more balanced, and emotionally strong. Lower stress allows you to be present, enjoy positive experiences, and maintain a greater sense of overall happiness.</p>", unsafe_allow_html=True)

    if st.button("Exercise Frequency"):
        st.markdown("<p style='color: fuchsia; '>Regular physical activity boosts endorphins and other “feel-good” chemicals that lift your mood and increase energy. Exercise also improves confidence, sleep, and mental clarity, all of which contribute to a happier, more vibrant life.</p>", unsafe_allow_html=True)

    if st.button("Social Media Use"):
        st.markdown("<p style='color: coral; '>Using social media mindfully can strengthen relationships, spark joy, and provide positive connection. However, excessive or passive scrolling may lead to comparison, anxiety, or feelings of isolation, so balance is key to maintaining happiness.</p>", unsafe_allow_html=True)

    if st.button("Screen Time"):
        st.markdown("<p style='color: orange; '>Mindful screen time can be a source of learning, entertainment, and connection, supporting your overall well-being. Too much passive screen use, however, can interfere with sleep, reduce social interaction, and lower your mood, so balancing online and offline life helps maximize happiness.</p>", unsafe_allow_html=True)

    st.subheader("🌻 Data Set")
    st.markdown("<p style='color: hotpink; '>Here’s a detailed breakdown of our dataset, highlighting key factors that influence mental health and happiness.</p>", unsafe_allow_html=True)

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









if page == "🪻 Data Visualization":
    looker_url = "https://lookerstudio.google.com/embed/reporting/78ce404a-a5e0-4180-8739-dcbac8f7c5bb/page/OUwgF"
    components.iframe(src=looker_url, width=1000, height=600)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    
    X = df[['Age', 'Gender', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
           'Stress_Level(1-10)', 'Days_Without_Social_Media',
           'Exercise_Frequency(week)', 'Social_Media_Platform']]
    y = df['Happiness_Index(1-10)']
    
    
    categorical_features = ['Gender', 'Social_Media_Platform']
    numeric_features = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
                       'Stress_Level(1-10)', 'Days_Without_Social_Media', 'Exercise_Frequency(week)']
    
    
    preprocessor = ColumnTransformer(
       transformers=[
           ('cat', OneHotEncoder(), categorical_features)
       ],
       remainder='passthrough'  # keep numeric features as-is
    )
    
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])
    model.fit(X, y)
    
    
    st.subheader("🌼 Happiness Index Predictor")
    
    
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", df['Gender'].unique())
    screen_time = st.slider("Daily Screen Time (hrs)", 0, 12, 3)
    sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    days_without_social_media = st.slider("Days Without Social Media", 0, 30, 0)
    exercise_freq = st.slider("Exercise Frequency (per week)", 0, 14, 3)
    platform = st.selectbox("Social Media Platform", df['Social_Media_Platform'].unique())
    
    
    input_df = pd.DataFrame({
       'Age': [age],
       'Gender': [gender],
       'Daily_Screen_Time(hrs)': [screen_time],
       'Sleep_Quality(1-10)': [sleep_quality],
       'Stress_Level(1-10)': [stress_level],
       'Days_Without_Social_Media': [days_without_social_media],
       'Exercise_Frequency(week)': [exercise_freq],
       'Social_Media_Platform': [platform]
    })




    predicted_happiness = model.predict(input_df)[0]
    st.markdown("<h4 style='color: hotpink; '>Your Personal Predicted Happiness Score:</p>", unsafe_allow_html=True)
    st.subheader(f"Predicted Happiness Index: {predicted_happiness:.2f}/10")




    target_happiness = min(predicted_happiness + 1, 10)  # aim for +1 happiness, max 10
    additional_days_needed = int((target_happiness - predicted_happiness) / 0.1)
   # st.markdown("<h4 style='color: salmon; '>How many days should you skip social media (+1 Happiness Point)?:</p>", unsafe_allow_html=True)
    if st.button("How many days should you skip social media (+1 Happiness Point)?"):
        st.markdown(f"+1 Happiness: {additional_days_needed} days")












if page == "🌸 Modeling & Prediction":
    import streamlit as st
    import pandas as pd
    import seaborn as sns       
    import sklearn as sk         
    import matplotlib.pyplot as plt     
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    st.subheader("🌷 Mental Health & Social Media - ML Models")
    st.markdown("<p style='color: coral; '>Compare two models: Logistic Regression vs Random Forest</p>", unsafe_allow_html=True)

    df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")
    TARGET_COL = "Happiness_Index(1-10)"
    if TARGET_COL not in df.columns:
       st.error(f"Target column '{TARGET_COL}' not found in dataset. "
                f"Available columns: {list(df.columns)}")
       st.stop()
    df = df.dropna(subset=[TARGET_COL])
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    st.markdown("<h3 style='color: pink; '>Target Distribution</p>", unsafe_allow_html=True)
    st.write(y.value_counts())
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    st.markdown("<h3 style='color: orange; '>Feature Types</p>", unsafe_allow_html=True)
    st.write("**Numeric features:**", numeric_cols)
    st.write("**Categorical features:**", categorical_cols)
    numeric_pipeline = Pipeline(steps=[
       ("imputer", SimpleImputer(strategy="median")),
       ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
       ("imputer", SimpleImputer(strategy="most_frequent")),
       ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
       transformers=[
           ("num", numeric_pipeline, numeric_cols),
           ("cat", categorical_pipeline, categorical_cols)
       ]
    )
    log_model = Pipeline(steps=[
       ("preprocess", preprocessor),
       ("clf", LogisticRegression(max_iter=1000))
    ])
    rf_model = Pipeline(steps=[
       ("preprocess", preprocessor),
       ("clf", RandomForestClassifier(
           n_estimators=200,
           random_state=42
       ))
    ])
    MODELS = {
       "Logistic Regression": log_model,
       "Random Forest": rf_model
    }
    X_train, X_test, y_train, y_test = train_test_split(
       X, y,
       test_size=0.2,
       random_state=42,
       stratify=y
    )
    st.markdown("### Choose a model")
    model_choice = st.selectbox(
       "Select the model to train:",
       list(MODELS.keys())
    )
    def run_model(model_name):
       model = MODELS[model_name]
       st.write(f"### Training: {model_name}")
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       # ---- Accuracy ----
       acc = accuracy_score(y_test, y_pred)
       st.write(f"**Accuracy:** {acc:.3f}")
       # ---- Classification Report ----
       st.write("**Classification Report:**")
       st.text(classification_report(y_test, y_pred))
       # ---- Confusion Matrix ----
       st.write("**Confusion Matrix:**")
       classes = np.unique(y_test)
       cm = confusion_matrix(y_test, y_pred, labels=classes)
       cm_df = pd.DataFrame(cm, index=classes, columns=classes)
       fig, ax = plt.subplots()
       sns.heatmap(
           cm_df,
           annot=True,
           fmt='d',
           cmap='Purples',
           ax=ax
       )
       ax.set_xlabel("Predicted Label")
       ax.set_ylabel("True Label")
       st.pyplot(fig)
       return model
    if st.button("💐 Run Selected Model"):
        trained_model = run_model(model_choice)
   
        
        st.subheader("🏋️ Weights & Biases Experiment Tracking")
        st.info("Click the button below to view your dashboard:")

        st.link_button("🔗 Open W&B Dashboard", "https://wandb.ai/mrw9818-new-york-university/three_models_demo?nw=nwusermrw9818")
            



        

 












if page == "💐 The Garden":
    st.image("_.jpeg", use_container_width=True)
    st.markdown("<h2 style='color: hotpink; text-align: center;'>The Garden</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: coral; text-align: center;'>You're safe and loved here. Feel free to use these mental health resources.</p>", unsafe_allow_html=True)

    ## Mental Healthcare Links
    st.subheader("🪷 Mental Health Resources")


    st.markdown("##### Befrienders Worldwide")
    st.markdown("<p style='color: salmon;'>Global network offering confidential emotional support in many countries</p>", unsafe_allow_html=True)
    st.markdown('<a href="https://befrienders.org befrienders.org+1" target="_blank">Visit Befrienders Worldwide</a>', unsafe_allow_html=True)

    st.markdown("##### Find A Helpline")
    st.markdown("<p style='color: deeppink; '>Free directory connecting you to hotlines and crisis support services in 130+ countries.</p>", unsafe_allow_html=True)
    st.markdown('<a href="https://findahelpline.com/ Samaritans+1" target="_blank">Visit Find A Helpline</a>', unsafe_allow_html=True)

    st.markdown("##### Open Counseling")
    st.markdown("<p style='color: orange; '>Provides a global directory for mental‑health services, affordable therapy, and crisis hotlines around the world.</p>", unsafe_allow_html=True)
    st.markdown('<a href="https://caps.arizona.edu/international-help-lines">Visit Open Counseling</a>', unsafe_allow_html=True)

    st.markdown("##### International Association for Suicide Prevention (IASP)")
    st.markdown("<p style='color: fuchsia; '>Worldwide information on crisis centres and suicide‑prevention resources.</p>", unsafe_allow_html=True)
    st.markdown('<a href="https://dev.new.iasp.info/crisis-centres-helplines/ IASP+1" target="_blank">Visit International Association for Suicide Prevention</a>', unsafe_allow_html=True)

    st.markdown("##### Samaritans")
    st.markdown("<p style='color: coral;'>Offers emotional support and crisis helplines internationally (beyond their UK/Ireland origins), often via email or phone.</p>", unsafe_allow_html=True)
    st.markdown('<a href="hhttps://www.samaritans.org/how-we-can-help/if-youre-having-difficult-time/other-sources-help/ Samaritans+1" target="_blank">Visit Lifeline International</a>', unsafe_allow_html=True)

    st.markdown("##### Lifeline International")
    st.markdown("<p style='color: pink; '>Global umbrella or reference for a number of crisis hotlines and mental health support services worldwide.</p>", unsafe_allow_html=True)
    st.markdown('<a href="https://lifeline-intl.findahelpline.com/" target="_blank">Visit Lifeline International</a>', unsafe_allow_html=True)




    ## Breath Break

    st.subheader("💐 For a little extra joy in your day!")

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
        

