import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
 
bag_clf = joblib.load('bag_clf')
processor=joblib.load("processor")

lung= pd.read_csv("lung_cancer.csv")
lung.columns=lung.columns.str.lower()
lung=pd.get_dummies(lung, columns=["lung_cancer"], drop_first=True) 
lung.columns=lung.columns.str.replace("lung_cancer_YES", "cancer_risk")
lung.columns=lung.columns.str.replace(" ", "_")
 
st.title("Prediction of lung cancer occurrance")
st.caption("About the app")
st.write("""Simple approach of lung cancer risk occurance probability according to some early reported signs""")
st.write("""Options:1=no, 2= yes""")

# app icon
icon = Image.open('l.cancer.jpg')
st.image(icon, use_column_width= True)

gender=st.sidebar.selectbox('gender',lung["gender"].unique())
age= st.slider("age",lung["age"].min(), lung["age"].max())
smoking= st.sidebar.selectbox("smoking",lung["smoking"].unique())
yellow_fingers= st.sidebar.selectbox("yellow_fingers",lung["yellow_fingers"].unique())
anxiety= st.sidebar.selectbox('anxiety', lung["anxiety"].unique())
peer_pressure= st.sidebar.selectbox('peer_pressure',lung["peer_pressure"].unique())
chronic_disease= st.sidebar.selectbox('chronic_disease',lung["chronic_disease"].unique())
fatigue_= st.sidebar.selectbox('fatigue_',lung["fatigue_"].unique())
allergy_= st.sidebar.selectbox('allergy_', lung["allergy_"].unique())
wheezing= st.sidebar.selectbox('wheezing',lung["wheezing"].unique())
alcohol_consuming= st.sidebar.selectbox('alcohol_consuming',lung["alcohol_consuming"].unique())
coughing= st.sidebar.selectbox('coughing',lung["coughing"].unique())
shortness_of_breath= st.sidebar.selectbox('shortness_of_breath',lung["shortness_of_breath"].unique())
swallowing_difficulty= st.sidebar.selectbox('swallowing_difficulty',lung["swallowing_difficulty"].unique())
chest_pain= st.sidebar.selectbox('chest_pain',lung["chest_pain"].unique())

#dict
user_data={
    "gender":gender, 
    "age":age,
    "smoking": smoking,
    "yellow_fingers": yellow_fingers,
    "anxiety":anxiety,
    "peer_pressure":peer_pressure,
    "chronic_disease": chronic_disease,
    "fatigue_": fatigue_,
    "allergy_": allergy_,
    "wheezing": wheezing,
    "alcohol_consuming":alcohol_consuming,
    "coughing": coughing,
    "shortness_of_breath": shortness_of_breath,
    "swallowing_difficulty": swallowing_difficulty,
    "chest_pain": chest_pain
}
        
         
lung_parm=pd.DataFrame(user_data, index=[0]) 
lung_parm_ready=processor.transform(lung_parm)
st.button("Probability: Neg-Pos")
lung_pred= bag_clf.predict_proba(lung_parm_ready)*100

#display
st.markdown("""# {}""".format(lung_pred))         
