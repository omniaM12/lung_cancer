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
smoking= st.sidebar.selectbox("smoking", ('Yes', 'No'))
yellow_fingers= st.sidebar.selectbox("yellow_fingers",('Yes', 'No'))
anxiety= st.sidebar.selectbox('anxiety', ('Yes', 'No'))
peer_pressure= st.sidebar.selectbox('peer_pressure',('Yes', 'No'))
chronic_disease= st.sidebar.selectbox('chronic_disease',('Yes', 'No'))
fatigue_= st.sidebar.selectbox('fatigue_',('Yes', 'No'))
allergy_= st.sidebar.selectbox('allergy_', ('Yes', 'No'))
wheezing= st.sidebar.selectbox('wheezing',('Yes', 'No'))
alcohol_consuming= st.sidebar.selectbox('alcohol_consuming',('Yes', 'No'))
coughing= st.sidebar.selectbox('coughing',('Yes', 'No'))
shortness_of_breath= st.sidebar.selectbox('shortness_of_breath',('Yes', 'No'))
swallowing_difficulty= st.sidebar.selectbox('swallowing_difficulty',('Yes', 'No'))
chest_pain= st.sidebar.selectbox('chest_pain',('Yes', 'No'))

def convert_response(response):
 if response == 'Yes':
  return 2
 else:
  return 1
#dict
user_data={
    "gender":gender, 
    "age":age,
    "smoking": convert_response(smoking),
    "yellow_fingers": convert_response(yellow_fingers),
    "anxiety":convert_response(anxiety),
    "peer_pressure":convert_response(peer_pressure),
    "chronic_disease": convert_response(chronic_disease),
    "fatigue_": convert_response(fatigue_),
    "allergy_": convert_response(allergy_),
    "wheezing": convert_response(wheezing),
    "alcohol_consuming":convert_response(alcohol_consuming),
    "coughing": convert_response(coughing),
    "shortness_of_breath": convert_response(shortness_of_breath),
    "swallowing_difficulty": convert_response(swallowing_difficulty),
    "chest_pain": convert_response(chest_pain)
}
        
         
lung_parm=pd.DataFrame(user_data, index=[0]) 
lung_parm_ready=processor.transform(lung_parm)
lung_pred= bag_clf.predict_proba(lung_parm_ready)[0]*100

#display
if st.button("Probability: Neg-Pos"):
 st.markdown("""# {}""".format(lung_pred))         
