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
 
st.title("Prediction of lung cancer occurrence")
st.caption("About the app")
st.write("""Simple approach of prediction of lung cancer risk occurrence according to some early reported signs""")
 

# app icon
icon = Image.open('l.cancer.jpg')
st.image(icon, use_column_width= True)

gender=st.sidebar.selectbox('gender',lung["gender"].unique())
age= st.slider("age",lung["age"].min(), lung["age"].max())
smoking= st.sidebar.selectbox("smoking",("Yes","NO"))
yellow_fingers= st.sidebar.selectbox("yellow_fingers",("Yes","NO"))
anxiety= st.sidebar.selectbox('anxiety', ("Yes","NO"))
peer_pressure= st.sidebar.selectbox('peer_pressure',("Yes","NO"))
chronic_disease= st.sidebar.selectbox('chronic_disease',("Yes","NO"))
fatigue_= st.sidebar.selectbox('fatigue_',("Yes","NO"))
allergy_= st.sidebar.selectbox('allergy_', ("Yes","NO"))
wheezing= st.sidebar.selectbox('wheezing',("Yes","NO"))
alcohol_consuming= st.sidebar.selectbox('alcohol_consuming',("Yes","NO"))
coughing= st.sidebar.selectbox('coughing',("Yes","NO"))
shortness_of_breath= st.sidebar.selectbox('shortness_of_breath',("Yes","NO"))
swallowing_difficulty= st.sidebar.selectbox('swallowing_difficulty',("Yes","NO"))
chest_pain= st.sidebar.selectbox('chest_pain',lung["chest_pain"],("Yes","NO"))
def cat_resp (response):
 if response=="Yes":
  return 2
 else:
  return 1
#dict
user_data={
    "gender":cat_resp(gender), 
    "age":cat_resp(age),
    "smoking":cat_resp(smoking),
    "yellow_fingers":cat_resp(yellow_fingers),
    "anxiety":cat_resp(anxiety),
    "peer_pressure":cat_resp(peer_pressure),
    "chronic_disease": cat_resp(chronic_disease),
    "fatigue_":cat_resp(fatigue_),
    "allergy_": cat_resp(allergy_),
    "wheezing": cat_resp(wheezing),
    "alcohol_consuming":cat_resp(alcohol_consuming),
    "coughing": cat_resp(coughing),
    "shortness_of_breath": cat_resp(shortness_of_breath),
    "swallowing_difficulty": cat_resp(swallowing_difficulty),
    "chest_pain": cat_resp(chest_pain)
}
        
         
lung_parm=pd.DataFrame(user_data, index=[0]) 
lung_parm_ready=processor.transform(lung_parm)
lung_pred= bag_clf.predict_proba(lung_parm_ready)[0][1]*100

#display
if st.button("Probability of occurrence:"):
  st.markdown("""# {} %""".format(lung_pred))         
