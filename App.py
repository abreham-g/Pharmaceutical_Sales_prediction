import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(pagetitle="Pharmacutical Sales prediction", layout="wide")



#def loadData():
#    print("loading started ")
#    dataframe = pd.read_csv("./features/my_features.csv")
#    print("loading complleted")
#    return dataframe

#dataframe =  loadData()

st.sidebar.title("Pharmacutical-sales-Prediction")
option = st.sidebar.selectbox('select result',('Raw Data',
'Plots',' prediction','insight'))


#st.title(option)
