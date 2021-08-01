# Libraries
import streamlit as st
import pandas as pd 
import awesome_streamlit as ast

def write():
    
    with st.spinner("Loading Data ..."):
        st.title('Data description  ')
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('src/pages/train.csv', na_values=na_value)
        store = pd.read_csv('src/pages/store.csv', na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        full_train = full_train.set_index('Store')
        st.write(full_train.sample(20))
