
import streamlit as st
import awesome_streamlit as ast



def write():
   
    with st.spinner("Loading Home ..."):
        st.title('Rossmann Pharmaceuticals Salse prediction')
        #st.image('../assets/ross.jpg', use_column_width=True)
        st.write(
            """
            Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality.
            With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

            The company is guided by the following virtues:
            - **Practical Wisdom.**
            - **Moral Rule**,  **Moral Virtue** and **Moral Sense**.
            - **Personal Virtue**.
                """
        )