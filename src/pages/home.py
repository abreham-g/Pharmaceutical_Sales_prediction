
import streamlit as st
import awesome_streamlit as ast



def write():
   
    with st.spinner("Loading Home ..."):
        st.title('Rossmann Pharmaceuticals')
        #st.image('../assets/ross.jpg', use_column_width=True)
        st.write(
            """
           The company was founded by Dirk Rossmann with its headquarters in Burgwedel near Hanover in Germany. The Rossmann family owns 60%, and the Hong Kong-based A.S. Watson Group 40% of the company.


            Dirk Rossmann GmbH (usual: Rossmann) is one of the largest drug store chains in Europe with around 56,200 employees and more than 4000 stores across Europe. In 2019 Rossmann had more than €10 billion turnover in Germany, Poland, Hungary, the Czech Republic, Turkey, Albania, Kosovo and Spain.

            The company logo consists of a red name and the symbol of a centaur integrated in the letter O: a mythical creature made of horse and man from Greek mythology, which symbolically stands for "Rossmann" (English: "Horse man"). The company's own brands have a small centaur symbol above the name.

            Since 2018, Rossmann has been publishing a sustainability report for the development of corporate climate protection activities.
                """
        )
