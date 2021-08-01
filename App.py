# import the child scripts
import streamlit as st
import awesome_streamlit as ast
import src.pages.home
import src.pages.data 
import src.pages.rawplots
import src.pages.pred

ast.core.services.other.set_logging_format()

# create the pages
PAGES = {
    "Home": src.pages.home,
    "Data":src.pages.data,
    "Data visualisations": src.pages.rawplots,
    "Predictions": src.pages.pred,
}


# render the pages
def main():
   
    st.sidebar.title("Salse Prediction")
    selection = st.sidebar.selectbox("Select", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
    if selection =="Home":
        st.sidebar.title("About")
        st.sidebar.info(
        """
        This App is an end-to-end product that enables the Rosemann pharmaceutical company to 
        view predictions on sales across their stores and 6 weeks ahead of time and the trends expected.
"""
    )

# run it
if __name__ == "__main__":
    main()
#st.title(option)
