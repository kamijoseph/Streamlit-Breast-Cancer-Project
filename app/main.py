
import streamlit as st

def main():
    # page configuration
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":male-scientist:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.write("Doctor Here")



if __name__ == "__main__":
    main()