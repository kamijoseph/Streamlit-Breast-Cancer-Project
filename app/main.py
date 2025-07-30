
import streamlit as st
import pandas as pd
import pickle

#data cleaning function
def cleaned_data():
    #importing the dataset
    data = pd.read_csv("dataset/breast_cancer.csv")

    # dropping unnamed: 32 column and the id column
    data = data.drop(["Unnamed: 32", "id"], axis=1)

    # encoding categorical variable: {diagnosis} into numerical values
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    return data


def add_sidebar():

    # creating the sidebar
    st.sidebar.header("Nuclei Values")

    # importing and cleaning the data
    data = cleaned_data()

    # initializing the slider labels
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # creating the sliders one by one
    for  label, key in slider_labels:

        # minimum , mean and maximum values of the sliders
        min_val = float(0)
        max_val = float(data[key].max())
        mean_val = float(data[key].mean())

        # sliders
        st.sidebar.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

# main function
def main():

    # page configuration
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":male-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # calling the sidebar fnction to create it
    add_sidebar()

    # title and introduction container
    with st.container():
        st.title("Breast Cancer Prediction Model")
        st.write("The prediction model works by taking different inputs through the slider or by connecting your cytosis lab results. The model classifies breast lumps, mass or tumour into either Malignant or Benign with Malignant being cancerous tumour and the opposite for Benign")
    
    # columns initialization
    col1, col2 = st.columns([4, 1])

    # columns application
    with col1:
        st.write("Column 1")
    with col2:
        st.write("Column 2")

# running the app
if __name__ == "__main__":
    main()