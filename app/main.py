
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

#data cleaning function
def cleaned_data():
    #importing the dataset
    data = pd.read_csv("dataset/breast_cancer.csv")

    # dropping unnamed: 32 column and the id column
    data = data.drop(["Unnamed: 32", "id"], axis=1)

    # encoding categorical variable: {diagnosis} into numerical values
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    return data

# sidebar function
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

    # empty input dictionary
    input_dict = {}

    # creating the sliders one by one
    for  label, key in slider_labels:

        # minimum , mean and maximum values of the sliders
        min_val = float(0)
        max_val = float(data[key].max())
        mean_val = float(data[key].mean())

        # sliders
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    
    # return the populated input dictionary
    return input_dict

# scaler function to scale the input for the radar chart
def scaled_values(input_dict):
    data = cleaned_data()
    X= data.drop(["diagnosis"], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

# radar chart function
def build_radar_chart(input_data):

    input_data = scaled_values(input_data)

    categories = [
        'Radar', 'Texture', 'Perimeter', 'Area', 'Smoothness',
        'Compactness', 'Concavity', 'Concave Points',
        'Symmetry', 'Fractal Dimension'
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))
    fig.update_layout(
        width=800,
        height=700,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )
    
    return fig

# prediction column functiom
def predictions_column(input_data):

    #importing the model and scaler from our training
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    # converting input data to a numpy array
    input_asarray = np.array(list(input_data.values()))

    # reshaping the input array to 2d
    input_reshaped = input_asarray.reshape(1, -1)

    # scaling the input data using the imported scaler
    input_scaled = scaler.transform(input_reshaped)

    # Prediction using the imported model
    prediction = model.predict(input_scaled)

    st.subheader("Cell Cluster Prediction")
    st.divider()
    st.write("The cell cluster is:")

    # checking if prediction is malignant or benign
    if (prediction[0] == 1):
        st.write("**MALIGNANT TUMOUR**")
    else:
        st.write("**BENIGN TUMOUR**")

    # probability for both classes
    benign_prob = model.predict_proba(input_scaled)[0][0]
    malignant_prob = model.predict_proba(input_scaled)[0][1]

    st.divider()

    # write the probabilities to the interface
    st.write("Benign Probability:", benign_prob)
    st.write(f"Malignant Probability:", malignant_prob)

    st.divider()
    
    # Awareness
    st.write("This project can assist medical professional in making a diagnosis but is never a substitute for professional and proper medical diagnosis")
 
# main function
def main():

    # page configuration
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":male-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # creating input variable by calling the sidebar fnction to create it
    input_data = add_sidebar()

    # title and introduction container
    with st.container():
        st.title("Breast Cancer Prediction Model")
        st.write("The prediction model works by taking different inputs through the slider or by connecting your cytosis lab results. The model classifies breast lumps, mass or tumour into either Malignant or Benign with Malignant being cancerous tumour and the opposite for Benign")
    
    # columns initialization
    col1, col2 = st.columns([4, 1], border=True )

    # columns application
    with col1:
        radar_chart = build_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        predictions_column(input_data)

# running the app
if __name__ == "__main__":
    main()