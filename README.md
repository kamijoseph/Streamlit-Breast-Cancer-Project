# 🧠 Streamlit Breast Cancer Prediction App

🔗 **Live Demo**: [Click here to try the deployed app](https://breast-cancer-prediction-f3wtpgbjzvpvohqysgbefx.streamlit.app/)

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-Logistic%20Regression-brightgreen.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)


This project aims to provide a lightweight, interpretable, and interactive tool for preliminary **breast cancer risk prediction** using a machine learning model trained on cell nucleus features. Built with **Streamlit**, the app enables users to simulate diagnostic scenarios and gain insight into how different features impact classification outcomes.

An interactive web application for classifying breast tumors as **Benign** or **Malignant** using a logistic regression model

The application is based on the **Breast Cancer Wisconsin (Diagnostic) dataset**, available on both [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29), containing measurements from digitized images of fine needle aspirate (FNA) of breast mass tissues.

> ⚠️ **Disclaimer**: This application is for **educational and demonstrative purposes only**. It is **not a diagnostic tool** and should **never be used as a substitute for professional medical advice or diagnosis**. Always consult qualified healthcare providers for medical concerns.



---

## 📁 Project Structure

Streamlit-Breast-Cancer-Project/
│

├── app/

│ └── main.py # Streamlit UI application

│

├── model/

│ ├── main.py # Data cleaning, preprocessing, training

│ └── model/

│ ├── model.pkl # Trained Logistic Regression model

│ └── scaler.pkl # Trained StandardScaler object

│

├── dataset/

│ └── breast_cancer.csv # UCI Breast Cancer dataset

│

├── requirements.txt # List of required libraries

└── README.md


---

## 🚀 Features

- 🔧 **Input Sliders** for 30 tumor cell nucleus measurements (mean, SE, and worst-case values)
- 📊 **Radar Chart Visualization** (Plotly-based) of input data
- 🤖 **Real-Time Prediction** of tumor class (Malignant/Benign) via logistic regression
- 🎯 **Probability Scores** for each class
- 💡 **Medical Caution Message** to contextualize model predictions

---

## ⚙️ Installation

### 🔐 Prerequisites
- Python ≥ 3.10
- Conda (recommended)
- Git


### 📦 Setup Guide

### 1. Clone this repository
```bash
git clone https://github.com/kamijoseph/Streamlit-Breast-Cancer-Project.git
cd Streamlit-Breast-Cancer-Project
```
### 2. Create a new Conda environment
```bash
conda create -n cancer-predictor python=3.10
```
### 3. Activate environment
```bash
conda activate cancer-predictor
```

### 4. Install dependencies
```bash
conda install --file requirements.txt
```

### 4. Run the Streamlit application
```bash
cd app
streamlit run main.py
```

---

## 🙋‍♂️ Questions or Feedback?

Feel free to [open an issue](https://github.com/kamijoseph/Streamlit-Breast-Cancer-Project/issues) or reach out if you have suggestions, questions, or ideas to improve this project.

---

### Built by [@kamijoseph](https://github.com/kamijoseph) using [Streamlit](https://streamlit.io/)
---
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://breast-cancer-prediction-f3wtpgbjzvpvohqysgbefx.streamlit.app//)
