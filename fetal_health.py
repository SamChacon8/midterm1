# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the pre-trained models
dt_pickle = open('dt.pickle', 'rb') 
dt_clf = pickle.load(dt_pickle) 
dt_pickle.close()

rf_pickle = open('rf.pickle', 'rb') 
rf_clf = pickle.load(rf_pickle) 
rf_pickle.close()

ada_pickle = open('ada.pickle', 'rb') 
ada_clf = pickle.load(ada_pickle) 
ada_pickle.close()

svc_pickle = open('svc.pickle', 'rb') 
svc_clf = pickle.load(svc_pickle) 
svc_pickle.close()

fetal_health_df = pd.read_csv('fetal_health.csv')

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 

# Display an image
st.image('fetal_health_image.gif', width=600)

st.write('Utilize our advanced Machine Learning application to predict fetal health classifications.')

# Create a sidebar for data collection
st.sidebar.header('Fetal Health Features Input')
st.sidebar.write('Upload your data file')
uploaded_file = st.sidebar.file_uploader("", type=["csv"], help='File type must be csv')
st.sidebar.write('Example CSV file format:')  
st.sidebar.warning('⚠️ Ensure your data strictly follows the format outlined below.')
original_df = fetal_health_df.drop(columns=['fetal_health'])
st.sidebar.write(original_df.head(5))

# Choose model
model = st.sidebar.radio('Choose a Machine Learning model:', ['Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting'])

# Required columns for the model
required_columns = original_df.columns

if uploaded_file is not None:
    upload_df = pd.read_csv(uploaded_file)
    
    # Check if required columns are in uploaded data
    missing_cols = [col for col in required_columns if col not in upload_df.columns]
    if missing_cols:
        st.error(f" ❌ The uploaded file is missing the following required columns: {', '.join(missing_cols)}")
    else:
        st.success('✅ CSV file uploaded and verified successfully')
        
        # Prediction based on selected model
        if model == 'Random Forest':
            prediction = rf_clf.predict(upload_df)
            probabilities = rf_clf.predict_proba(upload_df).max(axis=1) * 100
            feature_svg = 'rf_feature_imp.svg'
            confusion_svg = 'rf_confusion_matrix.svg'
            classification = 'rf_class_report.csv'

        elif model == 'Decision Tree':
            prediction = dt_clf.predict(upload_df)
            probabilities = dt_clf.predict_proba(upload_df).max(axis=1) * 100
            feature_svg = 'dt_feature_imp.svg'
            confusion_svg = 'dt_confusion_matrix.svg'
            classification = 'dt_class_report.csv'
        elif model == 'AdaBoost':
            prediction = dt_clf.predict(upload_df)
            probabilities = ada_clf.predict_proba(upload_df).max(axis=1) * 100
            feature_svg = 'ada_feature_imp.svg'
            confusion_svg = 'ada_confusion_matrix.svg'
            classification = 'ada_class_report.csv'
        elif model == 'Soft Voting':
            prediction = svc_clf.predict(upload_df)
            probabilities = svc_clf.predict_proba(upload_df).max(axis=1) * 100
            feature_svg = 'sv_feature_imp.svg'
            confusion_svg = 'sv_confusion_matrix.svg'
            classification = 'sv_class_report.csv'
        
        def highlight_predictions(val):
            color = ''
            if val == 'Normal':
                color = 'background-color: lime'
            elif val == 'Suspect':
                color = 'background-color: yellow'
            elif val == 'Pathological':
                color = 'background-color: orange'
            return color

        # Append predictions to uploaded data and display
        upload_df['Fetal Health Prediction'] = prediction
        upload_df['Prediction Probability (%)'] = probabilities

        # Display predictions with custom styling for the prediction column
        st.write('Predictions:')
        styled_df = upload_df.style.applymap(highlight_predictions, subset=['Fetal Health Prediction'])
        st.dataframe(styled_df)

        st.subheader("Prediction Performance")
        tab1, tab2, tab3, = st.tabs([ "Feature Importance", "Confusion Matrix", "Classification Report"])

        # # Tab 1: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image(feature_svg)
            st.caption("Features used in this prediction are ranked by relative importance.")

        # # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image(confusion_svg)
            st.caption("Confusion Matrix of model predictions.")

        # # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv(classification, index_col = 0)
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")


else:
    st.warning('Please upload a CSV file to proceed.')
