# app.py

import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Load the Saved Model and Components ---
try:
    model = joblib.load('nids_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please run the 'train_and_save.py' script first.")
    st.stop()

# Define the original numerical and categorical columns for preprocessing
ORIGINAL_NUMERICAL_COLS = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
ORIGINAL_CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']


# --- Web App Interface ---
st.set_page_config(page_title="Network Intrusion Detector", layout="wide")
st.title("🌐 Network Intrusion Detection System")
st.write("Upload a CSV file with network traffic data to predict whether a connection is **Normal** or an **Attack**.")


# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # 1. Read the uploaded data
    input_df = pd.read_csv(uploaded_file)
    st.write("### Input Data Preview")
    st.dataframe(input_df.head())
    
    # Keep a copy of the original data for display
    display_df = input_df.copy()

    # 2. Preprocess the input data
    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df, columns=ORIGINAL_CATEGORICAL_COLS)
    
    # Align columns with the model's training columns
    # Create a new DataFrame with the model's columns and fill with 0
    processed_df = pd.DataFrame(columns=model_columns)
    processed_df = pd.concat([processed_df, input_df])
    processed_df.fillna(0, inplace=True)
    
    # Ensure the order of columns is the same as during training
    processed_df = processed_df[model_columns]
    
    # Scale numerical features
    processed_df[ORIGINAL_NUMERICAL_COLS] = scaler.transform(processed_df[ORIGINAL_NUMERICAL_COLS])
    
    # 3. Make predictions
    predictions = model.predict(processed_df)
    prediction_labels = ['Normal' if p == 0 else 'Attack' for p in predictions]
    
    # 4. Display results
    display_df['Prediction'] = prediction_labels
    
    st.write("### Prediction Results")
    st.dataframe(display_df)
    
    # Add styling for the results
    def highlight_attack(s):
        return ['background-color: #ffcccc' if v == 'Attack' else '' for v in s]
    
    st.write("### Styled Results")
    st.dataframe(display_df.style.apply(highlight_attack, subset=['Prediction']))

else:
    st.info("Awaiting for CSV file to be uploaded.")
