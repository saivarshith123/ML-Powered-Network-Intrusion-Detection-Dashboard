# app.py
import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Page Configuration ---
st.set_page_config(page_title="Network Intrusion Detector", layout="wide")

# --- Load Saved Assets ---
try:
    model = joblib.load('nids_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    performance_df = pd.read_csv('model_performance.csv')
except FileNotFoundError:
    st.error("Model assets not found! Please run the 'train_and_evaluate.py' script first.")
    st.stop()

# --- Preprocessing Information ---
ORIGINAL_NUMERICAL_COLS = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]
ORIGINAL_CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# --- Main Page Layout ---

st.title("🌐 Network Intrusion Detection Dashboard")
st.write("Upload a CSV file to analyze network traffic and get a real-time prediction summary.")

# --- Create a two-column layout ---
col1, col2 = st.columns((1, 1))

# --- Column 1: File Uploader ---
with col1:
    st.subheader("📤 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file for analysis",
        type="csv",
        help="The CSV should contain network traffic features."
    )

# --- The rest of the page only appears AFTER a file is uploaded ---
if uploaded_file is not None:
    # --- Column 2: Model Performance Information (Now displayed only after upload) ---
    with col2:
        st.subheader("📄 Pre-Trained Model Information")
        with st.container(border=True):
            st.markdown("##### Algorithm Performance Comparison")
            st.dataframe(performance_df.style.highlight_max(axis=0, color='#2ECC71'))

            best_model = performance_df.loc[performance_df['Accuracy'].idxmax()]
            st.info(f"🏆 **Best Algorithm:** The **{best_model['Model']}** is used for predictions below, based on its highest accuracy of **{best_model['Accuracy']:.4f}**.")

    st.divider() # A visual separator for the results section

    # --- Results Section ---
    input_df = pd.read_csv(uploaded_file)

    # Preprocessing logic
    processed_df = pd.get_dummies(input_df, columns=ORIGINAL_CATEGORICAL_COLS)
    processed_df = processed_df.reindex(columns=model_columns, fill_value=0)
    num_cols_in_df = [col for col in ORIGINAL_NUMERICAL_COLS if col in processed_df.columns]
    processed_df[num_cols_in_df] = scaler.transform(processed_df[num_cols_in_df])

    # Prediction logic
    predictions = model.predict(processed_df)
    prediction_labels = ['Attack' if p == 1 else 'Normal' for p in predictions]
    
    # Create Tabs for a cleaner results display
    tab1, tab2, tab3 = st.tabs(["📊 Prediction Summary", "📋 Detailed Results", "📥 Input Data"])

    with tab1:
        st.subheader("📊 Prediction Summary")
        attack_count = prediction_labels.count('Attack')
        normal_count = prediction_labels.count('Normal')

        if attack_count > 0:
            st.warning(f"🚨 **Attack Detected!** The model identified potential threats in the uploaded data.")
        else:
            st.success(f"✅ **No Attacks Detected.** All connections were classified as normal.")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Connections Analyzed", len(prediction_labels))
        metric_col2.metric("Attacks Found", f"🚨 {attack_count}")
        metric_col3.metric("Normal Connections Found", f"✅ {normal_count}")

    with tab2:
        st.subheader("📋 Detailed Prediction Results")
        results_df = input_df.copy()
        results_df['Prediction'] = prediction_labels
        
        def highlight_attacks(s):
            return ['background-color: #FADBD8' if v == 'Attack' else '' for v in s]
            
        st.dataframe(results_df.style.apply(highlight_attacks, subset=['Prediction']), height=350, use_container_width=True)

    with tab3:
        st.subheader("📥 Input Data Preview")
        st.dataframe(input_df, height=350, use_container_width=True)

else:
    # This message is shown on the initial page load before any file is uploaded
    st.info("Please upload a file to begin analysis.")
