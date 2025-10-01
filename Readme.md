🛡️ ML-Powered Network Intrusion Detection Dashboard
====================================================

\[!\[Python Version\]\[python-shield\]\]\[python-url\]\[!\[Streamlit Version\]\[streamlit-shield\]\]\[streamlit-url\]\[!\[Scikit-learn Version\]\[sklearn-shield\]\]\[sklearn-url\]\[!\[License: MIT\]\[license-shield\]\]\[license-url\]

> A comprehensive and interactive web dashboard for real-time analysis and detection of network intrusions using classical machine learning algorithms on the benchmark NSL-KDD dataset.

This project provides an end-to-end solution, from data preprocessing and model training to a fully functional and user-friendly Streamlit application that can classify network traffic as 'Normal' or 'Attack'.

📖 Table of Contents
--------------------

*   [1\. Project Overview](https://www.google.com/search?q=#-1-project-overview)
    
    *   [1.1. The Problem](https://www.google.com/search?q=#11-the-problem)
        
    *   [1.2. The Solution](https://www.google.com/search?q=#12-the-solution)
        
    *   [1.3. Core Features](https://www.google.com/search?q=#13-core-features)
        
*   [2\. System Architecture](https://www.google.com/search?q=#-2-system-architecture)
    
    *   [2.1. Phase 1: Offline Training Pipeline](https://www.google.com/search?q=#21-phase-1-offline-training-pipeline)
        
    *   [2.2. Phase 2: Live Prediction Dashboard](https://www.google.com/search?q=#22-phase-2-live-prediction-dashboard)
        
*   [3\. Technology Stack](https://www.google.com/search?q=#-3-technology-stack)
    
    *   [3.1. Core Technologies](https://www.google.com/search?q=#31-core-technologies)
        
    *   [3.2. Machine Learning & Data Science](https://www.google.com/search?q=#32-machine-learning--data-science)
        
    *   [3.3. Web Development & UI](https://www.google.com/search?q=#33-web-development--ui)
        
*   [4\. Dataset Deep Dive: NSL-KDD](https://www.google.com/search?q=#-4-dataset-deep-dive-nsl-kdd)
    
    *   [4.1. Dataset Origin](https://www.google.com/search?q=#41-dataset-origin)
        
    *   [4.2. Feature Categories](https://www.google.com/search?q=#42-feature-categories)
        
    *   [4.3. Attack Categories](https://www.google.com/search?q=#43-attack-categories)
        
*   [5\. Methodology and ML Pipeline](https://www.google.com/search?q=#-5-methodology-and-ml-pipeline)
    
    *   [5.1. Data Preprocessing](https://www.google.com/search?q=#51-data-preprocessing)
        
    *   [5.2. Model Selection](https://www.google.com/search?q=#52-model-selection)
        
*   [6\. Installation and Local Setup](https://www.google.com/search?q=#-6-installation-and-local-setup)
    
    *   [6.1. Prerequisites](https://www.google.com/search?q=#61-prerequisites)
        
    *   [6.2. Step-by-Step Installation](https://www.google.com/search?q=#62-step-by-step-installation)
        
*   [7\. How to Use the Dashboard](https://www.google.com/search?q=#-7-how-to-use-the-dashboard)
    
    *   [7.1. Running the Application](https://www.google.com/search?q=#71-running-the-application)
        
    *   [7.2. Performing an Analysis](https://www.google.com/search?q=#72-performing-an-analysis)
        
*   [8\. Performance and Results](https://www.google.com/search?q=#-8-performance-and-results)
    
    *   [8.1. Performance Metrics](https://www.google.com/search?q=#81-performance-metrics)
        
    *   [8.2. Comparative Analysis](https://www.google.com/search?q=#82-comparative-analysis)
        
*   [9\. Future Work and Roadmap](https://www.google.com/search?q=#-9-future-work-and-roadmap)
    
    *   [9.1. Short-Term Goals](https://www.google.com/search?q=#91-short-term-goals)
        
    *   [9.2. Long-Term Vision](https://www.google.com/search?q=#92-long-term-vision)
        
*   [10\. Contributing](https://www.google.com/search?q=#-10-contributing)
    
*   [11\. License](https://www.google.com/search?q=#-11-license)
    
*   [12\. Acknowledgments](https://www.google.com/search?q=#-12-acknowledgments)
    
*   [13\. Project Team](https://www.google.com/search?q=#-13-project-team)
    

✨ 1. Project Overview
---------------------

### 1.1. The Problem

*   **Evolving Cyber Threats:** Traditional security systems struggle to keep up with new, sophisticated "zero-day" attacks.
    
*   **Limitations of Signature-Based NIDS:** These systems rely on a database of known attack patterns and are blind to novel threats.
    
*   **Need for Intelligent Systems:** There is a critical need for adaptive, anomaly-based detection systems that can identify threats without prior knowledge.
    

### 1.2. The Solution

*   **Anomaly-Based Detection:** This project builds a Network Intrusion Detection System (NIDS) that first learns what "normal" network traffic looks like.
    
*   **Machine Learning at the Core:** It leverages classical ML algorithms to autonomously learn these patterns and flag any significant deviations as potential attacks.
    
*   **Practical Application:** The project culminates in a user-friendly Streamlit dashboard that translates complex ML predictions into actionable insights for a user.
    

### 1.3. Core Features

*   **End-to-End ML Pipeline:** A complete, reproducible workflow from raw data to a deployed model.
    
*   **Empirical Model Comparison:** Rigorously evaluates and compares the performance of KNN, LDA, and SVM.
    
*   **Interactive Web Interface:** A modern and intuitive dashboard built with Streamlit for real-time analysis.
    
*   **Dynamic UI:** The interface adapts after file upload to present a clean and organized summary of results.
    
*   **Detailed Reporting:** Provides both a high-level summary of threats and a detailed, row-by-row breakdown of predictions.
    

🏗️ 2. System Architecture
--------------------------

The project architecture is logically separated into two distinct phases.

### 2.1. Phase 1: Offline Training Pipeline

*   **Purpose:** To perform all the heavy computational work of data processing and model training a single time.
    
*   **Process:** The train\_and\_evaluate.py script ingests the raw NSL-KDD datasets, applies a rigorous preprocessing pipeline, trains all three ML models, evaluates their performance on a test set, and finally saves the best model and all necessary assets.
    
*   **Output:** A set of serialized files (.pkl and .csv) that are ready for use by the live application.
    

### 2.2. Phase 2: Live Prediction Dashboard

*   **Purpose:** To provide a user-facing, interactive tool for on-demand network traffic analysis.
    
*   **Process:** The app.py script (the Streamlit dashboard) loads the assets generated in Phase 1 on startup. When a user uploads a CSV file, the app uses the loaded scaler and model to preprocess the new data, make predictions, and render the results in a user-friendly format.
    
*   **Output:** An interactive web page with summaries, tables, and color-coded results.
    

🛠️ 3. Technology Stack
-----------------------

### 3.1. Core Technologies

*   **Programming Language:** Python
    
*   **Version Control:** Git & Git LFS (for handling large model files)
    

### 3.2. Machine Learning & Data Science

*   **Core ML Library:** Scikit-learn
    
*   **Data Manipulation:** Pandas, NumPy
    
*   **Model Serialization:** Joblib
    

### 3.3. Web Development & UI

*   **Web Framework:** Streamlit
    

📊 4. Dataset Deep Dive: NSL-KDD
--------------------------------

### 4.1. Dataset Origin

*   **Source:** A refined and improved version of the original KDD Cup 1999 dataset.
    
*   **Advantage:** It removes redundant records from the KDD'99 set, providing a more balanced and realistic benchmark for evaluating intrusion detection models.
    

### 4.2. Feature Categories

*   **Basic Features:** Intrinsic attributes of a TCP connection (e.g., duration, protocol\_type).
    
*   **Content Features:** Features based on the packet's payload (e.g., num\_failed\_logins).
    
*   **Traffic Features:** Statistical features computed over a time window (e.g., same\_srv\_rate).
    

### 4.3. Attack Categories

*   **Denial of Service (DoS):** Flooding a server to make it unavailable.
    
*   **Probe:** Scanning a network to find vulnerabilities.
    
*   **Remote to Local (R2L):** Gaining local access from a remote machine.
    
*   **User to Root (U2R):** Gaining administrator access from a user-level account.
    

🔬 5. Methodology and ML Pipeline
---------------------------------

### 5.1. Data Preprocessing

*   **Feature Encoding:** One-Hot Encoding was used to convert the three categorical features (protocol\_type, service, flag) into a numerical format.
    
*   **Label Encoding:** The multi-class target variable (class) was converted into a binary format: 0 for 'Normal' and 1 for 'Attack'.
    
*   **Data Normalization:** Min-Max Scaling was applied to all numerical features to scale them to a uniform range of \[0, 1\].
    

### 5.2. Model Selection

*   **K-Nearest Neighbors (KNN):** An instance-based algorithm that classifies based on the majority class of its 'k' closest neighbors.
    
*   **Linear Discriminant Analysis (LDA):** A statistical method that finds a linear combination of features to best separate the classes.
    
*   **Support Vector Machine (SVM):** A powerful classifier that finds an optimal hyperplane to separate data points into different classes.
    

⚙️ 6. Installation and Local Setup
----------------------------------

### 6.1. Prerequisites

*   Python (version 3.8 or higher)
    
*   pip package installer
    
*   Git
    
*   Git LFS (for handling large model files)
    

### 6.2. Step-by-Step Installation

1.  ```bash
    git clone https://github.com/saivarshith123/ML-Powered-Network-Intrusion-Detection-Dashboard.git
    ```
2. ```bash
    cd ML-Powered-Network-Intrusion-Detection-Dashboard
    ```
    
3.  ```bash
    git lfs install
    git lfs pull
    ```
    
    *   This step is crucial to download the large .pkl model files correctly.
        
3.  ```bash
    python -m venv venv
    source venv/bin/activate 
    ```
    
4.  ```bash
    pip install -r requirements.txt
    ```
    
5.  ```bash
    python train_and_evaluate.py
    ```
    
    *   This is a mandatory one-time step that generates all the model assets (.pkl, .csv) needed by the app.
        

🖥️ 7. How to Use the Dashboard
-------------------------------

### 7.1. Running the Application

*   Ensure all dependencies are installed and the training script has been run.
    
*   ```bash
    streamlit run app.py
    ```

📈 8. Performance and Results

### 8.1. Performance Metrics

*   **Accuracy:** Overall percentage of correct predictions.
    
*   **Precision:** Ability of the model to avoid false alarms (false positives).
    
*   **Recall:** Ability of the model to find all actual attacks (avoiding false negatives).
    
*   **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure.
    

### 8.2. Comparative Analysis

*   The following table summarizes the performance of the models on the unseen KDDTest+.txt data.
    
*   Based on the highest Accuracy and F1-Score, the **KNN model was selected** for deployment in the dashboard.
    

ModelAccuracyPrecisionRecallF1-Score**K-Nearest Neighbors (KNN)0.7694**0.9240**0.64820.7619**Linear Discriminant Analysis (LDA)0.7617**0.9249**0.63270.7514Support Vector Machine (SVM)0.75390.91630.62480.7430Export to Sheets

🔮 9. Future Work and Roadmap
-----------------------------

### 9.1. Short-Term Goals

*   **Deep Learning Integration:** Implement and evaluate advanced models like LSTMs and Autoencoders.
    
*   **Hyperparameter Tuning:** Systematically optimize the parameters of the existing models to improve performance.
    

### 9.2. Long-Term Vision

*   **Real-Time Packet Capture:** Transition from file-based analysis to a live pipeline that captures and analyzes network traffic on the fly.
    
*   **Multi-Class Classification:** Enhance the model to identify the specific _type_ of attack (DoS, Probe, etc.).
    
*   **Cloud Deployment:** Deploy the Streamlit application to a public cloud service (e.g., Streamlit Community Cloud, Heroku) for wider accessibility.
    

🤝 10. Contributing
-------------------

*   Contributions are what make the open-source community such an amazing place to learn, inspire, and create.
    
*   Any contributions you make are **greatly appreciated**.
    
*   Please feel free to fork the repository and submit a pull request with any enhancements.
    

📜 11. License
--------------

*   This project is distributed under the MIT License. See the LICENSE file for more information.
    

🙏 12. Acknowledgments
----------------------

*   This project was made possible by the public availability of the **NSL-KDD Dataset**.
    
*   A special thank you to the developers of **Streamlit**, **Scikit-learn**, and the entire open-source Python ecosystem.
    

👨‍💻 13. Project Team
----------------------
*   **Gotam Sai Varshith**
*   **PAVAN KRISHNA R** 
*   **MRINAL SWAIN** 
