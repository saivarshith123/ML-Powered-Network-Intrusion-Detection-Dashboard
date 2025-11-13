#!/usr/bin/env python

"""
Full analysis pipeline for Network Intrusion Detection.

This script performs the following steps:
1.  Loads the NSL-KDD dataset (KDDTrain+.txt and KDDTest+.txt).
2.  Applies comprehensive preprocessing:
    - One-hot encoding for categorical features.
    - Binary label encoding (normal=0, attack=1).
    - Min-Max scaling for numerical features.
    - Ensures consistent columns between train and test sets.
3.  Trains three machine learning models:
    - K-Nearest Neighbors (KNN)
    - Linear Discriminant Analysis (LDA)
    - Support Vector Machine (SVM)
4.  Evaluates each model on the test set, calculating:
    - Accuracy, Precision, Recall, F1-Score
    - A full Classification Report
    - A Confusion Matrix
5.  Prints a final summary table of all model performances.
6.  Generates and saves heatmap plots for each model's confusion matrix
    (e.g., "K-Nearest Neighbors (KNN)_confusion_matrix.png").
"""

# --- 1. Import Necessary Libraries ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Suppress common warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def main():
    """
    Main function to run the entire data loading, preprocessing,
    training, and evaluation pipeline.
    """
    
    # --- 2. Data Loading ---
    print("--- Starting Data Loading ---")

    # Define the column names for the NSL-KDD dataset
    # (Based on the documentation and your notebook)
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
    ]

    # Load the training and test datasets
    try:
        df_train = pd.read_csv('KDDTrain+.txt', header=None, names=columns)
        df_test = pd.read_csv('KDDTest+.txt', header=None, names=columns)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'KDDTrain+.txt' and 'KDDTest+.txt' are in the same directory as this script.")
        return

    # Drop the 'difficulty' column as it's not needed for detection
    df_train.drop(['difficulty'], axis=1, inplace=True)
    df_test.drop(['difficulty'], axis=1, inplace=True)
    
    print("--- Data Loading Complete ---")
    
    # --- 3. Data Preprocessing ---
    print("--- Starting Data Preprocessing ---")

    # Identify categorical and numerical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Get all columns that are numbers
    numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()

    # Combine train and test sets for consistent one-hot encoding
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # Perform one-hot encoding on the combined DataFrame
    for col in categorical_cols:
        dummies = pd.get_dummies(df_combined[col], prefix=col, dtype=int)
        df_combined = pd.concat([df_combined, dummies], axis=1)
        df_combined.drop(col, axis=1, inplace=True)

    # Separate back into training and testing sets
    train_rows = len(df_train)
    train_df = df_combined.iloc[:train_rows].copy()
    test_df = df_combined.iloc[train_rows:].copy()

    # Create binary labels: 0 for 'normal' and 1 for 'attack'
    train_labels = train_df['class'].apply(lambda x: 0 if x == 'normal' else 1)
    test_labels = test_df['class'].apply(lambda x: 0 if x == 'normal' else 1)

    # Drop the original 'class' column
    train_df.drop(['class'], axis=1, inplace=True)
    test_df.drop(['class'], axis=1, inplace=True)

    # Align columns: Ensure test set has same columns as train set
    train_cols = train_df.columns
    test_cols = test_df.columns

    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        test_df[c] = 0  # Add missing columns with 0
    
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        train_df[c] = 0 # Add missing columns with 0

    # Ensure the column order is identical
    test_df = test_df[train_cols]

    # Normalize numerical features
    # We must use the *same* scaler fitted on the training data
    scaler = MinMaxScaler()
    
    # Fit on training data and transform it
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    
    # Transform test data using the *already fitted* scaler
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

    # Convert dataframes to numpy arrays for scikit-learn
    X_train = train_df.values
    y_train = train_labels.values
    X_test = test_df.values
    y_test = test_labels.values

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print("--- Preprocessing Complete ---")

    # --- 4. Train, Evaluate, and Plot Models ---
    
    # Initialize the models
    models = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(solver='svd'),
        "Support Vector Machine (SVM)": SVC(kernel='linear')
    }

    # Create a list to store the results dictionaries
    results_list = []

    # Loop through each model
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        
        print(f"--- Evaluating {name} ---")
        y_pred = model.predict(X_test)
        
        # --- Calculate and Store Performance Metrics ---
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Classification Report for {name}:\n")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
        
        # --- Store results in a dictionary ---
        model_results = {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
        results_list.append(model_results)
        
        # --- Generate and Save Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for {name}:\n{cm}\n")

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted: Normal', 'Predicted: Attack'],
                    yticklabels=['Actual: Normal', 'Actual: Attack'])
        plt.title(f'Confusion Matrix for {name} Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save the figure to a file
        plot_filename = f"{name}_confusion_matrix.png"
        plt.savefig(plot_filename)
        print(f"Confusion matrix plot saved as: {plot_filename}")
        
        # Clear the plot for the next loop
        plt.clf() 
        
        print("-" * 60)

    # --- 5. Display Final Summary Table ---
    print("\n--- Final Model Performance Summary ---")

    # Create a DataFrame from the results list for pretty printing
    results_df = pd.DataFrame(results_list)
    
    # Set display options for better console printing
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    
    print(results_df.to_string(index=False, float_format="%.6f"))
    print("\n--- Analysis Complete ---")


# Standard Python entry point
if __name__ == "__main__":
    main()
