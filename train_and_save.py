# train_and_evaluate.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
print("--- Starting Model Training, Evaluation, and Saving ---")

# --- (All the preprocessing code from before remains the same) ---
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
]
df_train = pd.read_csv('KDDTrain+.txt', header=None, names=columns)
df_test = pd.read_csv('KDDTest+.txt', header=None, names=columns)
df_train.drop(['difficulty'], axis=1, inplace=True)
df_test.drop(['difficulty'], axis=1, inplace=True)
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = [col for col in df_train.columns if df_train[col].dtype != 'object' and col != 'class']
df = pd.concat([df_train, df_test])
df = pd.get_dummies(df, columns=categorical_cols)
train_rows = len(df_train)
train_df = df.iloc[:train_rows]
test_df = df.iloc[train_rows:]
train_labels = train_df['class'].apply(lambda x: 0 if x == 'normal' else 1)
test_labels = test_df['class'].apply(lambda x: 0 if x == 'normal' else 1)
train_df = train_df.drop('class', axis=1)
test_df = test_df.drop('class', axis=1)
train_cols = train_df.columns
test_df = test_df.reindex(columns=train_cols, fill_value=0)
scaler = MinMaxScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
X_train = train_df.values
y_train = train_labels.values
X_test = test_df.values
y_test = test_labels.values

# --- Evaluate all models ---
models = {
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(solver='svd'),
    "Support Vector Machine (SVM)": SVC(kernel='linear')
}
results_list = []
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_list.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

# --- Save the performance results to a CSV file ---
performance_df = pd.DataFrame(results_list)
performance_df.to_csv('model_performance.csv', index=False)
print("\nModel performance comparison saved to 'model_performance.csv'")

# --- Find and save the best model (based on Accuracy) ---
best_model_info = performance_df.loc[performance_df['Accuracy'].idxmax()]
best_model_name = best_model_info['Model']
print(f"\nBest model found: {best_model_name} with Accuracy {best_model_info['Accuracy']:.4f}")

best_model = models[best_model_name]
print(f"Saving the best model ({best_model_name}), scaler, and columns...")
joblib.dump(best_model, 'nids_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(train_cols, 'model_columns.pkl')

print("--- All tasks complete! ---")
