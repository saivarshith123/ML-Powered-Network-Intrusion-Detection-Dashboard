# train_and_save_all_models.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
print("--- Training and Saving ALL Models ---")

# --- (The preprocessing code is the same as before) ---
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
df = pd.get_dummies(df_train, columns=['protocol_type', 'service', 'flag'])
train_labels = df['class'].apply(lambda x: 0 if x == 'normal' else 1)
train_df = df.drop(['class', 'difficulty'], axis=1)
model_columns = train_df.columns
numerical_cols = [col for col in df_train.columns if df_train[col].dtype != 'object' and col not in ['class', 'difficulty']]
scaler = MinMaxScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
X_train = train_df.values
y_train = train_labels.values

# --- Train and Save Each Model ---
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LDA": LinearDiscriminantAnalysis(solver='svd'),
    "SVM": SVC(kernel='linear')
}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name.lower()}_model.pkl')
    print(f"Saved {name} model.")

# Save the scaler and columns
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model_columns, 'model_columns.pkl')

print("\n--- All models and assets saved! ---")
