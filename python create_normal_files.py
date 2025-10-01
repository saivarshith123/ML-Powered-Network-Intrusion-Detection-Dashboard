# create_evaluation_files.py

import pandas as pd
import numpy as np

# --- Configuration ---
NUMBER_OF_FILES = 50
SOURCE_FILE = 'KDDTest+.txt'

# Define column headers
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

print(f"Reading data from '{SOURCE_FILE}'...")
df = pd.read_csv(SOURCE_FILE, header=None, names=columns)

# Drop only the 'difficulty' column, KEEPING the 'class' column
df_to_split = df.drop(['difficulty'], axis=1)

df_chunks = np.array_split(df_to_split, NUMBER_OF_FILES)

print(f"\nSplitting data into {NUMBER_OF_FILES} files with labels...")
for i, chunk in enumerate(df_chunks):
    output_filename = f'evaluation_chunk_{i + 1}.csv'
    chunk.to_csv(output_filename, index=False)
    print(f"Successfully created '{output_filename}'")

print("\nFinished!")
