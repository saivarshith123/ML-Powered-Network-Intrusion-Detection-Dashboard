# create_normal_files.py

import pandas as pd
import numpy as np
import os

# --- Configuration ---
NUMBER_OF_FILES = 100
# We use the training file as it has a large number of 'normal' records.
SOURCE_FILE = 'KDDTrain+.txt' 

# --- Main Script ---

# Define the original column headers
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

# Check if the source file exists
if not os.path.exists(SOURCE_FILE):
    print(f"Error: The source file '{SOURCE_FILE}' was not found.")
    print("Please make sure it's in the same directory as this script.")
else:
    # Read the original data file
    print(f"Reading data from '{SOURCE_FILE}'...")
    df = pd.read_csv(SOURCE_FILE, header=None, names=columns)

    # --- THIS IS THE KEY STEP: Filter for 'normal' records only ---
    normal_df = df[df['class'] == 'normal'].copy()
    print(f"Found {len(normal_df)} 'normal' traffic records.")

    # Drop the columns that the model is supposed to predict
    normal_df_to_split = normal_df.drop(['class', 'difficulty'], axis=1)

    # Split the 'normal-only' dataframe into the desired number of chunks
    df_chunks = np.array_split(normal_df_to_split, NUMBER_OF_FILES)

    print(f"\nSplitting normal-only data into {NUMBER_OF_FILES} files...")

    # Loop through the chunks and save each one to a new CSV file
    for i, chunk in enumerate(df_chunks):
        output_filename = f'normal_traffic_chunk_{i + 1}.csv'
        
        # Save the chunk to a CSV file
        chunk.to_csv(output_filename, index=False)
        
        print(f"Successfully created '{output_filename}' with {len(chunk)} rows.")

    print(f"\nFinished! You now have {NUMBER_OF_FILES} CSV files containing only normal traffic.")
