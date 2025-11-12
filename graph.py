import matplotlib.pyplot as plt
import numpy as np

# --- Data from TABLE I ---
models = ['KNN', 'LDA', 'SVM']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Data for each model, in the order of the 'metrics' list
knn_scores = [0.7694, 0.9240, 0.6482, 0.7619]
lda_scores = [0.7617, 0.9249, 0.6327, 0.7514]
svm_scores = [0.7539, 0.9163, 0.6248, 0.7430]

# --- Plotting the Bar Graph ---

# Set the positions and width for the bars
x = np.arange(len(metrics))  # the label locations
width = 0.25  # the width of the bars
margin = 0.03 # small margin between bar groups

# Create the figure and a set of subplots
# We increase the figure size to make it clear and readable for the paper
fig, ax = plt.subplots(figsize=(12, 7))

# Create the bars for each model
rects1 = ax.bar(x - width - margin, knn_scores, width, label='KNN', color='#1f77b4')
rects2 = ax.bar(x, lda_scores, width, label='LDA', color='#ff7f0e')
rects3 = ax.bar(x + width + margin, svm_scores, width, label='SVM', color='#2ca02c')

# --- Add labels, title, and custom x-axis tick labels ---
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Fig. 4. Visual Comparison of Model Performance Metrics on NSL-KDD Test Set', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.1) # Set Y-axis limit from 0 to 1.1 to make space for labels

# Add a legend
ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))

# --- Add data labels on top of each bar ---
ax.bar_label(rects1, padding=3, fmt='%.4f', rotation=90, fontsize=9)
ax.bar_label(rects2, padding=3, fmt='%.4f', rotation=90, fontsize=9)
ax.bar_label(rects3, padding=3, fmt='%.4f', rotation=90, fontsize=9)

# Add grid lines for easier reading
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent labels from being cut off
fig.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust right margin to fit legend

# --- Save the figure ---
# This will save the plot as a high-resolution PNG file in the same folder
plt.savefig("model_performance_graph.png", dpi=300)

print("Graph 'model_performance_graph.png' has been saved successfully.")

# Optionally, display the plot
# plt.show()
