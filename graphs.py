import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

def create_bar_chart():
    """
    Loads the model performance CSV and creates a grouped bar chart.
    """
    print("--- Generating Comparative Performance Bar Chart ---")
    
    try:
        # Try to load the performance data from your CSV file
        results_df = pd.read_csv('model_performance.csv')
        print("Loaded data from 'model_performance.csv'")
        
    except FileNotFoundError:
        # If the file isn't found, use the data from your report as a fallback
        print("Warning: 'model_performance.csv' not found.")
        print("Using fallback data hardcoded from your paper...")
        
        # This data is from your 'review 2.pdf' 
        results_data = {
            'Model': ['KNN', 'LDA', 'SVM'],
            'Accuracy': [0.7694, 0.7617, 0.7539],
            'Precision': [0.9240, 0.9249, 0.9163],
            'Recall': [0.6482, 0.6327, 0.6248],
            'F1-Score': [0.7619, 0.7514, 0.7430]
        }
        results_df = pd.DataFrame(results_data)

    # "Melt" the DataFrame to "long" format
    # This turns metrics (Accuracy, Precision, etc.) into a single column
    # so we can group the bars by 'Metric'.
    df_melted = results_df.melt('Model', var_name='Metric', value_name='Score')
    
    # --- Create the Plot ---
    plt.figure(figsize=(12, 8)) # Set the figure size
    
    # Create the grouped bar plot
    sns.barplot(data=df_melted, 
                x='Metric',  # X-axis will be the metrics
                y='Score',   # Y-axis will be the score
                hue='Model') # Group bars by 'Model'
    
    plt.title('Comparative Model Performance on NSL-KDD Test Set')
    plt.ylim(0, 1.0) # Set Y-axis from 0 to 1
    plt.legend(loc='upper right') # Add a legend
    
    # Save the figure as a high-quality PNG
    output_filename = "Model_Performance_Bar_Chart.png"
    plt.savefig(output_filename)
    
    print(f"\nBar chart saved as: {output_filename}")
    
    # Display the plot
    plt.show()

# Standard Python entry point
if __name__ == "__main__":
    create_bar_chart()
