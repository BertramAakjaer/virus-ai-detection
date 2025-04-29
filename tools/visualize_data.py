import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def visualize_excel_data(file_path, x_column, y_column):
    # Read the Excel file with a loading bar
    print("Loading data...")
    df = pd.read_excel(file_path)
    
    print("Processing data...")
    # Create color map based on Label column with progress bar
    colors = []
    for label in tqdm(df['Label'], desc="Creating color map"):
        colors.append('red' if label == 'malware' else 'green')
    
    print("Creating visualization...")
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], c=colors, alpha=0.6)
    
    # Add labels and title
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{x_column} vs {y_column} by Label (First 100 rows)')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Malware', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Harmless', markersize=10)
    ]
    plt.legend(handles=legend_elements)
    
    # Show the plot
    plt.grid(True, alpha=0.3)
    print("Done! Displaying plot...")
    plt.show()

if __name__ == "__main__":
    file_path = r"C:\Users\bertr\Downloads\processed_features.xlsx"  # Example file path
    x_column = 'dll_kernel32_dll_count'  # Example X-axis column
    y_column = 'dll_comsvcs_dll_count'  # Example Y-axis column  
    
    visualize_excel_data(file_path, x_column, y_column)