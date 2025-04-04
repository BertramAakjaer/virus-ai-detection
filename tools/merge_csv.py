import pandas as pd
import tkinter as tk
from tkinter import filedialog

def merge_csv_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Select first CSV file
    print("Select first CSV file")
    file1 = filedialog.askopenfilename(title="Select first CSV file",
                                      filetypes=[("CSV files", "*.csv")])
    
    # Select second CSV file
    print("Select second CSV file")
    file2 = filedialog.askopenfilename(title="Select second CSV file",
                                      filetypes=[("CSV files", "*.csv")])
    
    if file1 and file2:
        # Read the CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Merge the dataframes
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Save the merged file
        output_file = filedialog.asksaveasfilename(defaultextension=".csv",
                                                  filetypes=[("CSV files", "*.csv")])
        if output_file:
            merged_df.to_csv(output_file, index=False)
            print(f"Merged file saved as: {output_file}")
    else:
        print("File selection cancelled")

if __name__ == "__main__":
    merge_csv_files()