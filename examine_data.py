import pandas as pd

# Load the dataset
try:
    df = pd.read_csv("neurodiversity_dataset_no_country.csv")
    # Write just column names to file
    with open("columns.txt", "w") as f:
        f.write("Column names:\n")
        for i, col in enumerate(df.columns):
            f.write(f"{i+1}. {col}\n")
except Exception as e:
    with open("error.txt", "w") as f:
        f.write(f"Error: {str(e)}") 