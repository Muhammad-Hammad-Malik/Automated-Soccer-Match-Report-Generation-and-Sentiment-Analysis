import pandas as pd

# File paths
input_csv = "commentary_dataset_with_reports.csv"
output_csv = "cleaned_commentary_dataset.csv"

# Load the CSV
df = pd.read_csv(input_csv)
print(df.shape)

df_cleaned = df.dropna(how='any')
print(df.shape)
df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: row.astype(str).str.strip().eq("").any(), axis=1)]
df_cleaned.to_csv(output_csv, index=False)

print(f"✅ Cleaned CSV saved to '{output_csv}' with {len(df_cleaned)} rows.")
