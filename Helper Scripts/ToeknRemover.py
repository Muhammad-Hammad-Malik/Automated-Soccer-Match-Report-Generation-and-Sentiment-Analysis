import pandas as pd

# Load the CSV
input_csv = "cleaned_commentary_dataset.csv"
df = pd.read_csv(input_csv)

# Print shape before filtering
print(f"📊 Before filtering: {df.shape[0]} rows, {df.shape[1]} columns")

# Filter out rows with >500 tokens in Commentary
df_filtered = df[df['Commentary'].apply(lambda x: len(str(x).split()) <= 500)]

# Print shape after filtering
print(f"✅ After filtering: {df_filtered.shape[0]} rows, {df_filtered.shape[1]} columns")

# Save to a new CSV
df_filtered.to_csv("final_commentary_dataset.csv", index=False)
