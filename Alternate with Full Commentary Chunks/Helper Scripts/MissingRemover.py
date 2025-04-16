import pandas as pd

# Load the CSV file
df = pd.read_csv("Commentaries_with_Reports.csv")

# Drop rows where Commentary or MatchReport is NaN or empty
df_cleaned = df[
    df['Commentary'].notna() & df['Commentary'].ne("") &
    df['MatchReport'].notna() & df['MatchReport'].ne("")
]

# Save the cleaned DataFrame
df_cleaned.to_csv("Cleaned_Commentaries.csv", index=False)

print(f"✅ Cleaned file saved as 'Cleaned_Commentaries.csv' with {len(df_cleaned)} rows.")
