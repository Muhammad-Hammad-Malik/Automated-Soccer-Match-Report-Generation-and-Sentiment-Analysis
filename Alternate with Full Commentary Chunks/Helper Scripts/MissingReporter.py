import pandas as pd

# Load the CSV file
df = pd.read_csv("Commentaries_with_Reports.csv")

# Count rows where either Commentary or MatchReport is missing or empty
missing_count = df[df['Commentary'].isnull() | df['Commentary'].eq("") |
                   df['MatchReport'].isnull() | df['MatchReport'].eq("")].shape[0]

print(f"🕵️ Rows with missing values in 'Commentary' or 'MatchReport': {missing_count}")
