import pandas as pd

# Load the CSV
df = pd.read_csv('Commentary2.csv')

# Check if 'Commentary' column exists
if 'Commentary' not in df.columns:
    raise ValueError("'Commentary' column not found in Commentary2.csv")

# Get total number of entries
total = len(df)

# Print the 4th and 500th entry if they exist
print("✅ Total entries in 'Commentary':", total)

if total >= 4:
    print("🟩 4th entry:", df.loc[3, 'Commentary'])  # Index 3 is the 4th row
else:
    print("⚠️ Not enough rows to show 4th entry.")

if total >= 500:
    print("🟩 500th entry:", df.loc[499, 'Commentary'])  # Index 499 is the 500th row
else:
    print("⚠️ Not enough rows to show 500th entry.")
