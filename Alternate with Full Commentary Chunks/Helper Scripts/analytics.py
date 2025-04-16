import pandas as pd

# Load the CSV
df = pd.read_csv('Commentaries.csv')

# Check if 'Commentary' column exists
if 'Commentary' not in df.columns:
    raise ValueError("'Commentary' column not found in Commentary2.csv")

# Drop NaN and convert to string
comments = df['Commentary'].dropna().astype(str)

# Calculate word counts for each entry
word_counts = comments.apply(lambda x: len(x.split()))

# Print analytics
print("📊 Commentary Word Count Analytics")
print("-----------------------------------")
print("🧾 Total Entries:", len(word_counts))
print("🔠 Max Words in an Entry:", word_counts.max())
print("🔡 Min Words in an Entry:", word_counts.min())
print("📉 Average Words per Entry:", round(word_counts.mean(), 2))
print("🔢 Total Words Across All Entries:", word_counts.sum())
