import pandas as pd

def fix_commentary(text):
    if pd.isna(text):
        return text
    # Replace newlines and carriage returns with space
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Collapse multiple spaces into one
    text = ' '.join(text.split())
    return text

# Load your CSV file
df = pd.read_csv('Commentary2.csv')

# Apply the cleaning function to the "Commentary" column
df['Commentary'] = df['Commentary'].apply(fix_commentary)

# Save the cleaned file
df.to_csv('cleaned_output.csv', index=False)

print("Done! Cleaned CSV saved as 'cleaned_output.csv'")
