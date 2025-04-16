import pandas as pd

# Step 1: Read commentary1.csv and extract 'events' column
df1 = pd.read_csv('23_24_match_details.csv', usecols=['events'])

# Step 2: Read commentary2.csv
df2 = pd.read_csv('commentary_dataset.csv')

# Step 3: Check if 'Commentary' column exists
if 'Commentary' not in df2.columns:
    raise ValueError("'Commentary' column not found in Commentary2.csv")

# Step 4: Append events to the Commentary column
# Convert events to a list of strings
events_list = df1['events'].dropna().astype(str).tolist()

# Create a new DataFrame to match the shape
new_rows = pd.DataFrame({'Commentary': events_list})

# Step 5: Append new rows to df2
df2 = pd.concat([df2, new_rows], ignore_index=True)

# Step 6: Save back to Commentary2.csv
df2.to_csv('Commentary2.csv', index=False)

print("✅ Events from commentary1.csv appended to Commentary2.csv successfully!")
