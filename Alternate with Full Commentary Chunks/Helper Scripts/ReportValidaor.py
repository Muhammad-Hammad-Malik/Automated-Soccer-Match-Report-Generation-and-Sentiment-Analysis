import pandas as pd

def print_first_two_reports(csv_path="Commentaries_with_Reports.csv"):
    try:
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if 'Commentary' not in df.columns or 'MatchReport' not in df.columns:
            raise ValueError("CSV must contain 'Commentary' and 'MatchReport' columns")

        # Read and print first two reports
        for i in range(min(615, len(df))):
            commentary = df.at[i, 'Commentary']
            report = df.at[i, 'MatchReport']

            print(f"\n=== Match Commentary #{i+1} ===\n{commentary}")
            print(f"\n--- Match Report #{i+1} ---\n{report}")
            print("="*60)

    except Exception as e:
        print(f"Error: {e}")

# Run it
if __name__ == "__main__":
    print_first_two_reports()
