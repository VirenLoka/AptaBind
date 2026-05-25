import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Add unique IDs to a sequence TSV file.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input TSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to save the output TSV file")
    parser.add_argument("--prefix", "-p", default="seq_", help="Prefix for the unique ID (default: 'seq_')")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Load the TSV file
    df = pd.read_csv(args.input, sep='\t')

    # Generate unique IDs using the prefix (e.g., seq_1, seq_2)
    df['sequence_id'] = [f"{args.prefix}{i+1}" for i in range(len(df))]

    # Reorder columns to put 'sequence_id' first
    # This dynamically grabs the original columns and appends them after the ID
    original_cols = [col for col in df.columns if col != 'sequence_id']
    df = df[['sequence_id'] + original_cols]

    # Save to the new TSV file
    df.to_csv(args.output, sep='\t', index=False)
    
    print(f"Success! Generated {len(df)} unique IDs.")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()