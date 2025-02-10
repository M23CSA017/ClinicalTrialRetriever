import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re


def extract_field(complete_text, field):
    """
    Extract specific field from complete_text using regex.
    """
    pattern = fr"\[{field}\] (.*?)(?=\[|$)"
    match = re.search(pattern, complete_text, re.DOTALL)
    return match.group(1).strip() if match else "N/A"


def run_query(query, nct_number, top_k=10, 
              index_path="data/processed/faiss_index.bin", 
              data_path="data/processed/chunked_data_with_embeddings.pkl", 
              fine_tuned_model_path="checkpoints/fine_tuned_model", 
              output_dir="data/processed"):
    """
    Perform semantic search for a single query and save results to CSV.
    """
    # Load the fine-tuned model
    model = SentenceTransformer(fine_tuned_model_path)

    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Load FAISS index and chunked data
    index = faiss.read_index(index_path)
    chunked_df = pd.read_pickle(data_path)

    # Perform the search
    D, I = index.search(query_embedding, top_k * 5)  # Search more to allow for duplicate removal

    # Get top results
    top_results = chunked_df.iloc[I[0]]

    # Remove duplicates and extract specific columns
    unique_trials = top_results.drop_duplicates(subset='NCT_NUMBER').head(top_k)

    # Extract cluster labels if available
    cluster_labels = unique_trials.get("cluster_label", ["N/A"] * len(unique_trials))

    # Create final output DataFrame 
    # we can extend this to include more fields if needed
    final_output = pd.DataFrame({
        'Rank': range(1, len(unique_trials) + 1),
        'NCT_Number': unique_trials['NCT_NUMBER'],
        'Study_Title': unique_trials['complete_text'].apply(lambda x: extract_field(x, 'STUDY_TITLE')),
        'Primary_Outcome_Measures': unique_trials['complete_text'].apply(lambda x: extract_field(x, 'PRIMARY_OUTCOME_MEASURES')),
        'Secondary_Outcome_Measures': unique_trials['complete_text'].apply(lambda x: extract_field(x, 'SECONDARY_OUTCOME_MEASURES')),
        'Criteria_Cleaned': unique_trials['complete_text'].apply(lambda x: extract_field(x, 'CRITERIA_CLEANED')),
        'Brief_Summary': unique_trials['complete_text'].apply(lambda x: extract_field(x, 'BRIEF_SUMMARY'))
    })

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to a CSV file
    output_file = os.path.join(output_dir, f"{nct_number}_results.csv")
    final_output.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def process_test_file(test_csv_path, output_dir="data/processed"):
    """
    Process each row of the test.csv file and run queries.
    Test file contains the three clinical trials data which were mentioned in the problem statement pdf.
    """
    # Load test data
    test_data = pd.read_csv(test_csv_path)

    # Iterate over each row in the test data
    for _, row in test_data.iterrows():
        # Construct the query. We combine the Study Title, Primary Outcome Measures, Secondary Outcome Measures, and Criteria
        # These columns were mentioned in the problem statement pdf as examples so we used them.
        query = f"""
        {row['Study Title']}
        {row['Primary Outcome Measures']}
        {row['Secondary Outcome Measures']}
        {row['criteria_cleaned']}
        """

        # Use the Study Title for unique file names, sanitized for file paths
        sanitized_title = re.sub(r'[^A-Za-z0-9_]', '_', row['NCT_Number'][:20])

        # Run query and save results
        run_query(
            query=query,
            nct_number=sanitized_title,
            output_dir=output_dir
        )


if __name__ == "__main__":
    # Path to test.csv
    # output will be saved in the output_dir which is inside the data folder (data/processed)
    test_csv_path = "data/raw/test.csv"
    output_dir = "data/processed"

    # Process the test file
    process_test_file(test_csv_path, output_dir)
