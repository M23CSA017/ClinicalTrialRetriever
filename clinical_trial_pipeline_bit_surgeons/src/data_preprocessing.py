# src/data_preprocessing.py

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
import logging

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

# Configure logging
logger = logging.getLogger(__name__)

def load_data(data_path, frac=1.0, seed=42):
    
    
    try:
        df = pd.read_csv(data_path)
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=seed)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Loaded {len(df)} rows after sampling.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_text(text):
    
    if pd.isna(text) or not text:
        return ""
    text = str(text).strip()
    text = " ".join(text.split())  
    return text

def combine_columns(row, columns):
    
    segments = []
    for col in columns:
        raw_text = row.get(col, "")
        text = preprocess_text(raw_text)
        if text:
            col_label = col.upper().replace(" ", "_")
            segments.append(f"[{col_label}] {text}")
    return " ".join(segments).strip()

def sentence_based_chunking(complete_text, max_tokens=400):

    try:
        sents = sent_tokenize(complete_text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent in sents:
            tokens_in_sent = len(sent.split())
            if current_tokens + tokens_in_sent > max_tokens and current_chunk:
                # Finalize current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_tokens = tokens_in_sent
            else:
                current_chunk.append(sent)
                current_tokens += tokens_in_sent

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    except Exception as e:
        logger.error(f"Error in sentence_based_chunking: {e}")
        return []

def build_complete_text(row):

    short_cols = ["Study Title", "Conditions", "Phases", "healthy_volunteers"]
    long_cols = ["Primary Outcome Measures", "Secondary Outcome Measures", "criteria_cleaned", "Brief Summary"]
    all_cols = short_cols + long_cols
    complete_text = combine_columns(row, all_cols)
    return complete_text

def chunk_data(df, max_tokens=400):

    all_chunks = []

    for idx, row in df.iterrows():
        metadata = {
            "row_id": idx,
            "NCT_NUMBER": row.get("NCT_Number", ""),
            "cluster_label": row.get("cluster_labels", -1)
        }

        complete_text = build_complete_text(row)
        if not complete_text:
            continue  # Skip empty texts

        chunks = sentence_based_chunking(complete_text, max_tokens=max_tokens)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_data = {
                "NCT_NUMBER": metadata["NCT_NUMBER"],
                "complete_text": complete_text,
                "chunk_text": chunk,
                "row_id": metadata["row_id"],
                "original_column": "COMPLETE_TEXT",
                "token_count": len(chunk.split()),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "cluster_label": metadata["cluster_label"]
            }
            all_chunks.append(chunk_data)

    chunked_df = pd.DataFrame(all_chunks)
    logger.info(f"Created {len(chunked_df)} chunks from {len(df)} documents.")
    return chunked_df
