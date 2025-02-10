# src/sampling.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def assign_cluster_labels(df, labels):

    df["cluster_labels"] = labels
    return df

def sample_cluster_data(df, sample_frac=0.3, seed=42):

    try:
        anchor_list = []
        candidate_list = []

        grouped = df.groupby("cluster_labels", group_keys=True)
        for cluster_id, cluster_df in grouped:
            # Skip noise cluster if HDBSCAN returns -1
            if cluster_id == -1:
                continue
            anchor_size = max(1, int(len(cluster_df) * sample_frac))
            anchor_sample = cluster_df.sample(n=anchor_size, random_state=seed)
            candidate_sample = cluster_df.drop(anchor_sample.index)
            anchor_list.append(anchor_sample)
            candidate_list.append(candidate_sample)

        anchor_df = pd.concat(anchor_list).reset_index(drop=True) if anchor_list else pd.DataFrame()
        candidate_df = pd.concat(candidate_list).reset_index(drop=True) if candidate_list else pd.DataFrame()
        logger.info(f"Sampled {len(anchor_df)} anchor chunks and {len(candidate_df)} candidate chunks.")
        return anchor_df, candidate_df
    except Exception as e:
        logger.error(f"Error in sample_cluster_data: {e}")
        raise
