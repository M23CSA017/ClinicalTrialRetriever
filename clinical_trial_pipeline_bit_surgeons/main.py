# main.py

import logging
from pathlib import Path
import os
import numpy as np

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Importing from src package
from src.data_preprocessing import load_data, chunk_data
from src.embedding import embed_texts
from src.clustering import perform_clustering
from src.sampling import sample_cluster_data

from src.fine_tuning import fine_tune_sbert
from src.faiss_indexing import build_faiss_index, save_faiss_index
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.similarity import prepare_training_datasets
from src.similarity import compute_similarity_dataset
from sentence_transformers.losses import MultipleNegativesSymmetricRankingLoss



def setup_logging():
    """
    Configures logging for the pipeline.
    Logs are written to both console and a log file named 'pipeline.log'.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
        handlers=[
            logging.FileHandler("pipeline.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main function to orchestrate the clustering pipeline.
    """
    # Initialize logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration Parameters
    DATA_FILE = "data/raw/final_main_data_cleaned.csv"
    PROCESSED_DIR = Path("data/processed")
    CHECKPOINT_DIR = Path("checkpoints")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_PATH = PROCESSED_DIR / "faiss_index.bin"
    CHUNKED_DATA_PATH = PROCESSED_DIR / "chunked_data_with_embeddings.pkl"

    SAMPLING_FRAC = 0.70
    UMAP_COMPONENTS = 2
    HDBSCAN_ITER_SEARCH = 20
    CLUSTER_SAMPLE_FRAC = 0.5
    TOP_K = 3
    MAX_TOKENS = 350
    FINE_TUNE_EPOCHS = 1
    FINE_TUNE_BATCH_SIZE = 64
    FINE_TUNE_WARMUP = 100
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    try:
        # Step 1: Data Loading & Preprocessing
        logger.info("=== Step 1: Data Loading & Preprocessing ===")
        df = load_data(DATA_FILE, frac=SAMPLING_FRAC, seed=42)

        # Step 2: Sentence-Level Chunking
        logger.info("\n=== Step 2: Sentence-Level Chunking ===")
        chunked_df = chunk_data(df, max_tokens=MAX_TOKENS)
        chunked_df.to_pickle(CHUNKED_DATA_PATH)
        logger.info(f"Chunked data saved to {CHUNKED_DATA_PATH}.")


        # Step 3: Embedding with SBERT
        logger.info("\n=== Step 3: Embedding with SBERT ===")
        texts = chunked_df["chunk_text"].tolist()
        embeddings, base_model = embed_texts(texts, model_name=MODEL_NAME, batch_size=64)
        chunked_df['embedding'] = list(embeddings)


        # Step 4 & 5: Dimensionality Reduction and Clustering
        logger.info("\n=== Step 4 & 5: Dimensionality Reduction and Clustering ===")
        umap_params = {
            "n_components": UMAP_COMPONENTS,
            "n_neighbors": 50,
            "min_dist": 0.005,
            "metric": "euclidean"
        }
        hdbscan_params = {
            "min_samples": 125,
            "min_cluster_size": 400,
            "cluster_selection_method": "eom",
            "metric": "euclidean",
        }

      
        cluster_embedding, final_labels, hdbscan_model = perform_clustering(
            embeddings=embeddings,
            umap_params=umap_params,
            hdbscan_params=hdbscan_params,
            save_path="data/processed", 
            seed=42,
        )


        # Assign cluster labels to DataFrame
        chunked_df['cluster_labels'] = final_labels
        n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
        logger.info(f"Clusters found (excluding noise): {n_clusters}")

        # Step 7: Cluster-Based Sampling
        logger.info("\n=== Step 7: Cluster-Based Sampling ===")
        anchor_df, candidate_df = sample_cluster_data(chunked_df, sample_frac=CLUSTER_SAMPLE_FRAC, seed=42)


        # Step 8: Similarity Computation & Dataset Preparation
        logger.info("\n=== Step 8: Similarity Computation & Dataset Preparation ===")
        train_dataset, eval_dataset = prepare_training_datasets(
            anchor_df=chunked_df, 
            model=base_model, 
            train_split=0.8, 
            top_k=TOP_K
        )
        logger.info("Train and evaluation datasets prepared.")

        
        # Step 9: Fine-Tuning SBERT
        logger.info("\n=== Step 9: Fine-Tuning SBERT ===")

        fine_tuned_model = fine_tune_sbert(
            model=base_model,
            train_pairs=train_dataset,
            num_epochs=FINE_TUNE_EPOCHS,
            batch_size=FINE_TUNE_BATCH_SIZE,
            learning_rate=2e-5,
            output_dir=str(CHECKPOINT_DIR / "fine_tuned_model")
        )


        # Save the fine-tuned model
        fine_tuned_model_path = CHECKPOINT_DIR / "fine_tuned_model"
        fine_tuned_model.save(str(fine_tuned_model_path))
        logger.info(f"Fine-tuned model saved to {fine_tuned_model_path}")

        # Step 10: Building a FAISS Index
        logger.info("\n=== Step 10: Building a FAISS Index ===")
        final_embeddings = fine_tuned_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        faiss_index = build_faiss_index(final_embeddings)

        # Save FAISS Index and Metadata
        save_faiss_index(faiss_index, FAISS_INDEX_PATH)
        chunked_df['embedding'] = list(final_embeddings)
        chunked_df.to_pickle(CHUNKED_DATA_PATH)  # Save chunked data with embeddings
        logger.info(f"FAISS index and chunked data saved to {PROCESSED_DIR}.")

        # Step 11: Clustering Evaluation Metrics
        logger.info("\n=== Step 11: Clustering Evaluation Metrics ===")
        if len(set(final_labels)) > 1:
            silhouette = silhouette_score(cluster_embedding, final_labels)
            davies_bouldin = davies_bouldin_score(cluster_embedding, final_labels)
            calinski_harabasz = calinski_harabasz_score(cluster_embedding, final_labels)
            logger.info(f"Silhouette Score: {silhouette}")
            logger.info(f"Davies-Bouldin Index: {davies_bouldin}")
            logger.info(f"Calinski-Harabasz Index: {calinski_harabasz}")
        else:
            logger.warning("Only one cluster found. Evaluation metrics are not defined.")

        logger.info("\n=== Pipeline Completed Successfully ===")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
