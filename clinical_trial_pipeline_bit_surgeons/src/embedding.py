# src/embedding.py

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def embed_texts(texts, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=64):
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        logger.info(f"Generated embeddings of shape: {embeddings.shape}")
        return embeddings, model
    except Exception as e:
        logger.error(f"Error in embed_texts: {e}")
        raise
