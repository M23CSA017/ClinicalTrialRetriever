import faiss
import logging

logger = logging.getLogger(__name__)

def build_faiss_index(embeddings):

    try:
        d = embeddings.shape[1]  

        # Initialize GPU resources
        res = faiss.StandardGpuResources()

        
        index_flat = faiss.IndexFlatL2(d)  
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_flat)  # Transfer to GPU

        # Add embeddings to the index
        index_gpu.add(embeddings)
        logger.info(f"FAISS GPU FlatL2 index built with {index_gpu.ntotal} vectors.")
        return index_gpu
    except Exception as e:
        logger.error(f"Error in build_faiss_flat_index_gpu: {e}")
        raise

def save_faiss_index(index, filepath):

    try:
        index_cpu = faiss.index_gpu_to_cpu(index) 
        faiss.write_index(index_cpu, str(filepath))
        logger.info(f"FAISS index saved to {filepath}.")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        raise

def load_faiss_index(filepath):
    
    try:
        index_cpu = faiss.read_index(filepath)  
        res = faiss.StandardGpuResources()  
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu) 
        logger.info(f"FAISS index loaded from {filepath} and moved to GPU.")
        return index_gpu
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise
