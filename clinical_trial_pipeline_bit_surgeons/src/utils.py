# src/utils.py

import logging
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

def sentence_based_chunking(complete_text, max_tokens=400):
    """
    Splits text into chunks along sentence boundaries, ensuring each chunk <= max_tokens.
    Returns a list of chunk strings.
    """
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
