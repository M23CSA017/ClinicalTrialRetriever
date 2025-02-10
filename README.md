### **Clinical Trial Retriever 🚀**
#### **Novartis India Challenge (NEST)**

## **Overview**
This project focuses on improving **semantic retrieval for clinical trial data** by leveraging **unsupervised clustering, domain-specific fine-tuning, and efficient retrieval mechanisms**. Our approach combines **UMAP** for dimensionality reduction, **HDBSCAN** for clustering, and **fine-tuning a SentenceTransformer model** to improve search relevance.

---

## **Pipeline Breakdown 🛠️**
### **1️⃣ Preprocessing & Chunking**
- We **parse** raw clinical trial data and extract relevant fields such as:
  - **Study Title**
  - **Outcome Measures**
  - **Eligibility Criteria**
- The extracted text is split into smaller **semantic chunks** to ensure compatibility with transformer models.

### **2️⃣ Dimensionality Reduction & Clustering**
- **UMAP** reduces the high-dimensional embeddings to **2D space**, preserving **local and global structures**.
- **HDBSCAN** clusters the embeddings into meaningful groups without requiring a predefined number of clusters.
- **Why Clustering?** It allows us to group **semantically similar trials** without labels, which helps in generating high-quality training pairs for fine-tuning.

### **3️⃣ Generating Training Pairs for Fine-Tuning**
- **Anchor & Candidate Sampling:**  
  - Each **cluster** acts as a natural label.
  - A fraction of documents is **sampled as "anchors"**, and the remaining documents in the same cluster are **treated as candidates**.
  - **Noise points are ignored** to maintain data quality.
- **Pair Construction:**  
  - A **FAISS index** is built for candidates to enable **efficient nearest-neighbor search**.
  - **For each anchor, the top-K nearest candidates** are retrieved, forming **(anchor, candidate) pairs**.

### **4️⃣ Fine-Tuning with Multiple Negatives Symmetric Ranking Loss**
- **SentenceTransformer Model Adaptation:**  
  - Fine-tuned on (anchor, candidate) pairs to capture **domain-specific relationships**.
  - Uses **Multiple Negatives Symmetric Ranking Loss** to:
    - Bring **positive pairs** closer in embedding space.
    - Implicitly treat other batch samples as **negatives**.
- **Why This Works?**
  - Helps in learning **better semantic representations** tailored for clinical trial retrieval.
  - Overcomes the lack of **manually labeled negative pairs** by using implicit negatives.

### **5️⃣ Re-Encoding & Retrieval**
- After fine-tuning, **all clinical trial text chunks** are re-encoded using the improved SentenceTransformer model.
- A **FAISS index** is built for fast **semantic search**.
- During inference:
  - **User queries** are encoded using the fine-tuned model.
  - The nearest **top-K results** are retrieved from the FAISS index.

---

## **Evaluation Metrics 📊**
To ensure high-quality clustering and retrieval, we evaluated:
- **Clustering Quality:**
  - **Silhouette Score**: 0.1025 (moderate cohesion and separation)
  - **Davies-Bouldin Index**: 2.323 (lower is better; needs improvement)
  - **Calinski-Harabasz Index**: 2404.28 (higher indicates better-separated clusters)

### **Model Performance & Limitations**
✅ **Strengths:**
- Automated **unsupervised clustering** → eliminates the need for manual labels.
- Clusters generate **high-quality training pairs** → improves retrieval accuracy.
- **FAISS-based retrieval** ensures **scalability & speed**.

⚠ **Limitations:**
- Some **clusters overlap**, affecting **anchor-candidate pair quality**.
- Fine-tuning was **limited due to compute constraints** (not run on full data for 3+ epochs).
- Hyperparameters for **UMAP, HDBSCAN**, and **fine-tuning** need further optimization.


## **Acknowledgments 🎖️**
This project was developed as part of the **Novartis India Challenge (NEST)**.  
Our approach focuses on **explainability, domain adaptation, and scalable retrieval** for clinical trials.

