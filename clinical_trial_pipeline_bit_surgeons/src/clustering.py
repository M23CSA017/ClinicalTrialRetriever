import matplotlib.pyplot as plt
import umap
import hdbscan
import numpy as np
import os


def perform_clustering(embeddings, umap_params, hdbscan_params, save_path, seed=42):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Ensure save_path exists
    os.makedirs(save_path, exist_ok=True)

    # UMAP Dimensionality Reduction
    umap_reducer = umap.UMAP(**umap_params, random_state=seed)
    cluster_embedding = umap_reducer.fit_transform(embeddings)

    # HDBSCAN Clustering
    hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params)
    cluster_labels = hdbscan_model.fit_predict(cluster_embedding)

    # Plot the embedding
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label == -1:
            # Noise points
            label_mask = cluster_labels == label
            plt.scatter(
                cluster_embedding[label_mask, 0],
                cluster_embedding[label_mask, 1],
                c='gray',
                label='Noise',
                alpha=0.5,
                s=10
            )
        else:
            # Cluster points
            label_mask = cluster_labels == label
            plt.scatter(
                cluster_embedding[label_mask, 0],
                cluster_embedding[label_mask, 1],
                label=f'Cluster {label}',
                alpha=0.7,
                s=15
            )
    
    plt.title("UMAP Projection with HDBSCAN Clusters", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(loc="best", markerscale=2, fontsize=10)
    plt.grid(alpha=0.5)
    
    # Save the plot
    plot_path = os.path.join(save_path, "cluster_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Cluster plot saved at {plot_path}")
    
    return cluster_embedding, cluster_labels, hdbscan_model
