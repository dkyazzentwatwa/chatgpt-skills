#!/usr/bin/env python3
"""
Clustering Analyzer - Cluster data using multiple algorithms.
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


class ClusteringAnalyzer:
    """Analyze and cluster data using multiple algorithms."""

    def __init__(self):
        """Initialize the clustering analyzer."""
        self.data: Optional[pd.DataFrame] = None
        self.scaled_data: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.columns: List[str] = []
        self.model = None

    def load_csv(self, filepath: str, columns: List[str] = None) -> 'ClusteringAnalyzer':
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            columns: Columns to use for clustering (all numeric if None)

        Returns:
            Self for method chaining
        """
        df = pd.read_csv(filepath)
        return self.load_dataframe(df, columns)

    def load_dataframe(self, df: pd.DataFrame, columns: List[str] = None) -> 'ClusteringAnalyzer':
        """
        Load data from DataFrame.

        Args:
            df: Input DataFrame
            columns: Columns to use for clustering

        Returns:
            Self for method chaining
        """
        if columns:
            self.data = df[columns].copy()
        else:
            # Select numeric columns
            self.data = df.select_dtypes(include=[np.number]).copy()

        self.columns = list(self.data.columns)

        # Handle missing values
        self.data = self.data.fillna(self.data.mean())

        # Scale data
        self.scaled_data = self.scaler.fit_transform(self.data)

        return self

    def kmeans(self, n_clusters: int, **kwargs) -> Dict:
        """
        Perform K-Means clustering.

        Args:
            n_clusters: Number of clusters
            **kwargs: Additional KMeans parameters

        Returns:
            Clustering results
        """
        if self.scaled_data is None:
            raise ValueError("No data loaded")

        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
        self.labels = self.model.fit_predict(self.scaled_data)

        # Calculate silhouette score
        sil_score = silhouette_score(self.scaled_data, self.labels) if n_clusters > 1 else 0

        # Cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        # Centroids (inverse transform to original scale)
        centroids = self.scaler.inverse_transform(self.model.cluster_centers_)

        return {
            "labels": self.labels.tolist(),
            "n_clusters": n_clusters,
            "silhouette_score": float(sil_score),
            "inertia": float(self.model.inertia_),
            "cluster_sizes": cluster_sizes,
            "centroids": centroids.tolist()
        }

    def dbscan(self, eps: float = 0.5, min_samples: int = 5) -> Dict:
        """
        Perform DBSCAN clustering.

        Args:
            eps: Maximum distance between points
            min_samples: Minimum samples in neighborhood

        Returns:
            Clustering results
        """
        if self.scaled_data is None:
            raise ValueError("No data loaded")

        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = self.model.fit_predict(self.scaled_data)

        # Number of clusters (excluding noise)
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)

        # Calculate silhouette score (excluding noise)
        if n_clusters > 1:
            mask = self.labels != -1
            if mask.sum() > 0:
                sil_score = silhouette_score(
                    self.scaled_data[mask],
                    self.labels[mask]
                )
            else:
                sil_score = 0
        else:
            sil_score = 0

        # Cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        return {
            "labels": self.labels.tolist(),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette_score": float(sil_score),
            "cluster_sizes": cluster_sizes
        }

    def hierarchical(self, n_clusters: int, linkage_method: str = "ward") -> Dict:
        """
        Perform hierarchical clustering.

        Args:
            n_clusters: Number of clusters
            linkage_method: Linkage method ("ward", "complete", "average", "single")

        Returns:
            Clustering results
        """
        if self.scaled_data is None:
            raise ValueError("No data loaded")

        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        self.labels = self.model.fit_predict(self.scaled_data)

        # Calculate silhouette score
        sil_score = silhouette_score(self.scaled_data, self.labels) if n_clusters > 1 else 0

        # Cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        return {
            "labels": self.labels.tolist(),
            "n_clusters": n_clusters,
            "silhouette_score": float(sil_score),
            "cluster_sizes": cluster_sizes,
            "linkage": linkage_method
        }

    def find_optimal_clusters(self, max_k: int = 10) -> Dict:
        """
        Find optimal number of clusters using elbow method.

        Args:
            max_k: Maximum number of clusters to try

        Returns:
            Dictionary with optimal k and metrics
        """
        if self.scaled_data is None:
            raise ValueError("No data loaded")

        inertias = []
        silhouettes = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)

            inertias.append(float(kmeans.inertia_))
            silhouettes.append(float(silhouette_score(self.scaled_data, labels)))

        # Find elbow point using second derivative
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow_idx = np.argmax(diffs2) + 2  # +2 because we start at k=2
            optimal_k = elbow_idx + 2
        else:
            optimal_k = 2

        # Also consider silhouette score
        best_sil_k = silhouettes.index(max(silhouettes)) + 2

        return {
            "optimal_k": optimal_k,
            "best_silhouette_k": best_sil_k,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "k_range": list(range(2, max_k + 1))
        }

    def elbow_plot(self, output: str, max_k: int = 10) -> str:
        """
        Generate elbow plot.

        Args:
            output: Output file path
            max_k: Maximum number of clusters

        Returns:
            Output path
        """
        optimal = self.find_optimal_clusters(max_k)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        k_range = optimal["k_range"]

        # Inertia plot
        ax1.plot(k_range, optimal["inertias"], 'bo-')
        ax1.axvline(x=optimal["optimal_k"], color='r', linestyle='--',
                   label=f'Elbow at k={optimal["optimal_k"]}')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.legend()

        # Silhouette plot
        ax2.plot(k_range, optimal["silhouettes"], 'go-')
        ax2.axvline(x=optimal["best_silhouette_k"], color='r', linestyle='--',
                   label=f'Best at k={optimal["best_silhouette_k"]}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def silhouette_score_value(self) -> float:
        """Get silhouette score for current clustering."""
        if self.labels is None:
            raise ValueError("Run clustering first")

        if len(set(self.labels)) <= 1:
            return 0.0

        return float(silhouette_score(self.scaled_data, self.labels))

    def cluster_statistics(self) -> Dict:
        """
        Get detailed cluster statistics.

        Returns:
            Dictionary with cluster statistics
        """
        if self.labels is None:
            raise ValueError("Run clustering first")

        df_labeled = self.data.copy()
        df_labeled['cluster_label'] = self.labels

        # Cluster sizes
        cluster_sizes = df_labeled['cluster_label'].value_counts().to_dict()

        # Cluster means
        cluster_means = {}
        cluster_std = {}

        for cluster in sorted(set(self.labels)):
            if cluster == -1:  # Skip noise in DBSCAN
                continue

            cluster_data = df_labeled[df_labeled['cluster_label'] == cluster][self.columns]
            cluster_means[int(cluster)] = cluster_data.mean().to_dict()
            cluster_std[int(cluster)] = cluster_data.std().to_dict()

        return {
            "n_clusters": len([c for c in set(self.labels) if c != -1]),
            "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
            "cluster_means": cluster_means,
            "cluster_std": cluster_std,
            "overall_silhouette": self.silhouette_score_value()
        }

    def plot_clusters(self, output: str, dimensions: List[str] = None) -> str:
        """
        Plot clusters in 2D.

        Args:
            output: Output file path
            dimensions: Two dimensions to plot (uses PCA if None)

        Returns:
            Output path
        """
        if self.labels is None:
            raise ValueError("Run clustering first")

        # Prepare data for plotting
        if dimensions and len(dimensions) >= 2:
            x_idx = self.columns.index(dimensions[0])
            y_idx = self.columns.index(dimensions[1])
            x_data = self.scaled_data[:, x_idx]
            y_data = self.scaled_data[:, y_idx]
            x_label = dimensions[0]
            y_label = dimensions[1]
        else:
            # Use PCA for dimensionality reduction
            if self.scaled_data.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(self.scaled_data)
                x_data = reduced[:, 0]
                y_data = reduced[:, 1]
                x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
                y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
            else:
                x_data = self.scaled_data[:, 0]
                y_data = self.scaled_data[:, 1] if self.scaled_data.shape[1] > 1 else np.zeros_like(x_data)
                x_label = self.columns[0] if self.columns else "Feature 1"
                y_label = self.columns[1] if len(self.columns) > 1 else "Feature 2"

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Color map
        unique_labels = sorted(set(self.labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            if label == -1:
                color = 'black'
                marker = 'x'
                label_name = 'Noise'
            else:
                color = colors[i]
                marker = 'o'
                label_name = f'Cluster {label}'

            ax.scatter(x_data[mask], y_data[mask], c=[color], marker=marker,
                      label=label_name, alpha=0.7, s=50)

        # Plot centroids for K-Means
        if hasattr(self.model, 'cluster_centers_'):
            if dimensions and len(dimensions) >= 2:
                centers = self.model.cluster_centers_
                ax.scatter(centers[:, x_idx], centers[:, y_idx], c='red',
                          marker='*', s=200, label='Centroids', edgecolors='black')
            elif self.scaled_data.shape[1] > 2:
                centers_pca = pca.transform(self.model.cluster_centers_)
                ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red',
                          marker='*', s=200, label='Centroids', edgecolors='black')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Cluster Visualization')
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_dendrogram(self, output: str) -> str:
        """
        Plot dendrogram for hierarchical clustering.

        Args:
            output: Output file path

        Returns:
            Output path
        """
        if self.scaled_data is None:
            raise ValueError("No data loaded")

        # Compute linkage matrix
        Z = linkage(self.scaled_data, method='ward')

        fig, ax = plt.subplots(figsize=(12, 8))

        dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
                  leaf_rotation=90, leaf_font_size=8)

        ax.set_xlabel('Sample Index or Cluster Size')
        ax.set_ylabel('Distance')
        ax.set_title('Hierarchical Clustering Dendrogram')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_silhouette(self, output: str) -> str:
        """
        Plot silhouette diagram.

        Args:
            output: Output file path

        Returns:
            Output path
        """
        if self.labels is None:
            raise ValueError("Run clustering first")

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)

        if n_clusters <= 1:
            raise ValueError("Need at least 2 clusters for silhouette plot")

        # Exclude noise points for DBSCAN
        mask = self.labels != -1
        sample_silhouette_values = silhouette_samples(
            self.scaled_data[mask],
            self.labels[mask]
        )
        silhouette_avg = float(np.mean(sample_silhouette_values))

        fig, ax = plt.subplots(figsize=(10, 8))

        y_lower = 10
        labels_masked = self.labels[mask]

        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[labels_masked == i]
            cluster_silhouette_values.sort()

            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.tab10(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0,
                            cluster_silhouette_values, facecolor=color,
                            edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--",
                  label=f'Average: {silhouette_avg:.3f}')

        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.set_title('Silhouette Plot')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def get_labels(self) -> List[int]:
        """Get cluster labels."""
        if self.labels is None:
            raise ValueError("Run clustering first")
        return self.labels.tolist()

    def to_dataframe(self) -> pd.DataFrame:
        """Get data with cluster labels."""
        if self.labels is None:
            raise ValueError("Run clustering first")

        df = self.data.copy()
        df['cluster_label'] = self.labels
        return df

    def save_labeled(self, output: str) -> str:
        """Save data with cluster labels to CSV."""
        df = self.to_dataframe()
        df.to_csv(output, index=False)
        return output


def main():
    parser = argparse.ArgumentParser(
        description="Clustering Analyzer - Cluster data using multiple algorithms"
    )

    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--columns", "-c", help="Columns to use (comma-separated)")
    parser.add_argument("--method", "-m", choices=["kmeans", "dbscan", "hierarchical"],
                       default="kmeans", help="Clustering method")
    parser.add_argument("--clusters", "-k", type=int, default=3,
                       help="Number of clusters (for kmeans/hierarchical)")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps parameter")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--find-optimal", action="store_true",
                       help="Find optimal number of clusters")
    parser.add_argument("--plot", "-p", help="Output plot file")
    parser.add_argument("--output", "-o", help="Output CSV file with labels")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    analyzer = ClusteringAnalyzer()

    columns = args.columns.split(',') if args.columns else None
    analyzer.load_csv(args.input, columns=columns)

    if args.find_optimal:
        optimal = analyzer.find_optimal_clusters()

        if args.json:
            print(json.dumps(optimal, indent=2))
        else:
            print(f"\n=== Optimal Clusters Analysis ===")
            print(f"Elbow method suggests: k = {optimal['optimal_k']}")
            print(f"Best silhouette score at: k = {optimal['best_silhouette_k']}")
            print(f"\nSilhouette scores by k:")
            for k, sil in zip(optimal['k_range'], optimal['silhouettes']):
                print(f"  k={k}: {sil:.4f}")

        if args.plot:
            analyzer.elbow_plot(args.plot)
            print(f"\nElbow plot saved to: {args.plot}")

    else:
        # Run clustering
        if args.method == "kmeans":
            result = analyzer.kmeans(args.clusters)
        elif args.method == "dbscan":
            result = analyzer.dbscan(eps=args.eps, min_samples=args.min_samples)
        else:
            result = analyzer.hierarchical(args.clusters)

        if args.json:
            # Add statistics to result
            result["statistics"] = analyzer.cluster_statistics()
            print(json.dumps(result, indent=2))
        else:
            print(f"\n=== {args.method.upper()} Clustering Results ===")
            print(f"Number of clusters: {result.get('n_clusters', 'N/A')}")
            if 'n_noise' in result:
                print(f"Noise points: {result['n_noise']}")
            print(f"Silhouette score: {result.get('silhouette_score', 0):.4f}")
            print(f"\nCluster sizes:")
            for cluster, size in result['cluster_sizes'].items():
                label = "Noise" if cluster == -1 else f"Cluster {cluster}"
                print(f"  {label}: {size}")

        if args.plot:
            analyzer.plot_clusters(args.plot)
            print(f"\nCluster plot saved to: {args.plot}")

        if args.output:
            analyzer.save_labeled(args.output)
            print(f"Labeled data saved to: {args.output}")


if __name__ == "__main__":
    main()
