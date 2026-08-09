"""
visualize_cluster.py
Self-contained script for visualization and clustering of fungal spike features.
Loads features from outputs/features/waveform_features.npy, generates plots,
runs clustering, and saves results.

Run from project root:
    python src/analysis/visualize_cluster.py
"""

import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
import json

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ML libraries
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap

# ============================================
# CONFIGURATION
# ============================================
FEATURES_FILE = Path("outputs/features/waveform_features.npy")
OUTPUT_DIR = Path("outputs/features/visualizations")
FEATURE_NAMES = [
    'amplitude', 'spike_width_fwhm_ms', 'asymmetry_index', 
    'area_under_curve', 'snr', 'time_to_pp_ms', 
    'pp_duration_ms', 'fall_rise_contrast'
]

# Clustering parameters
KMEANS_RANGE = range(2, 16)
GMM_RANGE = range(1, 16)
DBSCAN_EPS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
DBSCAN_MIN_SAMPLES = [5, 10, 15, 20]

# Dimensionality reduction
TSNE_PERPLEXITY = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# ============================================
# LOAD DATA
# ============================================
def load_features():
    """Load extracted features from numpy file."""
    print("=" * 60)
    print("Loading Features")
    print("=" * 60)
    
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    
    data = np.load(FEATURES_FILE, allow_pickle=True).item()
    
    features = data['features']
    feature_names = data['feature_names']
    channel_ids = data['channel_ids']
    spike_times = data['spike_times']
    
    print(f"Loaded {features.shape[0]} spikes × {features.shape[1]} features")
    print(f"Feature names: {feature_names}")
    
    return features, feature_names, channel_ids, spike_times

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def create_output_dirs():
    """Create output directory structure."""
    dirs = {
        'main': OUTPUT_DIR,
        'pairplots': OUTPUT_DIR / 'pairplots',
        'distributions': OUTPUT_DIR / 'distributions',
        'pca': OUTPUT_DIR / 'pca',
        'tsne': OUTPUT_DIR / 'tsne',
        'umap': OUTPUT_DIR / 'umap',
        'clustering': OUTPUT_DIR / 'clustering',
        'spatial': OUTPUT_DIR / 'spatial',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def plot_feature_distributions(features, feature_names, output_dir):
    """Plot histograms with KDE for all features."""
    print("\nGenerating feature distributions...")
    
    n_features = len(feature_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        data = features[:, i]
        data = data[~np.isnan(data)]
        
        # Histogram with KDE
        ax.hist(data, bins=40, density=True, alpha=0.6, color='steelblue', label='Histogram')
        
        # KDE
        kde_x = np.linspace(data.min(), data.max(), 200)
        kde = gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
        
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    
    plt.suptitle('Feature Distributions with KDE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_features_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'all_features_distributions.png'}")

def plot_pair_scatter(features, feature_names, output_dir, sample_n=5000):
    """Create pairwise scatter plots for all feature combinations."""
    print("\nGenerating pair scatter plots...")
    
    n_features = len(feature_names)
    
    # Subsample for performance
    if features.shape[0] > sample_n:
        idx = np.random.choice(features.shape[0], sample_n, replace=False)
        data = features[idx]
    else:
        data = features
    
    fig, axes = plt.subplots(n_features, n_features, figsize=(20, 20))
    
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(data[:, i], bins=30, color='steelblue', alpha=0.7)
                ax.set_ylabel('Count', fontsize=8)
            else:
                # Off-diagonal: scatter
                ax.scatter(data[:, j], data[:, i], s=1, alpha=0.3, c='steelblue')
            
            if i == n_features - 1:
                ax.set_xlabel(feature_names[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(feature_names[i], fontsize=8)
            
            ax.tick_params(labelsize=6)
    
    plt.suptitle('Pairwise Feature Scatter Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pair_scatter_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'pair_scatter_matrix.png'}")

def plot_pca_scatter(features, feature_names, output_dir):
    """PCA scatter plots with variance explained."""
    print("\nGenerating PCA visualizations...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)
    
    # Plot PC1 vs PC2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], s=2, alpha=0.5, c=features[:, 0], cmap='viridis')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax1.set_title('PCA: PC1 vs PC2 (colored by amplitude)', fontsize=14)
    plt.colorbar(scatter1, ax=ax1, label='Amplitude (μV)')
    ax1.spines[['top', 'right']].set_visible(False)
    
    # Plot PC2 vs PC3
    scatter2 = ax2.scatter(pca_result[:, 1], pca_result[:, 2], s=2, alpha=0.5, c=features[:, 1], cmap='plasma')
    ax2.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=12)
    ax2.set_title('PCA: PC2 vs PC3 (colored by spike width)', fontsize=14)
    plt.colorbar(scatter2, ax=ax2, label='Spike Width (ms)')
    ax2.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # PCA explained variance
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    pca_full = PCA()
    pca_full.fit(features_scaled)
    
    ax.plot(range(1, 9), np.cumsum(pca_full.explained_variance_ratio_), 'bo-', linewidth=2)
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax.axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_explained_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved PCA plots to {output_dir}")
    
    return pca_result, pca

def plot_tsne(features, output_dir, perplexity=TSNE_PERPLEXITY):
    """t-SNE visualization."""
    print(f"\nGenerating t-SNE (perplexity={perplexity})...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    tsne_result = tsne.fit_transform(features_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], s=2, alpha=0.5, c=features[:, 0], cmap='viridis')
    ax1.set_title(f't-SNE (perplexity={perplexity}, colored by amplitude)', fontsize=14)
    ax1.set_xlabel('t-SNE 1', fontsize=12)
    ax1.set_ylabel('t-SNE 2', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Amplitude (μV)')
    ax1.spines[['top', 'right']].set_visible(False)
    
    scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], s=2, alpha=0.5, c=features[:, 4], cmap='plasma')
    ax2.set_title(f't-SNE (perplexity={perplexity}, colored by SNR)', fontsize=14)
    ax2.set_xlabel('t-SNE 1', fontsize=12)
    ax2.set_ylabel('t-SNE 2', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='SNR')
    ax2.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'tsne_scatter.png'}")
    
    return tsne_result

def plot_umap(features, output_dir, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST):
    """UMAP visualization."""
    print(f"\nGenerating UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_result = reducer.fit_transform(features_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = ax1.scatter(umap_result[:, 0], umap_result[:, 1], s=2, alpha=0.5, c=features[:, 0], cmap='viridis')
    ax1.set_title(f'UMAP (n_neighbors={n_neighbors}, colored by amplitude)', fontsize=14)
    ax1.set_xlabel('UMAP 1', fontsize=12)
    ax1.set_ylabel('UMAP 2', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Amplitude (μV)')
    ax1.spines[['top', 'right']].set_visible(False)
    
    scatter2 = ax2.scatter(umap_result[:, 0], umap_result[:, 1], s=2, alpha=0.5, c=features[:, 4], cmap='plasma')
    ax2.set_title(f'UMAP (n_neighbors={n_neighbors}, colored by SNR)', fontsize=14)
    ax2.set_xlabel('UMAP 1', fontsize=12)
    ax2.set_ylabel('UMAP 2', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='SNR')
    ax2.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'umap_scatter.png'}")
    
    return umap_result

def plot_spatial_heatmap(features, channel_ids, feature_names, output_dir):
    """Plot average feature values on 8x8 MEA grid."""
    print("\nGenerating spatial heatmaps...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        # Compute mean feature value per channel
        grid = np.full((8, 8), np.nan)
        
        for ch in range(64):
            mask = channel_ids == ch
            if np.any(mask):
                grid[ch // 8, ch % 8] = np.mean(features[mask, i])
        
        im = ax.imshow(grid, cmap='hot', interpolation='nearest')
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Spatial Distribution of Features (8×8 MEA Grid)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'spatial_heatmaps.png'}")

# ============================================
# CLUSTERING FUNCTIONS
# ============================================
def evaluate_kmeans(features, k_range=KMEANS_RANGE):
    """Run K-Means with different k and compute metrics."""
    print("\nEvaluating K-Means (k=2-15)...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    metrics = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features_scaled)
        
        metrics['k'].append(k)
        metrics['inertia'].append(km.inertia_)
        metrics['silhouette'].append(silhouette_score(features_scaled, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(features_scaled, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(features_scaled, labels))
        
        print(f"  k={k}: Sil={metrics['silhouette'][-1]:.3f}, CH={metrics['calinski_harabasz'][-1]:.1f}, DB={metrics['davies_bouldin'][-1]:.3f}")
    
    return metrics

def evaluate_gmm(features, n_range=GMM_RANGE):
    """Run GMM with different n_components and compute BIC/AIC."""
    print("\nEvaluating GMM (n_components=1-15)...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    metrics = {
        'n_components': [],
        'bic': [],
        'aic': [],
        'silhouette': []
    }
    
    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
        gmm.fit(features_scaled)
        labels = gmm.predict(features_scaled)
        
        metrics['n_components'].append(n)
        metrics['bic'].append(gmm.bic(features_scaled))
        metrics['aic'].append(gmm.aic(features_scaled))
        
        # Silhouette score requires at least 2 clusters
        if n > 1:
            metrics['silhouette'].append(silhouette_score(features_scaled, labels))
        else:
            metrics['silhouette'].append(np.nan)
        
        print(f"  n={n}: BIC={metrics['bic'][-1]:.1f}, AIC={metrics['aic'][-1]:.1f}, Sil={metrics['silhouette'][-1]:.3f}")
        
    return metrics

def plot_clustering_metrics(kmeans_metrics, gmm_metrics, output_dir):
    """Plot clustering evaluation metrics."""
    print("\nPlotting clustering metrics...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # K-Means metrics
    axes[0, 0].plot(kmeans_metrics['k'], kmeans_metrics['inertia'], 'bo-', linewidth=2)
    axes[0, 0].set_xlabel('k (number of clusters)', fontsize=10)
    axes[0, 0].set_ylabel('Inertia', fontsize=10)
    axes[0, 0].set_title('K-Means: Elbow Curve', fontsize=12)
    axes[0, 0].spines[['top', 'right']].set_visible(False)
    
    axes[0, 1].plot(kmeans_metrics['k'], kmeans_metrics['silhouette'], 'go-', linewidth=2)
    axes[0, 1].set_xlabel('k', fontsize=10)
    axes[0, 1].set_ylabel('Silhouette Score', fontsize=10)
    axes[0, 1].set_title('K-Means: Silhouette Score', fontsize=12)
    axes[0, 1].spines[['top', 'right']].set_visible(False)
    
    axes[0, 2].plot(kmeans_metrics['k'], kmeans_metrics['calinski_harabasz'], 'mo-', linewidth=2, label='Calinski-Harabasz')
    axes[0, 2].plot(kmeans_metrics['k'], kmeans_metrics['davies_bouldin'], 'co-', linewidth=2, label='Davies-Bouldin')
    axes[0, 2].set_xlabel('k', fontsize=10)
    axes[0, 2].set_ylabel('Score', fontsize=10)
    axes[0, 2].set_title('K-Means: CH & DB Indices', fontsize=12)
    axes[0, 2].legend()
    axes[0, 2].spines[['top', 'right']].set_visible(False)
    
    # GMM metrics
    axes[1, 0].plot(gmm_metrics['n_components'], gmm_metrics['bic'], 'ro-', linewidth=2)
    axes[1, 0].set_xlabel('n_components', fontsize=10)
    axes[1, 0].set_ylabel('BIC', fontsize=10)
    axes[1, 0].set_title('GMM: BIC', fontsize=12)
    axes[1, 0].spines[['top', 'right']].set_visible(False)
    
    axes[1, 1].plot(gmm_metrics['n_components'], gmm_metrics['aic'], 'yo-', linewidth=2)
    axes[1, 1].set_xlabel('n_components', fontsize=10)
    axes[1, 1].set_ylabel('AIC', fontsize=10)
    axes[1, 1].set_title('GMM: AIC', fontsize=12)
    axes[1, 1].spines[['top', 'right']].set_visible(False)
    
    axes[1, 2].plot(gmm_metrics['n_components'], gmm_metrics['silhouette'], 'go-', linewidth=2)
    axes[1, 2].set_xlabel('n_components', fontsize=10)
    axes[1, 2].set_ylabel('Silhouette Score', fontsize=10)
    axes[1, 2].set_title('GMM: Silhouette Score', fontsize=12)
    axes[1, 2].spines[['top', 'right']].set_visible(False)
    
    plt.suptitle('Clustering Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'clustering_metrics.png'}")

def run_dbscan(features, eps_values=DBSCAN_EPS, min_samples_values=DBSCAN_MIN_SAMPLES):
    """Run DBSCAN with different parameters."""
    print("\nEvaluating DBSCAN...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {}
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            
            key = f"eps{eps}_min{min_samples}"
            results[key] = {
                'eps': eps,
                'min_samples': min_samples,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(labels)
            }
            
            print(f"  eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise ({n_noise/len(labels):.1%})")
    
    return results

def apply_clustering(features, method='kmeans', n_clusters=3, dbscan_params=None):
    """Apply clustering and return labels."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(features_scaled)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        model.fit(features_scaled)
        labels = model.predict(features_scaled)
    elif method == 'dbscan':
        eps = dbscan_params.get('eps', 0.5)
        min_samples = dbscan_params.get('min_samples', 10)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(features_scaled)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels, scaler, features_scaled

def visualize_clusters_2d(features_scaled, labels, method_name, output_dir, reducer='pca'):
    """Visualize clusters in 2D (PCA/t-SNE/UMAP)."""
    print(f"\nVisualizing {method_name} clusters...")
    
    if reducer == 'pca':
        reducer_obj = PCA(n_components=2)
        result = reducer_obj.fit_transform(features_scaled)
        xlabel, ylabel = f'PC1', f'PC2'
    elif reducer == 'tsne':
        reducer_obj = TSNE(n_components=2, perplexity=30, random_state=42)
        result = reducer_obj.fit_transform(features_scaled)
        xlabel, ylabel = 't-SNE 1', 't-SNE 2'
    else:
        raise ValueError(f"Unknown reducer: {reducer}")
    
    # Unique labels (excluding noise if DBSCAN)
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    has_noise = -1 in unique_labels
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    if has_noise:
        colors = np.vstack([[0.7, 0.7, 0.7, 1.0], colors])  # Gray for noise
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(result[mask, 0], result[mask, 1], s=2, c='gray', alpha=0.3, label='Noise')
        else:
            ax.scatter(result[mask, 0], result[mask, 1], s=2, c=[colors[i % len(colors)]], 
                      alpha=0.6, label=f'Cluster {label}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{method_name} Clusters ({n_clusters} clusters) - {reducer.upper()}', fontsize=14, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    
    if n_clusters <= 10:
        ax.legend(markerscale=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'clusters_{method_name}_{reducer}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / f'clusters_{method_name}_{reducer}.png'}")

def save_cluster_results(features, labels, feature_names, method_name, output_dir, channel_ids=None, spike_times=None):
    """Save cluster assignments and statistics."""
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Compute cluster statistics
    stats = {}
    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        cluster_features = features[mask]
        
        stats[f'cluster_{label}'] = {
            'size': int(np.sum(mask)),
            'feature_means': cluster_features.mean(axis=0).tolist(),
            'feature_stds': cluster_features.std(axis=0).tolist(),
            'feature_names': feature_names
        }
        
        if channel_ids is not None:
            stats[f'cluster_{label}']['channel_distribution'] = np.bincount(channel_ids[mask], minlength=64).tolist()
    
    # Save to JSON
    output = {
        'method': method_name,
        'n_clusters': n_clusters,
        'total_spikes': len(labels),
        'unique_labels': [int(l) for l in unique_labels],
        'labels': labels.tolist(),
        'cluster_statistics': stats
    }
    
    with open(output_dir / f'cluster_results_{method_name}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Also save labels to numpy file for easy loading
    np.save(output_dir / f'cluster_labels_{method_name}.npy', labels)
    
    print(f"  Saved cluster results for {method_name}")

# ============================================
# MAIN PIPELINE
# ============================================
def main():
    print("=" * 60)
    print("Fungal Spike Visualization & Clustering Pipeline")
    print("=" * 60)
    
    # Load data
    features, feature_names, channel_ids, spike_times = load_features()
    
    # Create output directories
    dirs = create_output_dirs()
    
    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 1: VISUALIZATION")
    print("=" * 60)
    
    # Feature distributions
    plot_feature_distributions(features, feature_names, dirs['distributions'])
    
    # Pairwise scatter (skip if too many spikes for performance)
    if features.shape[0] <= 10000:
        plot_pair_scatter(features, feature_names, dirs['pairplots'])
    else:
        print("\nSkipping pair scatter (too many spikes, use sample instead)")
    
    # PCA
    pca_result, pca = plot_pca_scatter(features, feature_names, dirs['pca'])
    
    # t-SNE
    tsne_result = plot_tsne(features, dirs['tsne'])
    
    # UMAP
    try:
        umap_result = plot_umap(features, dirs['umap'])
    except ImportError:
        print("\nUMAP not installed, skipping UMAP visualization")
        print("Install with: pip install umap-learn")
    
    # Spatial heatmaps
    plot_spatial_heatmap(features, channel_ids, feature_names, dirs['spatial'])
    
    # ==========================================
    # CLUSTERING
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 2: CLUSTERING")
    print("=" * 60)
    
    # Evaluate K-Means
    kmeans_metrics = evaluate_kmeans(features)
    
    # Evaluate GMM
    gmm_metrics = evaluate_gmm(features)
    
    # Plot clustering metrics
    plot_clustering_metrics(kmeans_metrics, gmm_metrics, dirs['clustering'])
    
    # DBSCAN
    dbscan_results = run_dbscan(features)
    
    # ==========================================
    # SELECT BEST CLUSTERING & VISUALIZE
    # ==========================================
    print("\n" + "=" * 60)
    print("APPLYING CLUSTERING METHODS")
    print("=" * 60)
    
    # Find best k for K-Means (highest silhouette)
    best_k = kmeans_metrics['k'][np.argmax(kmeans_metrics['silhouette'])]
    print(f"\nBest K-Means k (by silhouette): {best_k}")
    
    # Find best n for GMM (lowest BIC)
    best_n = gmm_metrics['n_components'][np.argmin(gmm_metrics['bic'])]
    print(f"Best GMM n_components (by BIC): {best_n}")
    
    # Apply K-Means
    kmeans_labels, kmeans_scaler, kmeans_scaled = apply_clustering(features, method='kmeans', n_clusters=best_k)
    visualize_clusters_2d(kmeans_scaled, kmeans_labels, f'kmeans_k{best_k}', dirs['clustering'], reducer='pca')
    visualize_clusters_2d(kmeans_scaled, kmeans_labels, f'kmeans_k{best_k}', dirs['clustering'], reducer='tsne')
    save_cluster_results(features, kmeans_labels, feature_names, f'kmeans_k{best_k}', dirs['clustering'], channel_ids, spike_times)
    
    # Apply GMM
    gmm_labels, gmm_scaler, gmm_scaled = apply_clustering(features, method='gmm', n_clusters=best_n)
    visualize_clusters_2d(gmm_scaled, gmm_labels, f'gmm_n{best_n}', dirs['clustering'], reducer='pca')
    visualize_clusters_2d(gmm_scaled, gmm_labels, f'gmm_n{best_n}', dirs['clustering'], reducer='tsne')
    save_cluster_results(features, gmm_labels, feature_names, f'gmm_n{best_n}', dirs['clustering'], channel_ids, spike_times)
    
    # Apply DBSCAN with best params (lowest noise ratio with reasonable cluster count)
    best_dbscan_key = None
    best_dbscan_score = -1
    for key, result in dbscan_results.items():
        if result['n_clusters'] >= 2 and result['n_clusters'] <= 10:
            score = result['n_clusters'] / (1 + result['noise_ratio'])
            if score > best_dbscan_score:
                best_dbscan_score = score
                best_dbscan_key = key
    
    if best_dbscan_key:
        dbscan_labels = dbscan_results[best_dbscan_key]['labels']
        _, dbscan_scaler, dbscan_scaled = apply_clustering(
            features, method='dbscan', 
            dbscan_params={'eps': dbscan_results[best_dbscan_key]['eps'], 
                          'min_samples': dbscan_results[best_dbscan_key]['min_samples']}
        )
        visualize_clusters_2d(dbscan_scaled, dbscan_labels, f'dbscan_{best_dbscan_key}', dirs['clustering'], reducer='pca')
        save_cluster_results(features, dbscan_labels, feature_names, f'dbscan_{best_dbscan_key}', dirs['clustering'], channel_ids, spike_times)
        print(f"\nBest DBSCAN: {best_dbscan_key}")
    
    # ==========================================
    # SAVE UPDATED FEATURES FILE
    # ==========================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Load original data and add cluster labels
    original_data = np.load(FEATURES_FILE, allow_pickle=True).item()
    original_data['kmeans_labels'] = kmeans_labels
    original_data['gmm_labels'] = gmm_labels
    original_data['best_k_kmeans'] = int(best_k)
    original_data['best_n_gmm'] = int(best_n)
    
    np.save(FEATURES_FILE, original_data)
    print(f"Updated features file with cluster labels: {FEATURES_FILE}")
    
    print("\n" + "=" * 60)
    print("DONE! All visualizations and clustering complete.")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGenerated:")
    print(f"  - Feature distributions: {dirs['distributions']}")
    print(f"  - PCA/t-SNE/UMAP plots: {dirs['pca']}, {dirs['tsne']}, {dirs['umap']}")
    print(f"  - Clustering results: {dirs['clustering']}")
    print(f"  - Spatial heatmaps: {dirs['spatial']}")

if __name__ == "__main__":
    main()