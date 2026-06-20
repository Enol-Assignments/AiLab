import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def LoadMallCustomers(path='Datasets/Mall_Customers.csv'):
    df = pd.read_csv(path)
    print(f'Shape: {df.shape}')
    print(f'\nColumns: {list(df.columns)}')
    print(f'\nData types:')
    print(df.dtypes)
    print(f'\nMissing values:')
    print(df.isnull().sum())
    display(df.head(10))
    return df


def PreprocessMall(df):
    le = LabelEncoder()
    df['Gender_encoded'] = le.fit_transform(df['Genre'])

    features = ['Gender_encoded', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f'Clustering features: {features}')
    print(f'Data shape: {X_scaled.shape}')
    print(f'Standardized mean: {X_scaled.mean(axis=0).round(2)}')
    print(f'Standardized std:  {X_scaled.std(axis=0).round(2)}')
    return X_scaled, scaler, features


def ReducePCA(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f'PCA explained variance ratio: {pca.explained_variance_ratio_.round(4)}')
    print(f'Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}')
    return X_pca, pca


def ReduceTSNE(X_scaled):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    return tsne.fit_transform(X_scaled)


def PlotDistributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df['Age'], bins=20, color='#377eb8', edgecolor='black', alpha=0.7)
    axes[0].set_title('Age Distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')

    axes[1].hist(df['Annual Income (k$)'], bins=20, color='#ff7f00', edgecolor='black', alpha=0.7)
    axes[1].set_title('Annual Income Distribution')
    axes[1].set_xlabel('Annual Income (k$)')

    axes[2].hist(df['Spending Score (1-100)'], bins=20, color='#4daf4a', edgecolor='black', alpha=0.7)
    axes[2].set_title('Spending Score Distribution')
    axes[2].set_xlabel('Spending Score (1-100)')

    fig.suptitle('Customer Feature Distributions', fontsize=14)
    plt.tight_layout()
    plt.show()


def PlotGenderDistribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    df['Genre'].value_counts().plot(kind='bar', ax=axes[0], color=['#e41a1c', '#377eb8'])
    axes[0].set_title('Gender Distribution')
    axes[0].set_ylabel('Count')

    df['Genre'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                      colors=['#e41a1c', '#377eb8'])
    axes[1].set_title('Gender Ratio')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()


def PlotIncomeVsSpending(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['Annual Income (k$)'],
                         df['Spending Score (1-100)'],
                         c=df['Age'], cmap='viridis',
                         edgecolors='black', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Age')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Annual Income vs Spending Score (color=Age)')
    plt.tight_layout()
    plt.show()


def PlotBoxPlots(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    features_to_check = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    colors = ['#377eb8', '#ff7f00', '#4daf4a']

    for idx, (feat, color) in enumerate(zip(features_to_check, colors)):
        axes[idx].boxplot(df[feat], patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.7))
        axes[idx].set_title(feat)
        axes[idx].set_ylabel('Value')

    fig.suptitle('Box Plot - Outlier Detection', fontsize=14)
    plt.tight_layout()
    plt.show()


def PlotPCAVariance(X_scaled):
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_, color='#377eb8', alpha=0.7)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA: Explained Variance per Component')

    axes[1].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
                 np.cumsum(pca_full.explained_variance_ratio_), 'o-', color='#ff7f00')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('PCA: Cumulative Explained Variance')
    axes[1].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def PlotDimComparison(X_pca, X_tsne, df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                               c=df['Age'], cmap='viridis',
                               edgecolors='black', s=40, alpha=0.7)
    plt.colorbar(scatter1, ax=axes[0], label='Age')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PCA (color=Age)')

    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                               c=df['Age'], cmap='viridis',
                               edgecolors='black', s=40, alpha=0.7)
    plt.colorbar(scatter2, ax=axes[1], label='Age')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('t-SNE (color=Age)')

    plt.tight_layout()
    plt.show()


def EvaluateKMeans(X_scaled, K_range=range(2, 11)):
    inertias, silhouettes, calinski, davies = [], [], [], []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        calinski.append(calinski_harabasz_score(X_scaled, labels))
        davies.append(davies_bouldin_score(X_scaled, labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(K_range, inertias, 'o-', linewidth=2, color='#377eb8')
    axes[0, 0].set_xlabel('K')
    axes[0, 0].set_ylabel('Inertia (SSE)')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(K_range, silhouettes, 'o-', linewidth=2, color='#ff7f00')
    axes[0, 1].set_xlabel('K')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(K_range, calinski, 'o-', linewidth=2, color='#4daf4a')
    axes[1, 0].set_xlabel('K')
    axes[1, 0].set_ylabel('Calinski-Harabasz Index')
    axes[1, 0].set_title('Calinski-Harabasz Index (higher is better)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(K_range, davies, 'o-', linewidth=2, color='#e41a1c')
    axes[1, 1].set_xlabel('K')
    axes[1, 1].set_ylabel('Davies-Bouldin Index')
    axes[1, 1].set_title('Davies-Bouldin Index (lower is better)')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('K-Means: Evaluation Metrics for Different K', fontsize=14)
    plt.tight_layout()
    plt.show()

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f'Best K by silhouette score = {best_k} (Score = {max(silhouettes):.4f})')
    return best_k, silhouettes


def RunClustering(X_scaled, k):
    results = {}

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    results['KMeans'] = kmeans.fit_predict(X_scaled)

    agg = AgglomerativeClustering(n_clusters=k)
    results['Agglomerative'] = agg.fit_predict(X_scaled)

    dbscan = DBSCAN(eps=1.2, min_samples=5)
    results['DBSCAN'] = dbscan.fit_predict(X_scaled)

    for name, labels in results.items():
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f'\n{name}: {n_clusters} clusters, {n_noise} noise points')
        print(f'  Label distribution: {pd.Series(labels[labels != -1]).value_counts().sort_index().to_dict()}')
        mask = labels != -1
        if mask.sum() > n_clusters and n_clusters >= 2:
            print(f'  Silhouette Score: {silhouette_score(X_scaled[mask], labels[mask]):.4f}')
            print(f'  Calinski-Harabasz: {calinski_harabasz_score(X_scaled[mask], labels[mask]):.2f}')
            print(f'  Davies-Bouldin: {davies_bouldin_score(X_scaled[mask], labels[mask]):.4f}')

    return results


def PlotClusteringComparison(results, X_2d, method_name='PCA'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, labels) in enumerate(results.items()):
        scatter = axes[idx].scatter(X_2d[:, 0], X_2d[:, 1],
                                    c=labels, cmap='tab10',
                                    edgecolors='black', s=40, alpha=0.8)
        axes[idx].set_xlabel(f'{method_name} 1')
        axes[idx].set_ylabel(f'{method_name} 2')
        axes[idx].set_title(f'{name} Clustering Result ({method_name})')
        plt.colorbar(scatter, ax=axes[idx])

    fig.suptitle(f'Clustering Comparison ({method_name} 2D)', fontsize=14)
    plt.tight_layout()
    plt.show()


def PlotClusterCenters(kmeans_model, scaler, features, k):
    centers_original = scaler.inverse_transform(kmeans_model.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=features)
    centers_df.index.name = 'Cluster'
    print('K-Means Cluster Centers (original scale):')
    display(centers_df.round(2))


def PlotRadarChart(kmeans_model, features, k):
    N = len(features)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    centers_std = kmeans_model.cluster_centers_
    for i in range(k):
        values = centers_std[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=9)
    ax.set_title('K-Means Cluster Centers Radar Chart (Standardized)', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


def PlotFeatureDistributionByCluster(df, k, features, plot_titles):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (feat, title) in enumerate(zip(features, plot_titles)):
        ax = axes[idx // 2, idx % 2]
        for cluster in range(k):
            data = df[df['KMeans_Cluster'] == cluster][feat]
            ax.hist(data, bins=15, alpha=0.5, label=f'Cluster {cluster}')
        ax.set_title(title)
        ax.legend()
        ax.set_ylabel('Count')

    fig.suptitle('K-Means: Feature Distribution by Cluster', fontsize=14)
    plt.tight_layout()
    plt.show()


def PlotKComparison(X_scaled, X_pca, K_values=[2, 3, 4, 5, 6, 7]):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, k in enumerate(K_values):
        ax = axes[idx // 3, idx % 3]
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        ax.scatter(X_pca[:, 0], X_pca[:, 1],
                   c=labels, cmap='tab10',
                   edgecolors='black', s=30, alpha=0.8)
        sil = silhouette_score(X_scaled, labels)
        ax.set_title(f'K={k}, Silhouette={sil:.3f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    fig.suptitle('K-Means Clustering Results for Different K', fontsize=14)
    plt.tight_layout()
    plt.show()
