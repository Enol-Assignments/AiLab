import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris
from pandas.plotting import parallel_coordinates


warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


def LoadIris():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df['target'] = iris.target

    print(f'Shape: {df.shape}')
    print(f'Features: {list(iris.feature_names)}')
    print(f'Classes: {list(iris.target_names)}')
    print(f'Samples per class:')
    print(df['species'].value_counts())
    print()
    display(df.head(10))
    return iris, df


def CheckMissing(df):
    print('Missing values per column:')
    print(df.isnull().sum())
    print(f'\nTotal missing: {df.isnull().sum().sum()}')


def SplitAndScale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f'Training set: {X_train.shape[0]} samples')
    print(f'Test set:     {X_test.shape[0]} samples')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f'\nStandardization done.')
    print(f'Train mean: {X_train_scaled.mean(axis=0).round(2)}')
    print(f'Train std:  {X_train_scaled.std(axis=0).round(2)}')
    return X_train_scaled, X_test_scaled, y_train, y_test


def PlotBoxPlots(df, feature_names, target_names):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, feature in enumerate(feature_names):
        ax = axes[idx // 2, idx % 2]
        for species in target_names:
            data = df[df['species'] == species][feature]
            ax.boxplot(data, positions=[list(target_names).index(species)],
                       widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=plt.cm.Set1(list(target_names).index(species) / 3)))
        ax.set_xticks(range(3))
        ax.set_xticklabels(target_names)
        ax.set_title(feature)
        ax.set_ylabel('cm')
    fig.suptitle('Feature Distribution by Species (Box Plot)', fontsize=14)
    plt.tight_layout()
    plt.show()


def PlotPairPlot(df, feature_names):
    sns.pairplot(df, hue='species', vars=feature_names,
                 palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.7})
    plt.suptitle('Iris Pairwise Feature Scatter Matrix', y=1.02, fontsize=14)
    plt.show()


def PlotCorrelationHeatmap(df, feature_names):
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[feature_names].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def PlotParallelCoordinates(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    parallel_coordinates(df.drop('target', axis=1), 'species',
                         color=['#e41a1c', '#377eb8', '#4daf4a'], ax=ax)
    ax.set_title('Iris Parallel Coordinates Plot')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def PlotConfusionMatrices(results, y_test, target_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    for idx, (name, info) in enumerate(results.items()):
        cm = confusion_matrix(y_test, info['y_pred'])
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot(
            ax=axes[idx], cmap='Blues', colorbar=False
        )
        axes[idx].set_title(f'{name}\nAccuracy={info["accuracy"]:.4f}')
    fig.suptitle('Confusion Matrix Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()


def PlotPerformanceComparison(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    f1_scores_list = [results[n]['f1'] for n in names]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#377eb8')
    bars2 = ax.bar(x + width/2, f1_scores_list, width, label='F1 (weighted)', color='#ff7f00')

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.8, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    ax.bar_label(bars1, fmt='%.3f', padding=3)
    ax.bar_label(bars2, fmt='%.3f', padding=3)
    plt.tight_layout()
    plt.show()


def PlotDecisionBoundary(model, X, y, feature_names, title, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1',
               edgecolors='black', s=30, alpha=0.8)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)


def PlotROCCurves(results, X_test_scaled, y_test, target_names):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, info) in enumerate(results.items()):
        ax = axes[idx]
        model = info['model']
        y_score = model.predict_proba(X_test_scaled)

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{target_names[i]} (AUC={roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name} ROC Curve (OvR)')
        ax.legend(loc='lower right', fontsize=8)

    fig.suptitle('ROC Curve Comparison (One-vs-Rest)', fontsize=14)
    plt.tight_layout()
    plt.show()
