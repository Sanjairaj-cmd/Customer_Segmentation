import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def calculate_elbow_and_silhouette(scaled_data, max_k=10):
    """
    Calculate Inertia (WCSS) and Silhouette scores for K values from 1 to max_k.
    """
    k_range = range(1, min(max_k + 1, len(scaled_data)))
    inertia_list = []
    silhouette_list = [] # None for k=1
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(scaled_data)
        inertia_list.append(kmeans.inertia_)
        
        if k > 1 and len(np.unique(kmeans.labels_)) > 1:
            score = silhouette_score(scaled_data, kmeans.labels_)
            silhouette_list.append(score)
        else:
            silhouette_list.append(0.0)
            
    return list(k_range), inertia_list, silhouette_list

def find_optimal_k(k_values, inertia_values):
    """
    Automatically determine optimal elbow point using maximum perpendicular distance 
    from the line connecting the first and last points of the inertia curve.
    """
    if len(k_values) <= 2:
        return k_values[0]
        
    x1, y1 = k_values[0], inertia_values[0]
    xn, yn = k_values[-1], inertia_values[-1]
    
    # Line equation coefficients Ax + By + C = 0
    # (y1 - yn)x + (xn - x1)y + (x1*yn - xn*y1) = 0
    A = y1 - yn
    B = xn - x1
    C = x1 * yn - xn * y1
    denom = np.sqrt(A**2 + B**2)
    
    distances = []
    for k, inertia in zip(k_values, inertia_values):
        dist = abs(A * k + B * inertia + C) / (denom + 1e-9)
        distances.append(dist)
        
    optimal_idx = np.argmax(distances)
    return k_values[optimal_idx]

def fit_kmeans(scaled_data, n_clusters):
    """
    Fit K-Means algorithm with specified n_clusters.
    Returns (labels, model, silhouette, centers).
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    
    if n_clusters > 1 and len(np.unique(labels)) > 1:
        sil_score = silhouette_score(scaled_data, labels)
    else:
        sil_score = 0.0
        
    return labels, kmeans, sil_score, kmeans.cluster_centers_

def generate_cluster_personas(df, cluster_col, income_col=None, spending_col=None):
    """
    Generate meaningful persona titles and descriptions for each cluster.
    """
    persona_mapping = {}
    cluster_stats = {}
    
    unique_clusters = sorted(df[cluster_col].unique())
    
    # Check if standard Income & Spending columns exist
    is_standard = income_col in df.columns and spending_col in df.columns
    
    if is_standard:
        income_median = df[income_col].median()
        spending_median = df[spending_col].median()
        
        for c in unique_clusters:
            sub = df[df[cluster_col] == c]
            mean_inc = sub[income_col].mean()
            mean_spn = sub[spending_col].mean()
            
            # Determine high/low relative to median
            high_inc = mean_inc >= income_median
            high_spn = mean_spn >= spending_median
            
            if high_inc and high_spn:
                title = "💎 VIP / Target Spenders"
                desc = "High Annual Income & High Spending Score. Top revenue generator; ideal for premium offers and loyalty rewards."
            elif high_inc and not high_spn:
                title = "🏦 Careful / Conservative Savers"
                desc = "High Annual Income but Low Spending Score. High purchasing power; needs targeted incentives to boost engagement."
            elif not high_inc and high_spn:
                title = "🛍️ Impulsive / High Spenders"
                desc = "Low Annual Income but High Spending Score. Responsive to discounts, flash sales, and trendy items."
            elif not high_inc and not high_spn:
                title = "🛡️ Budget / Sensible Shoppers"
                desc = "Low Annual Income & Low Spending Score. Price-sensitive group; focus on value products and essential deals."
            else:
                title = "⚖️ Standard / Moderate Base"
                desc = "Moderate Income & Moderate Spending Score. Steady customer base; consistent average engagement."
                
            persona_mapping[c] = {
                'name': title,
                'description': desc,
                'mean_income': round(mean_inc, 2),
                'mean_spending': round(mean_spn, 2),
                'count': len(sub),
                'pct': round(len(sub) / len(df) * 100, 1)
            }
    else:
        # Generic feature-based naming
        for c in unique_clusters:
            sub = df[df[cluster_col] == c]
            title = f"Cluster {c}"
            desc = f"Customer Segment {c} containing {len(sub)} customers."
            persona_mapping[c] = {
                'name': title,
                'description': desc,
                'count': len(sub),
                'pct': round(len(sub) / len(df) * 100, 1)
            }
            
    return persona_mapping
