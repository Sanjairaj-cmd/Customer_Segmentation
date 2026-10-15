import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Distinct modern color palette for clusters
CLUSTER_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def plot_elbow_curve(k_values, inertia_values, optimal_k):
    """
    Plot the Elbow Method (WCSS vs K) interactive line chart with optimal point highlighted.
    """
    fig = go.Figure()
    
    # WCSS line
    fig.add_trace(go.Scatter(
        x=k_values,
        y=inertia_values,
        mode='lines+markers',
        name='Inertia (WCSS)',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8, color='#1d4ed8')
    ))
    
    # Highlight optimal K
    if optimal_k in k_values:
        idx = k_values.index(optimal_k)
        opt_wcss = inertia_values[idx]
        
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[opt_wcss],
            mode='markers+text',
            name=f'Optimal K = {optimal_k}',
            marker=dict(size=16, color='#ef4444', symbol='star'),
            text=[f"  Elbow Point (K={optimal_k})"],
            textposition="top right",
            textfont=dict(size=13, color='#ef4444')
        ))

    fig.update_layout(
        title="<b>Elbow Method (Within-Cluster Sum of Squares)</b>",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia / WCSS",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig

def plot_silhouette_scores(k_values, silhouette_scores, selected_k):
    """
    Plot Silhouette Scores for different values of K.
    """
    # Filter out K=1 if silhouette score is not available
    valid_ks = [k for k, s in zip(k_values, silhouette_scores) if k > 1]
    valid_scores = [s for k, s in zip(k_values, silhouette_scores) if k > 1]
    
    colors = ['#ef4444' if k == selected_k else '#60a5fa' for k in valid_ks]
    
    fig = go.Figure(data=[
        go.Bar(
            x=valid_ks,
            y=valid_scores,
            marker_color=colors,
            text=[f"{s:.3f}" for s in valid_scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="<b>Silhouette Coefficient vs. Cluster Count</b>",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score (Higher is Better)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_2d_clusters(df, x_col, y_col, cluster_col, centroids_df=None, persona_mapping=None):
    """
    Generate 2D Scatter Plot of Customer Segments with hover info and cluster centroids.
    """
    plot_df = df.copy()
    
    # Map cluster index to persona name if available
    if persona_mapping:
        plot_df['Segment'] = plot_df[cluster_col].apply(lambda c: persona_mapping.get(c, {}).get('name', f"Cluster {c}"))
    else:
        plot_df['Segment'] = plot_df[cluster_col].apply(lambda c: f"Cluster {c}")

    plot_df['Cluster_Str'] = plot_df[cluster_col].astype(str)

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color='Segment',
        hover_data=[col for col in ['CustomerID', 'Gender', 'Age', x_col, y_col] if col in plot_df.columns],
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.85
    )
    
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color='white')))

    # Add Centroids if provided
    if centroids_df is not None and x_col in centroids_df.columns and y_col in centroids_df.columns:
        fig.add_trace(go.Scatter(
            x=centroids_df[x_col],
            y=centroids_df[y_col],
            mode='markers',
            name='Cluster Centroids',
            marker=dict(
                size=18,
                color='black',
                symbol='cross',
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            text=[f"Centroid {i}" for i in range(len(centroids_df))]
        ))

    fig.update_layout(
        title=f"<b>Customer Segments ({x_col} vs {y_col})</b>",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        legend=dict(title="Segments", orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=80)
    )
    return fig

def plot_3d_clusters(df, x_col, y_col, z_col, cluster_col, persona_mapping=None):
    """
    Generate interactive 3D Scatter Plot for 3 features (e.g., Age, Income, Spending Score).
    """
    plot_df = df.copy()
    if persona_mapping:
        plot_df['Segment'] = plot_df[cluster_col].apply(lambda c: persona_mapping.get(c, {}).get('name', f"Cluster {c}"))
    else:
        plot_df['Segment'] = plot_df[cluster_col].apply(lambda c: f"Cluster {c}")

    fig = px.scatter_3d(
        plot_df,
        x=x_col,
        y=y_col,
        z=z_col,
        color='Segment',
        hover_data=[col for col in ['CustomerID', 'Gender'] if col in plot_df.columns],
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.8
    )
    
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        title=f"<b>3D Customer Segments ({x_col}, {y_col}, {z_col})</b>",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1)
    )
    return fig

def plot_cluster_comparison(df, cluster_col, features, persona_mapping=None):
    """
    Plot grouped bar chart comparing average values of selected features across clusters.
    """
    grouped = df.groupby(cluster_col)[features].mean().reset_index()
    
    if persona_mapping:
        grouped['Segment'] = grouped[cluster_col].apply(lambda c: persona_mapping.get(c, {}).get('name', f"Cluster {c}"))
    else:
        grouped['Segment'] = grouped[cluster_col].apply(lambda c: f"Cluster {c}")
        
    melted = pd.melt(grouped, id_vars=['Segment', cluster_col], value_vars=features, var_name='Feature', value_name='Average Value')
    
    fig = px.bar(
        melted,
        x='Segment',
        y='Average Value',
        color='Feature',
        barmode='group',
        text_auto='.1f',
        color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    )
    
    fig.update_layout(
        title="<b>Feature Mean Comparison Across Customer Segments</b>",
        xaxis_title="Customer Segment",
        yaxis_title="Average Metric Value",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=60)
    )
    return fig

def plot_correlation_heatmap(df, numeric_features):
    """
    Plot Pearson correlation matrix heatmap for numeric features.
    """
    corr = df[numeric_features].corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="<b>Feature Correlation Matrix</b>"
    )
    
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_gender_breakdown(df, cluster_col, gender_col, persona_mapping=None):
    """
    Plot stacked/grouped bar chart showing gender distribution per cluster.
    """
    ct = pd.crosstab(df[cluster_col], df[gender_col], normalize='index') * 100
    ct = ct.reset_index()
    
    if persona_mapping:
        ct['Segment'] = ct[cluster_col].apply(lambda c: persona_mapping.get(c, {}).get('name', f"Cluster {c}"))
    else:
        ct['Segment'] = ct[cluster_col].apply(lambda c: f"Cluster {c}")
        
    gender_cols = [c for c in ct.columns if c not in [cluster_col, 'Segment']]
    melted = pd.melt(ct, id_vars=['Segment'], value_vars=gender_cols, var_name='Gender', value_name='Percentage')
    
    fig = px.bar(
        melted,
        x='Segment',
        y='Percentage',
        color='Gender',
        barmode='stack',
        text_auto='.1f',
        color_discrete_map={'Male': '#3b82f6', 'Female': '#ec4899'},
        title="<b>Gender Distribution per Customer Segment (%)</b>"
    )
    
    fig.update_layout(
        template="plotly_white",
        yaxis_title="Percentage (%)",
        xaxis_title="Customer Segment",
        margin=dict(l=40, r=40, t=60, b=60)
    )
    return fig
