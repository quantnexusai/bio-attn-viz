import plotly.graph_objects as go
import numpy as np

def plot_attention_heatmap(tokens, attention_weights, title="Attention Heatmap"):
    """Create interactive attention heatmap using Plotly"""
    # Filter out padding tokens
    filtered_tokens = []
    filtered_attention = []
    
    for i, token in enumerate(tokens):
        if token not in ["[PAD]", "<pad>"]:
            filtered_tokens.append(token)
            filtered_attention.append(attention_weights[i, :len(filtered_tokens)])
    
    filtered_attention = np.array(filtered_attention)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=filtered_attention,
        x=filtered_tokens,
        y=filtered_tokens,
        colorscale="Viridis",
        hoverongaps=False))
    
    fig.update_layout(
        title=title,
        height=600,
        width=700,
        xaxis_title="Tokens",
        yaxis_title="Tokens",
        xaxis=dict(side="top")
    )
    
    return fig