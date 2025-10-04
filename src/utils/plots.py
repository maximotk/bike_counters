# utils/plots.py
import plotly.graph_objects as go

def make_plotly_cumulative_animation(x, y, title="Evolution", y_label="RMSE"):
    """
    Build a Plotly figure with Play/Pause controls and slider
    to animate cumulative RMSE evolution.
    """
    frames = []
    for i in range(0, len(x), 4):  # step by 4 for smoother animation
        frames.append(go.Frame(
            data=[go.Scatter(x=x[: i + 1], y=y[: i + 1], mode="lines+markers")],
            name=str(i)
        ))

    # initial trace
    fig = go.Figure(
        data=[go.Scatter(x=[x.iloc[0]], y=[y.iloc[0]], mode="lines+markers")],
        frames=frames
    )

    fig.update_layout(
        title=title,
        xaxis_title="Datetime",
        yaxis_title=y_label,
        xaxis=dict(range=[x.min(), x.max()]),
        yaxis=dict(range=[0, y.max()]),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "y": 1.05,
            "x": 1.02,
            "xanchor": "right",
            "yanchor": "top",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 1, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                },
            ],
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate"}],
                    "label": i,
                    "method": "animate",
                } for i, f in enumerate(frames)
            ],
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "len": 0.8
        }],
        margin=dict(l=40, r=20, t=60, b=40),
        height=420
    )
    return fig
