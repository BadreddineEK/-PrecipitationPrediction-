import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Precipitation Prediction",
    page_icon="🌧️",
    layout="wide"
)

# ── Load metrics ─────────────────────────────────────────────────────────────
METRICS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics.json')
with open(METRICS_PATH) as f:
    metrics = json.load(f)

df_models = pd.DataFrame(metrics["models"])
leaderboard_ref = metrics["leaderboard_reference"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌧️ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Project Overview", "📊 Model Comparison", "🖼️ Data Explorer", "📄 Report"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Polytech Lyon — MAM5A (2024)**")
st.sidebar.markdown("EL KHAMLICHI Badreddine & EL KHALFIOUI Nadir")
st.sidebar.markdown("Supervisor: Thierry Clopeau")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":
    st.title("🌧️ Short-Term Precipitation Forecasting")
    st.markdown("#### CNN & Lagrangian CNN — Polytech Lyon MAM5A (2024)")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best MSLE", "0.129", delta="-0.071 vs leaderboard", delta_color="inverse")
    col2.metric("Dataset size", "100,000 files", ".npz format")
    col3.metric("Image resolution", "128×128 px", "100km × 100km")
    col4.metric("Models compared", "4", "CNN 2D/3D + L-CNN x2")

    st.markdown("---")
    st.markdown("### 🎯 Objective")
    st.markdown("""
    Given **4 consecutive satellite images** of rainfall intensity (6 min apart, 128×128 px, covering 100km²),  
    predict the **next 8 precipitation values** at a 2km×2km resolution.

    Evaluated using **Mean Squared Logarithmic Error (MSLE)**.
    """)

    st.markdown("### 🧠 Approaches")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("""**CNN 2D**  
Standard convolutional network — spatial analysis only.  
Input: `(n, 128, 128, 4)`

**CNN 3D**  
Adds temporal dimension explicitly.  
Input: `(n, 4, 128, 128, 1)`""")
    with col_b:
        st.warning("""**L-CNN (Velocity)**  
Lagrangian CNN with matrix-difference velocity channels.  
Input: `(n, 128, 128, 7)` — 4 + 3 velocity channels

**L-CNN (Optical Flow)**  
Lucas-Kanade optical flow in polar coords.  
Input: `(n, 128, 128, 7)` — 4 + magnitude channels""")

    st.markdown("### 🔄 Pipeline")
    st.code("""
[100k .npz files]  →  log(x+1) transform  →  reshape  →  CNN / L-CNN  →  exp(pred)-1  →  MSLE evaluation
    """, language="text")

    st.markdown("### 💡 Key Insight")
    st.success("""
    ~**80% of data values are zero** (no rain). The `log(x+1)` normalization is critical  
    to prevent the model from being overwhelmed by the skewed distribution.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("---")

    # Table
    st.markdown("### Results Table")
    display_df = df_models[['name', 'msle', 'training_size', 'optimizer', 'batch_size', 'epochs', 'notes']].copy()
    display_df.columns = ['Model', 'MSLE ↓', 'Training Files', 'Optimizer', 'Batch Size', 'Epochs', 'Notes']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Bar chart
    st.markdown("### MSLE Score per Model")
    colors = ['#2ecc71' if m == df_models['msle'].min() else '#3498db' if m < leaderboard_ref else '#e74c3c'
              for m in df_models['msle']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_models['short_name'],
        y=df_models['msle'],
        marker_color=colors,
        text=df_models['msle'],
        textposition='outside'
    ))
    fig.add_hline(
        y=leaderboard_ref,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Leaderboard reference ({leaderboard_ref})",
        annotation_position="top right"
    )
    fig.update_layout(
        yaxis_title="MSLE (lower is better)",
        xaxis_title="Model",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(200,200,200,0.2)')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Key Takeaways")
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **CNN 2D (b32e20)** achieves best MSLE of 0.129 — surpasses leaderboard reference.")
        st.info("ℹ️ CNN 3D is competitive (0.138) but requires much more GPU/RAM.")
    with col2:
        st.warning("⚠️ L-CNN Velocity (0.25) and Optical Flow (0.324) are above the leaderboard reference.")
        st.info("ℹ️ Optical flow struggles with large displacements — a known Lucas-Kanade limitation.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Data Explorer":
    st.title("🖼️ Data Explorer")
    st.markdown("---")

    st.markdown("""
    Each `.npz` file contains **4 satellite images** (128×128 px) captured every 6 minutes,
    representing rainfall intensity over a 100km×100km area.
    """)

    st.markdown("### 📂 Upload a .npz file to explore")
    uploaded = st.file_uploader("Upload a .npz sample file", type=["npz"])

    if uploaded is not None:
        data = np.load(uploaded)
        arr = list(data.values())[0]  # (4, 128, 128)
        if arr.ndim == 3 and arr.shape[0] == 4:
            st.success(f"File loaded: shape {arr.shape}")
            st.markdown("### 4 Temporal Frames (t, t+6min, t+12min, t+18min)")
            cols = st.columns(4)
            for i, col in enumerate(cols):
                fig = px.imshow(
                    arr[i],
                    color_continuous_scale='Blues',
                    title=f"t + {i*6} min",
                    labels={'color': 'Rain intensity'}
                )
                fig.update_layout(coloraxis_showscale=(i == 3), height=300)
                col.plotly_chart(fig, use_container_width=True)

            st.markdown("### Statistics")
            flat = arr.flatten()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("% Zero values", f"{(flat == 0).mean() * 100:.1f}%")
            col2.metric("Max intensity", f"{flat.max():.3f}")
            col3.metric("Mean (non-zero)", f"{flat[flat > 0].mean():.4f}" if (flat > 0).any() else "N/A")
            col4.metric("Std", f"{flat.std():.4f}")

            st.markdown("### Log-transformed preview (frame 1)")
            fig2 = px.imshow(
                np.log1p(arr[0]),
                color_continuous_scale='Viridis',
                title="log(frame[0] + 1)"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error(f"Unexpected array shape: {arr.shape}. Expected (4, 128, 128).")
    else:
        st.info("Upload a `.npz` file from the PlumeLabs dataset to visualize its content.")
        st.markdown("**Expected format:** `(4, 128, 128)` — 4 rainfall intensity images")

        # Simulated demo
        st.markdown("### 🔬 Simulated Demo (random data)")
        if st.button("Generate random example"):
            rng = np.random.default_rng(42)
            fake = rng.exponential(scale=0.5, size=(4, 128, 128))
            fake[fake < 0.3] = 0  # simulate sparsity
            cols = st.columns(4)
            for i, col in enumerate(cols):
                fig = px.imshow(
                    fake[i],
                    color_continuous_scale='Blues',
                    title=f"t + {i*6} min (simulated)"
                )
                fig.update_layout(height=280)
                col.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Report":
    st.title("📄 Full Project Report")
    st.markdown("---")
    st.markdown("""
    The full technical report (PDF) is available directly in the repository.
    """)
    st.markdown(
        """[📥 Download Report (PDF)](https://github.com/BadreddineEK/-PrecipitationPrediction-/raw/main/Projet_MAM5A_EL_KHALFIOU_EL_KHAMLICHI.pdf)"""
    )
    st.markdown("---")
    st.markdown("### 📑 Report Structure")
    st.markdown("""
1. **Introduction** — Project objectives, dataset description, methods overview
2. **Theoretical Foundations** — CNN concepts, Lagrangian coordinates, Lucas-Kanade optical flow
3. **Implementation & Experimentation** — Data preprocessing, CNN 2D/3D, L-CNN
4. **Results & Comparison** — MSLE scores, training curves, prediction examples
5. **Conclusion** — Key findings and perspectives
    """)
