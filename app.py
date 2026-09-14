import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AI 3D House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# LOAD TRAINED ML MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


model = load_model()


# -----------------------------
# CREATE 3D BOX
# -----------------------------

def add_box(fig, x0, x1, y0, y1, z0, z1, color, name):

    vertices = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]

    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]

    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 6, 3, 3]
    j = [1, 2, 4, 2, 5, 3, 5, 6, 6, 7, 0, 7]
    k = [2, 3, 5, 5, 6, 7, 6, 7, 7, 4, 4, 4]

    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=0.9,
            name=name,
            hoverinfo="name"
        )
    )


# -----------------------------
# CREATE 3D HOUSE
# -----------------------------

def create_house_3d(area, bedrooms, floors, garage):

    width = max(10, min(30, np.sqrt(area) * 0.8))
    depth = max(8, min(25, area / width))

    floor_height = 3

    fig = go.Figure()

    # Floors
    for floor in range(floors):

        z0 = floor * floor_height
        z1 = z0 + floor_height

        add_box(
            fig,
            0,
            width,
            0,
            depth,
            z0,
            z1,
            "#D98B5F",
            f"Floor {floor + 1}"
        )

    # Door
    add_box(
        fig,
        width * 0.42,
        width * 0.58,
        -0.3,
        0,
        0,
        2.2,
        "#4A2C20",
        "Main Door"
    )

    # Windows
    window_count = max(2, bedrooms + floors)

    for i in range(window_count):

        x_position = (
            (i + 1) / (window_count + 1)
        ) * width

        add_box(
            fig,
            x_position - 0.6,
            x_position + 0.6,
            -0.3,
            -0.05,
            1,
            2,
            "#55B7E8",
            "Window"
        )

    # Garage
    if garage > 0:

        garage_width = min(width * 0.3, 5)

        add_box(
            fig,
            width - garage_width - 1,
            width - 1,
            -0.35,
            -0.03,
            0,
            2.2,
            "#555555",
            "Garage"
        )

    # Ground
    add_box(
        fig,
        -3,
        width + 3,
        -3,
        depth + 3,
        -0.3,
        0,
        "#6D9E5B",
        "Ground"
    )

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis_title="Width",
            yaxis_title="Depth",
            zaxis_title="Height",
            aspectmode="manual",
            aspectratio=dict(
                x=1.4,
                y=1.1,
                z=0.8
            ),
            camera=dict(
                eye=dict(
                    x=1.7,
                    y=1.7,
                    z=1.4
                )
            )
        ),
        showlegend=False
    )

    return fig


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("🏠 AI 3D House Price Predictor")

st.write(
    "Machine Learning based house price prediction "
    "using a real house sales dataset with interactive 3D visualization."
)

left, right = st.columns([1, 2])


# -----------------------------
# HOUSE INPUTS
# -----------------------------

with left:

    st.subheader("🏡 House Details")

    area = st.slider(
        "House Area (sq ft)",
        500,
        5000,
        1800,
        50
    )

    bedrooms = st.slider(
        "Bedrooms",
        1,
        7,
        3
    )

    bathrooms = st.slider(
        "Bathrooms",
        1,
        5,
        2
    )

    st.info(
        "The Machine Learning model uses "
        "Area, Bedrooms and Bathrooms for prediction."
    )

    # Create DataFrame with correct feature names
    input_data = pd.DataFrame(
        {
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms]
        }
    )

    # Prediction
    predicted_price = model.predict(input_data)[0]

    st.divider()

    st.metric(
        "💰 Estimated House Price",
        f"${predicted_price:,.0f}"
    )


# -----------------------------
# 3D VISUALIZATION
# -----------------------------

with right:

    st.subheader("🏠 Interactive 3D House")

    floors = st.slider(
        "Floors",
        1,
        3,
        2
    )

    garage = st.slider(
        "Garage",
        0,
        3,
        1
    )

    house = create_house_3d(
        area,
        bedrooms,
        floors,
        garage
    )

    st.plotly_chart(
        house,
        use_container_width=True
    )


# -----------------------------
# MACHINE LEARNING INFORMATION
# -----------------------------

st.divider()

st.subheader("📊 Machine Learning Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Algorithm**")
    st.write("Random Forest Regression")

with col2:
    st.write("**Training Dataset**")
    st.write("King County House Sales")

with col3:
    st.write("**Features**")
    st.write("Area, Bedrooms, Bathrooms")


st.caption(
    "The prediction is generated by a Random Forest "
    "Regression model trained on real house-price data."
)