import streamlit as st
from PIL import Image
import tempfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Microplastic Detection Dashboard",
    layout="wide"
)

st.title("🔬 Microplastic Detection Research Dashboard")
st.markdown("---")

# --------------------------------------------------
# SAFE MODEL LOADING (Delayed Import)
# --------------------------------------------------
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("runs/detect/train2/weights/best.pt")

model = load_model()

# --------------------------------------------------
# LOGGING FUNCTION
# --------------------------------------------------
def log_detection(image_name, count, conf):
    log_data = {
        "Timestamp": datetime.now(),
        "Image": image_name,
        "Detections": count,
        "Confidence_Threshold": conf
    }

    df = pd.DataFrame([log_data])

    try:
        existing = pd.read_csv("detection_logs.csv")
        df = pd.concat([existing, df], ignore_index=True)
    except:
        pass

    df.to_csv("detection_logs.csv", index=False)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Detection",
    "📊 Metrics",
    "📈 Analytics",
    "📚 Research"
])

# ======================================================
# TAB 1 — IMAGE DETECTION
# ======================================================
with tab1:

    st.subheader("Upload Microscopic Image")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.25, 0.05
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name, conf=confidence_threshold)

        result = results[0]
        annotated = result.plot()
        count = len(result.boxes)

        st.image(annotated, caption="Detection Result", use_column_width=True)
        st.success(f"Detected Microplastics: {count}")

        log_detection(uploaded_file.name, count, confidence_threshold)

        img_bytes = io.BytesIO()
        Image.fromarray(annotated).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        st.download_button(
            "Download Result Image",
            img_bytes,
            "detection_result.png",
            "image/png"
        )

# ======================================================
# TAB 2 — METRICS
# ======================================================
with tab2:

    st.subheader("Model Performance Metrics")

    # Stored validation values
    precision = 0.761
    recall = 0.685
    map50 = 0.743
    map5095 = 0.335

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", precision)
    col2.metric("Recall", recall)
    col3.metric("mAP@50", map50)
    col4.metric("mAP@50-95", map5095)

    st.markdown("### Confusion Matrix")

    if st.button("Generate Confusion Matrix"):

        cm = np.array([
            [2100, 177],
            [120, 0]
        ])

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig)

# ======================================================
# TAB 3 — ANALYTICS
# ======================================================
with tab3:

    st.subheader("Detection Analytics")

    try:
        df = pd.read_csv("detection_logs.csv")

        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.plot(df["Detections"])
        ax.set_ylabel("Detection Count")
        ax.set_title("Detection Trend")
        st.pyplot(fig)

        st.download_button(
            "Download Logs CSV",
            df.to_csv(index=False),
            "detection_logs.csv",
            "text/csv"
        )

    except:
        st.info("No detection logs available yet. Upload an image first.")

# ======================================================
# TAB 4 — RESEARCH
# ======================================================
with tab4:

    st.subheader("Research Summary")

    st.write("Project: Microplastic Detection in Water Bodies Using AI")
    st.write("Model: YOLOv8 Nano")
    st.write("Epochs: 50")
    st.write("Dataset Size: 751 images")

    st.markdown("""
    ### Performance
    - mAP@50: 0.743  
    - Precision: 0.761  
    - Recall: 0.685  

    ### Contribution
    - Real-time microplastic detection  
    - Interactive research dashboard  
    - Confusion matrix visualization  
    - Detection logging system  
    - Environmental monitoring system  
    """)

st.markdown("---")
st.caption("Deployed on Streamlit Cloud | Final Year AI Project")