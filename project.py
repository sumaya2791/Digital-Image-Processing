import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title="🍎 Smart Fruit Spoilage Detector", layout="wide")

# Sidebar styling
st.sidebar.title("🔧 Control Panel")
st.sidebar.markdown("Upload a fruit image to analyze its freshness and spoilage level.")
theme_color = st.sidebar.color_picker("Accent Color", "#4CAF50")

# ------------------------- Utilities -------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------- Spatial Filters -------------------------
def spatial_filtering(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F))
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return blurred, sobel_combined, laplacian, edges, morph, adaptive_thresh

# ------------------------- Color & Texture Analysis -------------------------
def analyze_color_texture(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    mean_hue, mean_sat, mean_val = np.mean(h), np.mean(s), np.mean(v)
    mean_light, std_light = np.mean(l), np.std(l)
    lap_var = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    return mean_hue, mean_sat, mean_light, std_light, lap_var

# ------------------------- Spoilage Classification -------------------------
def classify_spoilage(mean_sat, mean_light, std_light, lap_var):
    # Weighted scoring
    color_score = (mean_sat / 255) * 100
    brightness_score = (mean_light / 255) * 100
    texture_score = np.clip(lap_var / 80, 0, 100)
    contrast_score = np.clip(std_light / 10, 0, 100)

    spoilage_index = (100 - color_score) * 0.4 + (100 - brightness_score) * 0.2 + texture_score * 0.3 + contrast_score * 0.1
    freshness_percent = 100 - np.clip(spoilage_index, 0, 100)

    if freshness_percent > 70:
        label, emoji, desc = "Fresh", "🍏", "✅ Bright, smooth texture — very fresh fruit."
    elif 40 < freshness_percent <= 70:
        label, emoji, desc = "Slightly Spoiled", "🍊", "🟠 Some color dullness and texture changes detected."
    else:
        label, emoji, desc = "Heavily Spoiled", "🍂", "⚠️ Low color and uneven texture — high spoilage risk."

    return label, emoji, desc, freshness_percent

# ------------------------- Histogram Plot -------------------------
def plot_histogram(image):
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, col in enumerate(('b', 'g', 'r')):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
    ax.set_xlim([0, 256])
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

# ------------------------- PDF Report -------------------------
def generate_pdf_report(label, freshness, desc):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(180, 800, "Fruit Spoilage Analysis Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"Classification: {label}")
    c.drawString(50, 730, f"Freshness Score: {freshness:.2f}%")
    c.drawString(50, 710, f"Description: {desc}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------- MAIN APP -------------------------
st.title("🍎 Smart Fruit Spoilage Detector (Pro Edition)")

uploaded_file = st.file_uploader("Upload your fruit image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Unable to read image. Please try again.")
    else:
        # Original image
        st.markdown("## 📸 Uploaded Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)

        # Apply filters
        blurred, sobel, laplacian, edges, morph, adaptive_thresh = spatial_filtering(image)

        # Filters in tabs
        st.markdown("## 🧭 Filter Visualizations")
        tab1, tab2, tab3 = st.tabs(["Edges", "Morphological", "Threshold"])
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.image(sobel, caption="Sobel Edges", use_column_width=True)
            col2.image(laplacian, caption="Laplacian Edges", use_column_width=True)
            col3.image(edges, caption="Canny Edges", use_column_width=True)
        with tab2:
            st.image(morph, caption="Morphological Top-Hat Enhancement", use_column_width=True)
        with tab3:
            st.image(adaptive_thresh, caption="Adaptive Thresholding", use_column_width=True)

        # Analysis
        st.markdown("## 🧠 Freshness & Spoilage Analysis")
        mean_hue, mean_sat, mean_light, std_light, lap_var = analyze_color_texture(image)
        label, emoji, desc, freshness = classify_spoilage(mean_sat, mean_light, std_light, lap_var)

        # Metrics
        st.markdown(f"### Classification: **{emoji} {label}**")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Saturation", f"{mean_sat:.2f}")
        colB.metric("Lightness", f"{mean_light:.2f}")
        colC.metric("Contrast", f"{std_light:.2f}")
        colD.metric("Texture Variance", f"{lap_var:.2f}")

        # Freshness gauge
        st.markdown("### 🍃 Freshness Level")
        st.progress(int(freshness))
        st.info(desc)

        # Histogram
        st.markdown("## 📊 Color Histogram")
        hist_fig = plot_histogram(image)
        st.pyplot(hist_fig)

        # Download Report
        report_pdf = generate_pdf_report(label, freshness, desc)
        st.download_button(
            label="📄 Download Spoilage Report",
            data=report_pdf,
            file_name="fruit_spoilage_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Upload a fruit image (jpg, jpeg, png, bmp) to start analysis.")
