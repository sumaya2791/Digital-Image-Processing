import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to check allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Spatial filtering function
def spatial_filtering(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return blurred, edges, morph, adaptive_thresh

# Histogram plot function
def plot_histogram(image):
    color = ('b', 'g', 'r')
    plt.figure(figsize=(5, 3))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return plt

# Streamlit UI
st.title("🍎 Fruit Spoilage - Multi Image Filter & Comparison Tool")

uploaded_files = st.file_uploader("Upload multiple images", type=list(ALLOWED_EXTENSIONS), accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if allowed_file(uploaded_file.name):
            st.markdown(f"### 📸 {uploaded_file.name}")

            # Read image as OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Apply filters
            blurred, edges, morph, adaptive_thresh = spatial_filtering(image)

            # Display images in a grid layout
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
                st.image(blurred, caption="Blurred", use_column_width=True, clamp=True)
                st.image(morph, caption="Morphological Top Hat", use_column_width=True, clamp=True)
            with col2:
                st.image(edges, caption="Canny Edge Detection", use_column_width=True, clamp=True)
                st.image(adaptive_thresh, caption="Adaptive Threshold", use_column_width=True, clamp=True)

            # Show histogram
            st.subheader("Color Histogram")
            fig = plot_histogram(image)
            st.pyplot(fig)

    # --- Comparison Section ---
    if len(uploaded_files) > 1:
        st.markdown("## 🧾 Comparison Between Uploaded Images")

        # Display histograms of all images together for comparison
        plt.figure(figsize=(7, 4))
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.getbuffer()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(hist, label=uploaded_file.name.split('.')[0])
        plt.title("Histogram Comparison (Blue Channel)")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
else:
    st.info("Please upload one or more image files of type: " + ", ".join(ALLOWED_EXTENSIONS))

