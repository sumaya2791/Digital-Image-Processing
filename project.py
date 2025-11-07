import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

st.set_page_config(page_title="🍎 Fruit Spoilage Detector", layout="wide")

# Function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Image Processing Functions ---
def apply_filters(image):
    """Apply multiple spatial filters and edge detectors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge Detection Filters
    canny_edges = cv2.Canny(blurred, 50, 150)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Morphological Filters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    morph_blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Adaptive Threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    return {
        "gray": gray,
        "blurred": blurred,
        "canny": canny_edges,
        "sobel": sobel_combined,
        "laplacian": laplacian,
        "tophat": morph_tophat,
        "blackhat": morph_blackhat,
        "adaptive": adaptive_thresh
    }

def plot_histogram(image):
    """Plot color histogram for an image."""
    color = ('b', 'g', 'r')
    plt.figure(figsize=(6, 3))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title("Color Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    return plt

def estimate_spoilage(image):
    """Estimate spoilage based on color intensity and contrast."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    contrast = np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    if avg_brightness < 70 and contrast < 30:
        return "Heavily Spoiled 🍂", "High discoloration and low contrast indicate strong spoilage."
    elif avg_brightness < 120:
        return "Moderately Spoiled 🍊", "Some discoloration and softness detected."
    else:
        return "Fresh 🍏", "Color and texture suggest the fruit is still fresh."

# --- Streamlit UI ---
st.title("🍎 Intelligent Fruit Spoilage Detection System")
st.markdown("Upload a fruit image to visualize filters and detect spoilage level automatically.")

uploaded_file = st.file_uploader("📤 Upload an image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not load the image. Please try again.")
    else:
        # Layout: show original image
        st.subheader("📸 Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Apply filters
        filtered = apply_filters(image)

        # Display filters in two columns
        st.subheader("🎨 Spatial & Edge Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(filtered["blurred"], caption="Gaussian Blurred", use_column_width=True, clamp=True)
            st.image(filtered["sobel"], caption="Sobel Edge Detection", use_column_width=True, clamp=True)
        with col2:
            st.image(filtered["canny"], caption="Canny Edge Detection", use_column_width=True, clamp=True)
            st.image(filtered["laplacian"], caption="Laplacian Filter", use_column_width=True, clamp=True)
        with col3:
            st.image(filtered["tophat"], caption="Top-Hat Transform", use_column_width=True, clamp=True)
            st.image(filtered["blackhat"], caption="Black-Hat Transform", use_column_width=True, clamp=True)
            st.image(filtered["adaptive"], caption="Adaptive Threshold", use_column_width=True, clamp=True)

        # Histogram section
        st.subheader("📊 Color Histogram")
        fig = plot_histogram(image)
        st.pyplot(fig)

        # Spoilage estimation
        label, desc = estimate_spoilage(image)
        st.markdown("---")
        st.subheader("🍋 Spoilage Estimation Result")
        st.markdown(f"### **Status:** {label}")
        st.write(desc)

else:
    st.info("Please upload a valid image file (png, jpg, jpeg, gif, bmp).")

# Footer
st.markdown("---")
st.markdown("<center>✨ Developed using OpenCV & Streamlit | Smart Spoilage Detection System ✨</center>", unsafe_allow_html=True)
