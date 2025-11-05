import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utility ----------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Filters ----------
def spatial_filtering(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel Edge Detection (X and Y)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # Laplacian Edge Detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological Top-Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Adaptive Threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    return blurred, sobel_combined, laplacian, edges, morph, adaptive_thresh

# ---------- Frequency Domain (FFT) ----------
def frequency_analysis(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Spoilage detection heuristic:
    # Spoiled fruits often have irregular textures (more high-frequency components)
    high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 95)])
    total_energy = np.sum(magnitude_spectrum)
    spoilage_score = (high_freq_energy / total_energy) * 100

    return magnitude_spectrum, spoilage_score

# ---------- Histogram ----------
def plot_histogram(image):
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
    ax.set_title('Color Histogram')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Fruit Spoilage Detection", layout="wide")
st.title("🍎 Fruit Spoilage Detection using Spatial & Frequency Filtering")

uploaded_file = st.file_uploader("Upload a fruit image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    # Read image safely
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Could not read the image file.")
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)

        # Apply all filters
        blurred, sobel, laplacian, edges, morph, adaptive_thresh = spatial_filtering(image)
        magnitude_spectrum, spoilage_score = frequency_analysis(image)

        # Layout Columns
        st.subheader("Spatial Filtering Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(blurred, caption="Gaussian Blur", use_column_width=True, clamp=True)
            st.image(sobel, caption="Sobel Edges", use_column_width=True, clamp=True)
        with col2:
            st.image(laplacian, caption="Laplacian Edges", use_column_width=True, clamp=True)
            st.image(edges, caption="Canny Edges", use_column_width=True, clamp=True)
        with col3:
            st.image(morph, caption="Morphological Top-Hat", use_column_width=True, clamp=True)
            st.image(adaptive_thresh, caption="Adaptive Threshold", use_column_width=True, clamp=True)
       

        # Spoilage Result
        st.markdown(f"### 🧾 Estimated Spoilage Score: **{spoilage_score:.2f}%**")
        if spoilage_score > 4:
            st.error("⚠️ High-frequency irregularities detected — likely SPOILED fruit.")
        else:
            st.success("✅ Texture is smooth — fruit likely FRESH.")

        # Histogram
        st.subheader("📊 Color Histogram")
        hist_fig = plot_histogram(image)
        st.pyplot(hist_fig)

else:
    st.info("Please upload a valid image file (png, jpg, jpeg, gif, bmp).")
