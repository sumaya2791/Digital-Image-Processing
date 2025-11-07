import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Utility -------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------- Spatial Filters -------------------------
def spatial_filtering(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur (Noise Reduction)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel (Edge Gradients)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # Laplacian (All-Direction Edges)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Canny Edges
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological Top-Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Adaptive Threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    return blurred, sobel_combined, laplacian, edges, morph, adaptive_thresh

# ------------------------- Frequency Analysis -------------------------
def frequency_analysis(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Calculate high frequency content
    high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 95)])
    total_energy = np.sum(magnitude_spectrum)
    freq_score = (high_freq_energy / total_energy) * 100

    return magnitude_spectrum, freq_score

# ------------------------- Color Analysis -------------------------
def color_analysis(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_hue = np.mean(h)
    mean_sat = np.mean(s)
    mean_val = np.mean(v)

    return mean_hue, mean_sat, mean_val

# ------------------------- Spoilage Classification -------------------------
def classify_spoilage(freq_score, mean_hue, mean_sat):
    """
    Simple heuristic:
    - High frequency noise → surface irregularities (spoiled)
    - Low saturation or dull hue → discoloration (spoiled)
    """
    if freq_score > 5 or mean_sat < 70 or mean_hue < 15:
        return "🍂 Heavily Spoiled", "⚠️ High texture irregularity and dull color detected."
    elif 2.5 < freq_score <= 5 or 70 <= mean_sat < 100:
        return "🍊 Slightly Spoiled", "🟠 Minor surface roughness and mild discoloration detected."
    else:
        return "🍏 Fresh", "✅ Smooth surface texture and healthy color detected."

# ------------------------- Histogram Plot -------------------------
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

# ------------------------- Streamlit App -------------------------
st.set_page_config(page_title="Advanced Fruit Spoilage Detector", layout="wide")
st.title("🍎 Advanced Fruit Spoilage Detection System")

uploaded_file = st.file_uploader("Upload a fruit image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Could not load the image. Try uploading a valid file.")
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)

        # Apply analysis
        blurred, sobel, laplacian, edges, morph, adaptive_thresh = spatial_filtering(image)
        magnitude_spectrum, freq_score = frequency_analysis(image)
        mean_hue, mean_sat, mean_val = color_analysis(image)
        label, msg = classify_spoilage(freq_score, mean_hue, mean_sat)

        # Layout - Filters
        st.markdown("### 🔍 Spatial Filtering Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(blurred, caption="Gaussian Blur", use_column_width=True)
            st.image(sobel, caption="Sobel Edges", use_column_width=True)
        with col2:
            st.image(laplacian, caption="Laplacian Edges", use_column_width=True)
            st.image(edges, caption="Canny Edges", use_column_width=True)
        with col3:
            st.image(morph, caption="Morphological Top-Hat", use_column_width=True)
            st.image(adaptive_thresh, caption="Adaptive Threshold", use_column_width=True)

        # Frequency Analysis
        st.markdown("### 🌀 Frequency Domain (Texture) Analysis")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.set_title("Magnitude Spectrum (FFT)")
        ax2.axis('off')
        st.pyplot(fig2)

        # Classification Results
        st.markdown("### 🧠 Spoilage Detection Results")
        st.metric("Frequency Score", f"{freq_score:.2f}%")
        st.metric("Mean Hue", f"{mean_hue:.2f}")
        st.metric("Mean Saturation", f"{mean_sat:.2f}")
        st.markdown(f"### Result: **{label}**")
        st.info(msg)

        # Color Histogram
        st.markdown("### 📊 Color Histogram")
        hist_fig = plot_histogram(image)
        st.pyplot(hist_fig)

else:
    st.info("Please upload a valid image file (png, jpg, jpeg, bmp).")

