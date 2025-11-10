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

    # Gaussian Blur for denoising
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel Filter (edge gradients)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # Laplacian (detects fine surface defects)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological enhancement (Top Hat)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Adaptive Thresholding for dark/bright contrast
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # CLAHE contrast enhancement
    clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe_obj.apply(gray)

    # Otsu thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Gabor filter for texture (use valid Python ddepth and convert to displayable format)
    gabor_kernel = cv2.getGaborKernel((21, 21), 4.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    # use cv2.CV_8U (or -1) for ddepth; convertScaleAbs to get uint8 for display
    gabor = cv2.filter2D(gray, ddepth=cv2.CV_8U, kernel=gabor_kernel)
    gabor = cv2.convertScaleAbs(gabor)

    return blurred, sobel_combined, laplacian, edges, morph, adaptive_thresh, clahe_img, otsu, gabor

# ------------------------- Color & Texture Analysis -------------------------
def analyze_color_texture(image):
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_hue = np.mean(h)
    mean_sat = np.mean(s)
    mean_val = np.mean(v)

    # Convert to LAB (perceptual lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_lightness = np.mean(l)

    # Laplacian variance (measures roughness or blur)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Local variance (simple texture proxy)
    # compute variance in small patches (e.g., 9x9) as a texture measure
    ksize = 9
    mean_local = cv2.blur(gray.astype(np.float32), (ksize, ksize))
    sq_mean_local = cv2.blur((gray.astype(np.float32) ** 2), (ksize, ksize))
    local_var_map = sq_mean_local - (mean_local ** 2)
    local_var = np.mean(local_var_map)

    return mean_hue, mean_sat, mean_lightness, lap_var, local_var

# ------------------------- Spoilage Classification -------------------------
def classify_spoilage(mean_hue, mean_sat, mean_lightness, lap_var, local_var):
    """
    Simple heuristic model:
    - Low hue/saturation = discoloration
    - Low lightness = dark, oxidized
    - High Laplacian variance or local variance = rough texture (wrinkles/spots)
    """
    color_score = (mean_sat / 255) * 100
    # normalize texture score with a safe scale factor and clip
    texture_score = np.clip((lap_var / 100) + (local_var / 50), 0, 100)

    # Weighted spoilage score (lower is fresher)
    spoilage_score = (100 - color_score) * 0.55 + texture_score * 0.45

    if spoilage_score < 30:
        label = "🍏 Fresh"
        desc = "✅ Bright color, smooth texture — fruit appears fresh."
    elif 30 <= spoilage_score < 60:
        label = "🍊 Slightly Spoiled"
        desc = "🟠 Some color dullness or minor rough texture detected."
    else:
        label = "🍂 Heavily Spoiled"
        desc = "⚠️ Dull color and uneven texture — high spoilage likelihood."

    return label, desc, spoilage_score

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
st.set_page_config(page_title="Smart Fruit Spoilage Detector", layout="wide")
st.title("🍎 Smart Fruit Spoilage Detection System")

uploaded_file = st.file_uploader("Upload a fruit image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Could not load the image. Please upload a valid file.")
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)

        # Apply filters
        blurred, sobel, laplacian, edges, morph, adaptive_thresh, clahe_img, otsu, gabor = spatial_filtering(image)

        # Filter Visualizations
        st.markdown("### 🔍 Spatial Filter Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(blurred, caption="Gaussian Blur", use_column_width=True)
            st.image(sobel, caption="Sobel Edge Detection", use_column_width=True)
            st.image(clahe_img, caption="CLAHE (contrast)", use_column_width=True)
        with col2:
            st.image(laplacian, caption="Laplacian Edge Map", use_column_width=True)
            st.image(edges, caption="Canny Edges", use_column_width=True)
            st.image(otsu, caption="Otsu Threshold", use_column_width=True)
        with col3:
            st.image(morph, caption="Morphological Top Hat", use_column_width=True)
            st.image(adaptive_thresh, caption="Adaptive Threshold", use_column_width=True)
            st.image(gabor, caption="Gabor Texture Response", use_column_width=True)

        # Analysis
        st.markdown("### 🧠 Color and Texture Analysis")
        mean_hue, mean_sat, mean_lightness, lap_var, local_var = analyze_color_texture(image)
        label, desc, spoilage_score = classify_spoilage(mean_hue, mean_sat, mean_lightness, lap_var, local_var)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Mean Hue", f"{mean_hue:.2f}")
        col_b.metric("Mean Saturation", f"{mean_sat:.2f}")
        col_c.metric("Mean Lightness (LAB)", f"{mean_lightness:.2f}")
        col_d.metric("Texture Variance (Laplacian)", f"{lap_var:.2f}")

        st.markdown(f"### 🍇 Spoilage Classification: **{label}**")
        st.info(desc)
        st.progress(int(np.clip(spoilage_score, 0, 100)))

        # Histogram
        st.markdown("### 📊 Color Histogram")
        hist_fig = plot_histogram(image)
        st.pyplot(hist_fig)

else:
    st.info("Please upload an image file (png, jpg, jpeg, bmp).")
