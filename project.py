import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Streamlit Config -------------------------
st.set_page_config(page_title="Smart Fruit Spoilage Detector", layout="wide")
st.title("🍎 Smart Fruit Spoilage Detection System")

# ------------------------- Utility -------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ------------------------- Filters -------------------------
def sobel_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return sobel_combined

def laplacian_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

def canny_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def adaptive_threshold_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return adaptive_thresh

# ------------------------- Segmentation Core (shortened) -------------------------
def compute_spoilage_map(image_bgr, weights=None):
    if weights is None:
        weights = dict(lightness=0.35, saturation=0.2, texture=0.3, brown=0.15)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.abs(lap)
    lap = normalize(lap)

    L_norm = l.astype(np.float32) / 255.0
    S_norm = s.astype(np.float32) / 255.0
    a_signed = (a.astype(np.float32) - 128.0) / 127.0
    b_signed = (b.astype(np.float32) - 128.0) / 127.0
    chroma_pos = np.clip((a_signed + b_signed) / 2.0, 0.0, 1.0)
    brown_score = chroma_pos * (1.0 - L_norm)
    brown_score = normalize(brown_score)

    score_dark = 1.0 - L_norm
    score_desat = 1.0 - S_norm
    score_texture = lap
    score_brown = brown_score

    prob_map = (
        weights['lightness'] * score_dark +
        weights['saturation'] * score_desat +
        weights['texture'] * score_texture +
        weights['brown'] * score_brown
    )
    prob_map = normalize(prob_map)
    return prob_map

# ------------------------- Global Spoilage Classification -------------------------
def classify_spoilage_global(prob_map, mask):
    H, W = prob_map.shape
    area_fraction = float(np.count_nonzero(mask)) / float(H * W)
    if np.count_nonzero(mask) > 0:
        mean_intensity = float(prob_map[mask > 0].mean())
    else:
        mean_intensity = 0.0

    severity = 100.0 * (0.6 * area_fraction + 0.4 * mean_intensity)

    if severity < 15:
        label = "🍏 Fresh"
        desc = "✅ Minimal dark/rough/brown regions detected."
    elif severity < 40:
        label = "🍊 Slightly Spoiled"
        desc = "🟠 Some localized spoilage regions found."
    else:
        label = "🍂 Heavily Spoiled"
        desc = "⚠️ Large or intense spoiled regions detected."

    return label, desc, severity, area_fraction, mean_intensity

# ------------------------- Sidebar -------------------------
st.sidebar.header("⚙️ Detection Controls")
sensitivity = st.sidebar.slider("Detection sensitivity", 0, 100, 55)
overlay_alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.4, 0.05)

st.sidebar.markdown("### 🧩 Filters (separate view)")
show_sobel = st.sidebar.checkbox("Show Sobel Edge Detection", value=False)
show_laplacian = st.sidebar.checkbox("Show Laplacian Filter", value=False)
show_canny = st.sidebar.checkbox("Show Canny Edge Detection", value=False)
show_adaptive = st.sidebar.checkbox("Show Adaptive Thresholding", value=False)

# ------------------------- File Upload -------------------------
uploaded_file = st.file_uploader("Upload a fruit image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("❌ Could not load the image. Please upload a valid file.")
    else:
        image_rgb = to_rgb(image_bgr)
        st.image(image_rgb, caption='Original Image', use_column_width=True)

        # --- Spoilage Map Example (simplified) ---
        prob_map = compute_spoilage_map(image_bgr)
        mask = (prob_map > 0.5).astype(np.uint8) * 255

        st.markdown("### 🍇 Spoilage Classification")
        label, desc, severity, area_fraction, mean_intensity = classify_spoilage_global(prob_map, mask)
        st.info(f"**{label}** — {desc}")
        st.progress(int(np.clip(severity, 0, 100)))

        m1, m2, m3 = st.columns(3)
        m1.metric("Spoilage Severity", f"{severity:.1f}/100")
        m2.metric("Area Fraction", f"{area_fraction*100:.2f}%")
        m3.metric("Mean Intensity", f"{mean_intensity:.2f}")

        # ------------------------- Individual Filter Visualizations -------------------------
        if any([show_sobel, show_laplacian, show_canny, show_adaptive]):
            st.markdown("### 🔍 Individual Filter Visualizations")
            col1, col2 = st.columns(2)

            if show_sobel:
                sobel = sobel_filter(image_bgr)
                col1.image(sobel, caption="Sobel Edge Detection", use_column_width=True, clamp=True)

            if show_laplacian:
                lap = laplacian_filter(image_bgr)
                col2.image(lap, caption="Laplacian Filter (Fine Surface Defects)", use_column_width=True, clamp=True)

            if show_canny:
                edges = canny_filter(image_bgr)
                col1.image(edges, caption="Canny Edge Detection", use_column_width=True, clamp=True)

            if show_adaptive:
                adaptive = adaptive_threshold_filter(image_bgr)
                col2.image(adaptive, caption="Adaptive Thresholding (Dark/Bright Contrast)", use_column_width=True, clamp=True)

else:
    st.info("Please upload an image file (png, jpg, jpeg, bmp).")
