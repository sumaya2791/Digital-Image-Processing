import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def overlay_mask_on_rgb(rgb, mask, color=(255, 0, 0), alpha=0.4):
    # mask: uint8 0/255
    overlay = rgb.copy().astype(np.float32)
    m = (mask > 0)[..., None]
    overlay[m] = overlay[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)

def draw_contours_and_boxes(rgb, mask, min_area=300, thickness=2):
    out = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape[:2]
    total_area = H * W
    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        kept.append(c)
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, (255, 0, 0), thickness)  # red contours
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), thickness)  # yellow bbox
        pct = 100.0 * area / total_area
        cv2.putText(out, f"{area:.0f}px | {pct:.2f}%", (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return out, kept

def colorize_heatmap(prob_map):
    # prob_map: float 0..1
    heat = (normalize(prob_map) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return heat_rgb

# ------------------------- Spatial Filters (kept) -------------------------
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

    return blurred, sobel_combined, laplacian, edges, morph, adaptive_thresh

# ------------------------- Color & Texture Analysis (kept) -------------------------
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

    return mean_hue, mean_sat, mean_lightness, lap_var

# ------------------------- Improved Spoilage Segmentation -------------------------
def compute_spoilage_map(image_bgr, weights=None, smooth_ksize=5):
    """
    Returns:
    - prob_map: float32 0..1, per-pixel spoilage probability-like score
    - feature_maps: dict of intermediate maps for debugging
    """
    if weights is None:
        weights = dict(lightness=0.35, saturation=0.2, texture=0.3, brown=0.15)

    # Color spaces
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_raw, a_raw, b_raw = cv2.split(lab)

    # CLAHE on L for illumination robustness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l_raw)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    lap_norm = normalize(lap_abs)

    # Normalize channels
    L_norm = l.astype(np.float32) / 255.0
    S_norm = s.astype(np.float32) / 255.0

    # Brownness heuristic in Lab (positive a and b + low L)
    a_signed = (a_raw.astype(np.float32) - 128.0) / 127.0  # -1..1
    b_signed = (b_raw.astype(np.float32) - 128.0) / 127.0  # -1..1
    chroma_pos = np.clip((a_signed + b_signed) / 2.0, 0.0, 1.0)
    brown_score = chroma_pos * (1.0 - L_norm)
    brown_score = normalize(brown_score)

    # Individual contributions
    score_dark = 1.0 - L_norm            # darker -> more spoiled
    score_desat = 1.0 - S_norm           # lower saturation -> more spoiled
    score_texture = lap_norm             # rough/uneven -> more spoiled
    score_brown = brown_score            # brownness in Lab

    # Weighted fusion
    prob_map = (
        weights['lightness'] * score_dark +
        weights['saturation'] * score_desat +
        weights['texture'] * score_texture +
        weights['brown'] * score_brown
    )
    prob_map = normalize(prob_map)

    # Optional smoothing
    k = max(3, smooth_ksize | 1)  # make odd
    prob_map = cv2.GaussianBlur(prob_map, (k, k), 0)

    feature_maps = {
        "dark": score_dark, "desat": score_desat,
        "texture": score_texture, "brown": score_brown
    }
    return prob_map, feature_maps

def segment_spoilage(image_bgr, sensitivity=50, min_area=300, morph_kernel=5, weights=None):
    """
    Returns:
    - mask: uint8 binary mask (255 = suspected spoiled)
    - prob_map: float32 0..1
    - contours_kept: list of contours kept after filtering
    """
    prob_map, _ = compute_spoilage_map(image_bgr, weights=weights, smooth_ksize=5)

    # Otsu threshold then adjust by sensitivity
    pm8 = (prob_map * 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(pm8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # sensitivity: 0..100 -> lower threshold with higher sensitivity
    delta = (sensitivity - 50) / 50.0  # [-1, 1]
    thr = int(np.clip(otsu_thr * (1.0 - 0.4 * delta), 1, 254))

    _, mask = cv2.threshold(pm8, thr, 255, cv2.THRESH_BINARY)

    # Morphological clean-up
    mk = max(3, morph_kernel | 1)  # ensure odd >=3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    kept_areas = []
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max(1, int(min_area)):
            clean[labels == i] = 255
            kept_areas.append(area)

    # Extract contours for visualization
    contours_kept, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return clean, prob_map, contours_kept

# ------------------------- Histogram Plot (kept) -------------------------
def plot_histogram(image):
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, label=f'{col.upper()}')
        ax.set_xlim([0, 256])
    ax.set_title('Color Histogram')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    return fig

# ------------------------- Spoilage Classification (updated) -------------------------
def classify_spoilage_global(prob_map, mask):
    # Area fraction and mean intensity inside mask
    H, W = prob_map.shape
    area_fraction = float(np.count_nonzero(mask)) / float(H * W)
    if np.count_nonzero(mask) > 0:
        mean_intensity = float(prob_map[mask > 0].mean())
    else:
        mean_intensity = 0.0

    # Severity: combine area + intensity
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

# ------------------------- Streamlit App -------------------------
st.set_page_config(page_title="Smart Fruit Spoilage Detector", layout="wide")
st.title("🍎 Smart Fruit Spoilage Detection System")

# Sidebar controls
st.sidebar.header("⚙️ Detection Controls")
sensitivity = st.sidebar.slider("Detection sensitivity", 0, 100, 55, help="Higher = detect more (lower threshold).")
min_area = st.sidebar.slider("Min region area (px)", 50, 10000, 800, step=50)
morph_kernel = st.sidebar.slider("Morph kernel size", 3, 21, 7, step=2)
overlay_alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.4, 0.05)
show_debug = st.sidebar.checkbox("Show debug feature maps", value=False)
show_filters = st.sidebar.checkbox("Show spatial filter results", value=False)

uploaded_file = st.file_uploader("Upload a fruit image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("❌ Could not load the image. Please upload a valid file.")
    else:
        image_rgb = to_rgb(image_bgr)
        st.image(image_rgb, caption='Original Image', use_column_width=True)

        # Improved segmentation
        mask, prob_map, contours = segment_spoilage(
            image_bgr,
            sensitivity=sensitivity,
            min_area=min_area,
            morph_kernel=morph_kernel,
            weights=dict(lightness=0.35, saturation=0.20, texture=0.30, brown=0.15)
        )

        # Visualizations
        st.markdown("### 🧪 Spoilage Detection & Visualization")
        overlay_rgb = overlay_mask_on_rgb(image_rgb, mask, color=(255, 0, 0), alpha=overlay_alpha)
        boxed_rgb, _ = draw_contours_and_boxes(overlay_rgb, mask, min_area=min_area, thickness=2)
        heat_rgb = colorize_heatmap(prob_map)
        heat_overlay = overlay_mask_on_rgb(image_rgb, (normalize(prob_map) * 255).astype(np.uint8), color=(255, 0, 0), alpha=overlay_alpha)

        col1, col2 = st.columns(2)
        with col1:
            st.image(boxed_rgb, caption="Detected Spoiled Regions (overlay + boxes)", use_column_width=True)
            st.image(mask, caption="Spoilage Mask (binary)", use_column_width=True, clamp=True)
        with col2:
            st.image(heat_rgb, caption="Spoilage Heatmap (higher = more spoiled)", use_column_width=True)
            st.image(heat_overlay, caption="Heatmap Overlay", use_column_width=True)

        # Global classification
        label, desc, severity, area_fraction, mean_intensity = classify_spoilage_global(prob_map, mask)
        st.markdown(f"### 🍇 Spoilage Classification: **{label}**")
        st.info(desc)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Spoilage Severity", f"{severity:.1f}/100")
        m2.metric("Area Fraction", f"{area_fraction*100:.2f}%")
        m3.metric("Mean Spoilage Intensity (in mask)", f"{mean_intensity:.2f}")
        st.progress(int(np.clip(severity, 0, 100)))

        # Optional debug feature maps
        if show_debug:
            _, fmap = compute_spoilage_map(image_bgr, smooth_ksize=5)
            st.markdown("### 🧩 Debug Feature Maps")
            d1, d2, d3, d4 = st.columns(4)
            d1.image((normalize(fmap["dark"]) * 255).astype(np.uint8), caption="Darkness (1-L)", use_column_width=True, clamp=True)
            d2.image((normalize(fmap["desat"]) * 255).astype(np.uint8), caption="Desaturation (1-S)", use_column_width=True, clamp=True)
            d3.image((normalize(fmap["texture"]) * 255).astype(np.uint8), caption="Texture (Laplacian)", use_column_width=True, clamp=True)
            d4.image((normalize(fmap["brown"]) * 255).astype(np.uint8), caption="Brownness (Lab)", use_column_width=True, clamp=True)

        # Keep your original analysis
        st.markdown("### 🧠 Color and Texture Analysis")
        mean_hue, mean_sat, mean_lightness, lap_var = analyze_color_texture(image_bgr)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Mean Hue", f"{mean_hue:.2f}")
        col_b.metric("Mean Saturation", f"{mean_sat:.2f}")
        col_c.metric("Mean Lightness (LAB)", f"{mean_lightness:.2f}")
        col_d.metric("Texture Variance", f"{lap_var:.2f}")

        # Optional: show original spatial filter results
        if show_filters:
            st.markdown("### 🔍 Spatial Filter Results")
            blurred, sobel, laplacian, edges, morph, adaptive_thresh = spatial_filtering(image_bgr)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(blurred, caption="Gaussian Blur", use_column_width=True, clamp=True)
                st.image(sobel, caption="Sobel Edge Detection", use_column_width=True, clamp=True)
            with col2:
                st.image(laplacian, caption="Laplacian Edge Map", use_column_width=True, clamp=True)
                st.image(edges, caption="Canny Edges", use_column_width=True, clamp=True)
            with col3:
                st.image(morph, caption="Morphological Top Hat", use_column_width=True, clamp=True)
                st.image(adaptive_thresh, caption="Adaptive Threshold", use_column_width=True, clamp=True)

        # Histogram
        st.markdown("### 📊 Color Histogram")
        hist_fig = plot_histogram(image_bgr)
        st.pyplot(hist_fig)

else:
    st.info("Please upload an image file (png, jpg, jpeg, bmp).")
