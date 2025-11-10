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

def overlay_mask_on_rgb(rgb, mask, color=(255, 0, 0), alpha=0.4):
    """
    Safely overlays a semi-transparent color on the RGB image wherever mask > 0.
    Works with 2D or 3D masks and auto-resizes the mask if needed.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 array")

    # Ensure mask is 2D and matches image size
    if mask.ndim == 3:
        mask2d = mask[..., 0]
    else:
        mask2d = mask

    if mask2d.shape[:2] != rgb.shape[:2]:
        mask2d = cv2.resize(mask2d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask2d > 0

    overlay = rgb.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    blended = overlay * (1 - alpha) + color_arr * alpha

    # Apply blended color only where mask is True
    result = np.where(mask_bool[..., None], blended, overlay)
    return np.clip(result, 0, 255).astype(np.uint8)

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

    return blurred, sobel_combined, laplacian, edges, morph, adaptive_thresh

# ------------------------- Color & Texture Analysis -------------------------
def analyze_color_texture(image):
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_hue = np.mean(h)
    mean_sat = np.mean(s)

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
    # default weights (sum not required; final map is normalized)
    if weights is None:
        weights = dict(lightness=0.35, saturation=0.20, texture=0.30, brown=0.15)

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

    # Laplacian for texture
    lap = cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    lap_norm = normalize(lap_abs)

    # Sobel magnitude for gradients / scratches
    sobelx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_norm = normalize(sobel_mag)

    # Canny edges as another indicator of surface defects (binary -> smooth)
    canny = cv2.Canny(gray_blur, 50, 150)
    # Smooth and normalize canny so it contributes as a soft feature
    canny_blur = cv2.GaussianBlur(canny.astype(np.float32), (7, 7), 0)
    canny_norm = normalize(canny_blur)

    # Combine texture-related signals into a single texture score
    # weighted fusion inside texture (weights chosen to prefer Laplacian a bit)
    texture_comb = 0.55 * lap_norm + 0.30 * sobel_norm + 0.15 * canny_norm
    texture_norm = normalize(texture_comb)

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
    score_texture = texture_norm         # fused texture (lap + sobel + canny)
    score_brown = brown_score            # brownness in Lab

    # Weighted fusion (final map normalized afterwards)
    prob_map = (
        weights.get('lightness', 0.35) * score_dark +
        weights.get('saturation', 0.20) * score_desat +
        weights.get('texture', 0.30) * score_texture +
        weights.get('brown', 0.15) * score_brown
    )
    prob_map = normalize(prob_map)

    # Optional smoothing (ensure odd kernel)
    k = max(3, int(smooth_ksize) | 1)
    prob_map = cv2.GaussianBlur(prob_map, (k, k), 0)

    feature_maps = {
        "dark": score_dark, "desat": score_desat,
        "texture": score_texture, "brown": score_brown,
        "laplacian": lap_norm, "sobel": sobel_norm, "canny": canny_norm
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
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max(1, int(min_area)):
            clean[labels == i] = 255

    # Extract contours for visualization
    contours_kept, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return clean, prob_map, contours_kept

# ------------------------- Histogram Plot -------------------------
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

# ------------------------- Global Spoilage Classification -------------------------
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

# ------------------------- Sidebar Controls -------------------------
st.sidebar.header("⚙️ Detection Controls")
sensitivity = st.sidebar.slider("Detection sensitivity", 0, 100, 55, help="Higher = detect more (lower threshold).")
min_area = st.sidebar.slider("Min region area (px)", 50, 10000, 800, step=50)
morph_kernel = st.sidebar.slider("Morph kernel size", 3, 21, 7, step=2)
overlay_alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.4, 0.05)
show_debug = st.sidebar.checkbox("Show debug feature maps", value=False)
show_filters = st.sidebar.checkbox("Show spatial filter results", value=False)
show_edges_overlay = st.sidebar.checkbox("Overlay Canny edges on image", value=False)

# Allow user to tweak fusion weights (optional small UI)
st.sidebar.markdown("**Fusion weights (optional)**")
w_light = st.sidebar.slider("Lightness weight", 0.0, 1.0, 0.35, 0.05)
w_sat = st.sidebar.slider("Saturation weight", 0.0, 1.0, 0.20, 0.05)
w_tex = st.sidebar.slider("Texture weight", 0.0, 1.0, 0.30, 0.05)
w_brown = st.sidebar.slider("Brownness weight", 0.0, 1.0, 0.15, 0.05)

# ------------------------- File Uploader -------------------------
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
        weights = dict(lightness=w_light, saturation=w_sat, texture=w_tex, brown=w_brown)
        mask, prob_map, contours = segment_spoilage(
            image_bgr,
            sensitivity=sensitivity,
            min_area=min_area,
            morph_kernel=morph_kernel,
            weights=weights
        )

        # Also compute spatial filters for optional display / overlay
        blurred, sobel, laplacian, edges, morph, adaptive_thresh = spatial_filtering(image_bgr)

        # Visualizations
        st.markdown("### 🧪 Spoilage Detection & Visualization")
        overlay_rgb = overlay_mask_on_rgb(image_rgb, mask, color=(255, 0, 0), alpha=overlay_alpha)

        # Optionally overlay canny edges (thin cyan) for visual confirmation
        if show_edges_overlay:
            edges_color = np.zeros_like(image_rgb)
            edges_color[edges > 0] = (0, 255, 255)  # cyan-ish in RGB
            overlay_with_edges = cv2.addWeighted(overlay_rgb, 1.0, edges_color, 0.5, 0)
            display_rgb = overlay_with_edges
        else:
            display_rgb = overlay_rgb

        boxed_rgb, _ = draw_contours_and_boxes(display_rgb, mask, min_area=min_area, thickness=2)
        heat_rgb = colorize_heatmap(prob_map)
        # Blend full heatmap over the image for better visualization
        heat_overlay = cv2.addWeighted(image_rgb, 1 - overlay_alpha, heat_rgb, overlay_alpha, 0)

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
            _, fmap = compute_spoilage_map(image_bgr, weights=weights, smooth_ksize=5)
            st.markdown("### 🧩 Debug Feature Maps")
            d1, d2, d3, d4 = st.columns(4)
            d1.image((normalize(fmap["dark"]) * 255).astype(np.uint8), caption="Darkness (1-L)", use_column_width=True, clamp=True)
            d2.image((normalize(fmap["desat"]) * 255).astype(np.uint8), caption="Desaturation (1-S)", use_column_width=True, clamp=True)
            d3.image((normalize(fmap["texture"]) * 255).astype(np.uint8), caption="Texture (Laplacian+Sobel+Canny)", use_column_width=True, clamp=True)
            d4.image((normalize(fmap["brown"]) * 255).astype(np.uint8), caption="Brownness (Lab)", use_column_width=True, clamp=True)
            # extra maps
            e1, e2, e3 = st.columns(3)
            e1.image((normalize(fmap["laplacian"]) * 255).astype(np.uint8), caption="Laplacian", use_column_width=True, clamp=True)
            e2.image((normalize(fmap["sobel"]) * 255).astype(np.uint8), caption="Sobel Magnitude", use_column_width=True, clamp=True)
            e3.image((normalize(fmap["canny"]) * 255).astype(np.uint8), caption="Canny (soft)", use_column_width=True, clamp=True)

        # Keep original analysis
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(blurred, caption="Gaussian Blur", use_column_width=True, clamp=True)
                st.image(sobel, caption="Sobel Edge Detection (abs)", use_column_width=True, clamp=True)
            with col2:
                st.image(laplacian, caption="Laplacian Edge Map", use_column_width=True, clamp=True)
                st.image(edges, caption="Canny Edges (binary)", use_column_width=True, clamp=True)
            with col3:
                st.image(morph, caption="Morphological Top Hat", use_column_width=True, clamp=True)
                st.image(adaptive_thresh, caption="Adaptive Threshold", use_column_width=True, clamp=True)

        # Histogram
        st.markdown("### 📊 Color Histogram")
        hist_fig = plot_histogram(image_bgr)
        st.pyplot(hist_fig)

else:
    st.info("Please upload an image file (png, jpg, jpeg, bmp).")


