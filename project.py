import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

st.set_page_config(layout="wide")
st.title("🍎 Advanced Fruit Spoilage Detection & Color Analysis")

# ----------------------
# Calculate grayscale histogram
# ----------------------
def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

# ----------------------
# Detect spoilage area with adjustable HSV range
# ----------------------
def detect_spoilage_area(image, lower_brown, upper_brown):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return mask, dist

# ----------------------
# Apply sharpening with adjustable intensity
# ----------------------
def apply_sharpening(image, intensity=1):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    kernel = kernel * intensity
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# ----------------------
# Apply smoothing filters
# ----------------------
def apply_smoothing(image, method="Gaussian", ksize=3):
    if method == "Gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == "Median":
        return cv2.medianBlur(image, ksize)
    elif method == "Bilateral":
        return cv2.bilateralFilter(image, d=ksize, sigmaColor=75, sigmaSpace=75)
    else:
        return image

# ----------------------
# Histogram equalization
# ----------------------
def equalize_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    eq = cv2.equalizeHist(gray)
    eq_color = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    return eq_color

# ----------------------
# Edge detection (Canny)
# ----------------------
def edge_detection(image, low_threshold=100, high_threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

# ----------------------
# Morphological operations
# ----------------------
def morph_operation(mask, op, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if op == "Erosion":
        return cv2.erode(mask, kernel, iterations=1)
    elif op == "Dilation":
        return cv2.dilate(mask, kernel, iterations=1)
    elif op == "Opening":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif op == "Closing":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        return mask

# ----------------------
# Spot detection from mask (connected components)
# ----------------------
def detect_spots(mask):
    # Connected components on binary mask
    num_labels, labels_im = cv2.connectedComponents(mask)
    # Subtract 1 because background is counted as label 0
    spot_count = num_labels - 1
    return spot_count, labels_im

# ----------------------
# Color analysis: dominant color and stats
# ----------------------
def analyze_colors(image):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    dominant_hue = int(np.median(h))
    mean_saturation = int(np.mean(s))
    mean_value = int(np.mean(v))
    return dominant_hue, mean_saturation, mean_value

# ----------------------
# Show histogram comparison
# ----------------------
def show_histogram_comparison(images, filenames):
    st.subheader("📊 Histogram Comparison Over Days")
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = calculate_histogram(gray)
        ax.plot(hist, label=filenames[i])
    ax.set_title("Histogram Comparison")
    ax.set_xlim([0, 256])
    ax.legend()
    st.pyplot(fig)

# ----------------------
# Main app
# ----------------------
def main():
    uploaded_files = st.file_uploader(
        "Upload daily fruit images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    st.sidebar.header("Filter & Detection Settings")

    # Sharpening settings
    sharpening_intensity = st.sidebar.slider("Sharpening Intensity", 0, 5, 1, 1)

    # Smoothing filter options
    smoothing_method = st.sidebar.selectbox(
        "Smoothing Filter Method", 
        ["None", "Gaussian", "Median", "Bilateral"]
    )
    smoothing_ksize = st.sidebar.slider("Smoothing Kernel Size (odd)", 3, 15, 3, 2)
    smoothing_ksize = smoothing_ksize if smoothing_ksize % 2 == 1 else smoothing_ksize + 1

    # Histogram Equalization
    do_hist_eq = st.sidebar.checkbox("Apply Histogram Equalization")

    # Edge detection thresholds
    st.sidebar.markdown("### Edge Detection (Canny)")
    edge_low = st.sidebar.slider("Low Threshold", 50, 200, 100)
    edge_high = st.sidebar.slider("High Threshold", 100, 300, 200)

    # Morphological operations
    st.sidebar.markdown("### Morphological Operation on Spoilage Mask")
    morph_op = st.sidebar.selectbox("Operation", ["None", "Erosion", "Dilation", "Opening", "Closing"])
    morph_kernel = st.sidebar.slider("Morph Kernel Size", 3, 15, 3, 2)
    morph_kernel = morph_kernel if morph_kernel % 2 == 1 else morph_kernel + 1

    # Spoilage HSV range tuning
    st.sidebar.markdown("### Spoilage HSV Range (Brown Color)")
    h_lower = st.sidebar.slider("Hue Lower", 0, 179, 10)
    s_lower = st.sidebar.slider("Saturation Lower", 0, 255, 50)
    v_lower = st.sidebar.slider("Value Lower", 0, 255, 50)
    h_upper = st.sidebar.slider("Hue Upper", 0, 179, 30)
    s_upper = st.sidebar.slider("Saturation Upper", 0, 255, 255)
    v_upper = st.sidebar.slider("Value Upper", 0, 255, 200)

    lower_brown = np.array([h_lower, s_lower, v_lower])
    upper_brown = np.array([h_upper, s_upper, v_upper])

    if uploaded_files:
        images = []
        filenames = []

        for file in uploaded_files:
            image = Image.open(file).convert("RGB").resize((512, 512))
            img_np = np.array(image)

            # Sharpening
            if sharpening_intensity > 0:
                img_np = apply_sharpening(img_np, intensity=sharpening_intensity)

            # Smoothing
            if smoothing_method != "None":
                img_np = apply_smoothing(img_np, smoothing_method, smoothing_ksize)

            # Histogram Equalization
            if do_hist_eq:
                img_np = equalize_histogram(img_np)

            images.append(img_np)
            filenames.append(file.name)

        show_histogram_comparison(images, filenames)

        st.subheader("🧪 Spoilage & Color Analysis Results")

        for i, img in enumerate(images):
            # Spoilage detection
            mask, dist = detect_spoilage_area(img, lower_brown, upper_brown)

            # Morphological operation
            if morph_op != "None":
                mask = morph_operation(mask, morph_op, morph_kernel)

            # Edge detection
            edges = edge_detection(img, edge_low, edge_high)

            # Spot detection
            spot_count, labels_im = detect_spots(mask)

            # Color analysis
            dominant_hue, mean_saturation, mean_value = analyze_colors(img)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img, caption="Processed Image", use_column_width=True)
                st.markdown(f"**Color Analysis:**")
                st.markdown(f"- Dominant Hue: {dominant_hue}")
                st.markdown(f"- Mean Saturation: {mean_saturation}")
                st.markdown(f"- Mean Value (Brightness): {mean_value}")
            with col2:
                st.image(mask, caption="Spoilage Mask (Morph Applied)", use_column_width=True, clamp=True)
                st.markdown(f"**Detected Spoilage Spots:** {spot_count}")
            with col3:
                st.image(dist, caption="Distance Transform", use_column_width=True, clamp=True)
                st.image(edges, caption="Edge Detection (Canny)", use_column_width=True, clamp=True)

            st.markdown(f"**Image:** {filenames[i]}")
            st.markdown("---")
    else:
        st.info("Please upload one or more fruit images to begin.")

if __name__ == "__main__":
    main()
