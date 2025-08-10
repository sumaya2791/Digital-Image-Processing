import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy import ndimage

# --- Utils ---
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pil_to_cv2(image):
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    """Convert OpenCV BGR image to PIL RGB Image"""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def plot_histogram(image):
    """Plot and return histogram as a PIL image"""
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    plt.figure(figsize=(6,4))
    plt.plot(hist_b, color='blue', label='Blue', alpha=0.7)
    plt.plot(hist_g, color='green', label='Green', alpha=0.7)
    plt.plot(hist_r, color='red', label='Red', alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Color Histogram Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# --- Core Detector Class ---
class FruitSpoilageDetector:
    def __init__(self, image):
        """
        image: OpenCV BGR image
        """
        self.image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def color_analysis(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        healthy_ranges = [
            ([35, 40, 40], [85, 255, 255]),    # Green
            ([15, 40, 40], [35, 255, 255]),    # Yellow/Orange
            ([0, 40, 40], [15, 255, 255]),     # Red
            ([165, 40, 40], [180, 255, 255])   # Red upper range
        ]
        
        spoilage_ranges = [
            ([10, 100, 20], [30, 255, 180]),   # Brown/rot wider
            ([0, 0, 0], [180, 255, 50])        # Dark/mold
        ]
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, fruit_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        healthy_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        spoilage_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in healthy_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            healthy_mask = cv2.bitwise_or(healthy_mask, mask)
        for lower, upper in spoilage_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            spoilage_mask = cv2.bitwise_or(spoilage_mask, mask)
        
        healthy_mask = cv2.bitwise_and(healthy_mask, healthy_mask, mask=fruit_mask)
        spoilage_mask = cv2.bitwise_and(spoilage_mask, spoilage_mask, mask=fruit_mask)
        
        total_fruit_pixels = np.sum(fruit_mask > 0)
        if total_fruit_pixels == 0:
            total_fruit_pixels = 1
        
        healthy_pct = (np.sum(healthy_mask > 0) / total_fruit_pixels) * 100
        spoilage_pct = (np.sum(spoilage_mask > 0) / total_fruit_pixels) * 100
        
        color_analysis_img = self.image_rgb.copy()
        color_analysis_img[spoilage_mask > 0] = [255, 0, 0]  # Mark spoilage in red
        
        return healthy_pct, spoilage_pct, color_analysis_img, spoilage_mask

    def spatial_filtering(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
        return blurred, edges, morph, adaptive_thresh

    def histogram_analysis(self):
        mean_intensity = np.mean(self.image)
        std_intensity = np.std(self.image)
        hist_img = plot_histogram(self.image)
        return mean_intensity, std_intensity, hist_img

    def distance_transform(self, spoilage_mask):
        dist_transform = cv2.distanceTransform(spoilage_mask, cv2.DIST_L2, 5)
        dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dist_colored = cv2.applyColorMap(dist_normalized, cv2.COLORMAP_JET)
        
        max_distance = np.max(dist_transform)
        mean_distance = np.mean(dist_transform[dist_transform > 0]) if np.sum(dist_transform > 0) > 0 else 0
        
        return dist_colored, max_distance, mean_distance

    def detect_spoilage_spots(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7,7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 50
        spots = [c for c in contours if cv2.contourArea(c) > min_area]
        
        spot_image = self.image_rgb.copy()
        cv2.drawContours(spot_image, spots, -1, (255, 0, 0), 2)
        
        total_spoilage_area = sum([cv2.contourArea(c) for c in spots])
        total_image_area = self.image.shape[0] * self.image.shape[1]
        spoilage_area_pct = (total_spoilage_area / total_image_area) * 100
        
        return len(spots), spoilage_area_pct, spot_image

    def process_all(self):
        healthy_pct, spoilage_pct, color_img, spoilage_mask = self.color_analysis()
        blurred, edges, morph, adaptive_thresh = self.spatial_filtering()
        mean_intensity, std_intensity, hist_img = self.histogram_analysis()
        dist_img, max_dist, mean_dist = self.distance_transform(spoilage_mask)
        num_spots, spoilage_area_pct, spot_img = self.detect_spoilage_spots()

        spoilage_score = (
            spoilage_pct * 0.5 +
            spoilage_area_pct * 0.35 +
            min(max_dist * 2, 15)
        )
        spoilage_score = min(spoilage_score, 100)

        if spoilage_score < 5:
            freshness_level = "Fresh"
            freshness_color = "#4CAF50"
        elif spoilage_score < 20:
            freshness_level = "Slightly Aged"
            freshness_color = "#FFC107"
        elif spoilage_score < 40:
            freshness_level = "Moderately Spoiled"
            freshness_color = "#FF9800"
        else:
            freshness_level = "Heavily Spoiled"
            freshness_color = "#F44336"

        return {
            "spoilage_score": round(spoilage_score, 2),
            "freshness_level": freshness_level,
            "freshness_color": freshness_color,
            "healthy_percentage": round(healthy_pct, 2),
            "spoilage_percentage": round(spoilage_pct, 2),
            "num_spots": num_spots,
            "spoilage_area_percentage": round(spoilage_area_pct, 2),
            "max_distance": round(max_dist, 2),
            "mean_intensity": round(mean_intensity, 2),
            "color_analysis_image": color_img,
            "edges_image": edges,
            "morph_image": morph,
            "distance_transform_image": dist_img,
            "spots_detected_image": spot_img,
            "histogram_image": hist_img
        }

# --- Streamlit UI ---

st.title("Fruit Spoilage Estimator Using Image Filtering")

uploaded_file = st.file_uploader("Upload a fruit image", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_cv2 = pil_to_cv2(image_pil)
    
    detector = FruitSpoilageDetector(image_cv2)
    
    with st.spinner("Analyzing image..."):
        results = detector.process_all()

    # Show original image
    st.subheader("Original Image")
    st.image(image_pil, use_column_width=True)

    # Spoilage Score and freshness level
    st.markdown(f"### Spoilage Score: {results['spoilage_score']} / 100")
    st.markdown(f"<span style='color:{results['freshness_color']};font-weight:bold;'>Freshness Level: {results['freshness_level']}</span>", unsafe_allow_html=True)
    
    st.markdown(f"Healthy area: {results['healthy_percentage']}%")
    st.markdown(f"Spoilage area (color analysis): {results['spoilage_percentage']}%")
    st.markdown(f"Number of spoilage spots detected: {results['num_spots']}")
    st.markdown(f"Spoilage area (spot detection): {results['spoilage_area_percentage']}%")
    st.markdown(f"Max distance of spoilage spread: {results['max_distance']}")
    st.markdown(f"Mean image intensity: {results['mean_intensity']}")

    # Display processed images side-by-side
    st.subheader("Processed Images")

    col1, col2 = st.columns(2)

    with col1:
        st.image(results['color_analysis_image'], caption="Color Analysis (Spoilage marked in red)", use_column_width=True)
        st.image(results['edges_image'], caption="Edges (Canny)", use_column_width=True)
        st.image(results['distance_transform_image'], caption="Distance Transform (Spoilage Spread)", use_column_width=True)

    with col2:
        st.image(results['morph_image'], caption="Morphological Top-hat", use_column_width=True)
        st.image(results['spots_detected_image'], caption="Spoilage Spots Detection", use_column_width=True)
        st.image(results['histogram_image'], caption="Color Histogram", use_column_width=True)

else:
    st.info("Please upload an image to begin analysis.")
