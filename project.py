import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
import pandas as pd

# ------------------------- Utility -------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------- Enhanced Spatial Filters -------------------------
def spatial_filtering(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Gaussian Blur for denoising
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Bilateral Filter (preserves edges while smoothing)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. Median Filter (removes salt-and-pepper noise)
    median = cv2.medianBlur(gray, 5)

    # 4. Sobel Filter (edge gradients)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # 5. Laplacian (detects fine surface defects)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # 6. Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # 7. Morphological enhancement (Top Hat - detects bright spots)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 8. Black Hat (detects dark spots/defects)
    morph_blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 9. Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 10. Unsharp Masking (enhance details)
    unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # 11. Local Variance (texture roughness map)
    variance_map = ndimage.generic_filter(gray, np.var, size=9)
    variance_map = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        'blurred': blurred,
        'bilateral': bilateral,
        'median': median,
        'sobel': sobel_combined,
        'laplacian': laplacian,
        'edges': edges,
        'tophat': morph_tophat,
        'blackhat': morph_blackhat,
        'adaptive_thresh': adaptive_thresh,
        'unsharp': unsharp,
        'variance_map': variance_map
    }

# ------------------------- Spoilage Region Detection -------------------------
def detect_spoilage_regions(image):
    """
    Detect and segment spoiled regions using color and texture analysis
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Brown/Dark spot detection in HSV
    # Brown/dark spots typically have low V and specific H range
    lower_brown = np.array([5, 20, 20])
    upper_brown = np.array([30, 255, 120])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Method 2: Dark spots (low brightness)
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Method 3: Color deviation from mean (abnormal colors)
    mean_color = np.mean(image, axis=(0, 1))
    color_diff = np.abs(image - mean_color)
    color_dev = np.max(color_diff, axis=2).astype(np.uint8)
    _, deviation_mask = cv2.threshold(color_dev, 50, 255, cv2.THRESH_BINARY)
    
    # Method 4: Texture-based (high local variance = rough/wrinkled)
    variance_map = ndimage.generic_filter(gray, np.var, size=9)
    variance_norm = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(variance_norm, 100, 255, cv2.THRESH_BINARY)
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(brown_mask, dark_mask)
    combined_mask = cv2.bitwise_or(combined_mask, deviation_mask)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of defects
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return combined_mask, brown_mask, dark_mask, deviation_mask, texture_mask, contours

# ------------------------- Defect Analysis -------------------------
def analyze_defects(image, contours, mask):
    """
    Analyze detected defects and extract statistics
    """
    total_area = image.shape[0] * image.shape[1]
    defect_area = cv2.countNonZero(mask)
    defect_percentage = (defect_area / total_area) * 100
    
    defect_count = len([c for c in contours if cv2.contourArea(c) > 50])  # Filter small noise
    
    defect_sizes = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
    avg_defect_size = np.mean(defect_sizes) if defect_sizes else 0
    max_defect_size = np.max(defect_sizes) if defect_sizes else 0
    
    return {
        'total_area': total_area,
        'defect_area': defect_area,
        'defect_percentage': defect_percentage,
        'defect_count': defect_count,
        'avg_defect_size': avg_defect_size,
        'max_defect_size': max_defect_size,
        'defect_sizes': defect_sizes
    }

# ------------------------- Visualization -------------------------
def draw_defect_overlay(image, contours):
    """
    Draw bounding boxes and highlights on detected defects
    """
    overlay = image.copy()
    output = image.copy()
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 50:  # Filter small noise
            # Draw filled contour
            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
            
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            area = cv2.contourArea(contour)
            cv2.putText(output, f"#{i+1}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Blend overlay
    blended = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
    
    return blended, output

# ------------------------- Color Clustering -------------------------
def color_clustering(image, n_clusters=5):
    """
    K-means clustering to identify dominant colors and abnormalities
    """
    # Reshape image to list of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Get cluster centers (dominant colors)
    centers = kmeans.cluster_centers_.astype(int)
    
    # Count pixels in each cluster
    counts = np.bincount(labels)
    
    # Create segmented image
    segmented = centers[labels].reshape(image.shape)
    
    return segmented, centers, counts, labels.reshape(image.shape[:2])

# ------------------------- Advanced Color & Texture Analysis -------------------------
def analyze_color_texture(image):
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_hue = np.mean(h)
    std_hue = np.std(h)
    mean_sat = np.mean(s)
    std_sat = np.std(s)
    mean_val = np.mean(v)
    std_val = np.std(v)

    # Convert to LAB (perceptual lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_lightness = np.mean(l)
    std_lightness = np.std(l)

    # Texture metrics
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Edge density (indicator of roughness)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = (np.count_nonzero(edges) / edges.size) * 100
    
    # Color uniformity (lower std = more uniform = fresher)
    color_uniformity = 100 - (std_sat / 255 * 100)

    return {
        'mean_hue': mean_hue,
        'std_hue': std_hue,
        'mean_sat': mean_sat,
        'std_sat': std_sat,
        'mean_val': mean_val,
        'std_val': std_val,
        'mean_lightness': mean_lightness,
        'std_lightness': std_lightness,
        'lap_var': lap_var,
        'edge_density': edge_density,
        'color_uniformity': color_uniformity
    }

# ------------------------- Enhanced Classification -------------------------
def classify_spoilage(color_metrics, defect_stats):
    """
    Advanced classification using multiple metrics
    """
    # Color score (freshness based on saturation and variance)
    color_score = (color_metrics['mean_sat'] / 255) * 100
    color_uniformity_score = color_metrics['color_uniformity']
    
    # Texture score (roughness)
    texture_score = np.clip(color_metrics['lap_var'] / 100, 0, 100)
    
    # Defect score (based on detected spoilage area)
    defect_score = defect_stats['defect_percentage']
    
    # Edge density score (wrinkled surface)
    edge_score = color_metrics['edge_density']
    
    # Weighted spoilage score
    spoilage_score = (
        (100 - color_score) * 0.25 +          # Low saturation = spoiled
        (100 - color_uniformity_score) * 0.20 +  # Non-uniform = spoiled
        texture_score * 0.15 +                   # Rough texture = spoiled
        defect_score * 0.30 +                    # Defect area = spoiled
        edge_score * 0.10                        # High edges = wrinkled
    )
    
    spoilage_score = np.clip(spoilage_score, 0, 100)
    
    # Classification
    if spoilage_score < 25:
        label = "🍏 Fresh"
        desc = "✅ Excellent condition: Bright, uniform color with smooth texture and no visible defects."
        recommendation = "Safe to consume. Store properly to maintain freshness."
    elif 25 <= spoilage_score < 45:
        label = "🍊 Slightly Aged"
        desc = "🟡 Good condition: Minor color changes or small surface irregularities detected."
        recommendation = "Safe to consume. Use soon for best quality."
    elif 45 <= spoilage_score < 65:
        label = "🍂 Moderately Spoiled"
        desc = "🟠 Fair condition: Noticeable discoloration, rough texture, or visible defect spots."
        recommendation = "Inspect carefully. Cut away affected areas if consuming."
    else:
        label = "🥀 Heavily Spoiled"
        desc = "⚠️ Poor condition: Significant spoilage detected with dull colors, rough texture, and large defect areas."
        recommendation = "Not recommended for consumption. Discard to prevent health risks."
    
    breakdown = {
        'Color Vibrancy': 100 - ((100 - color_score) * 0.25 / spoilage_score * 100) if spoilage_score > 0 else 100,
        'Color Uniformity': color_uniformity_score,
        'Texture Smoothness': 100 - texture_score,
        'Defect-Free Area': 100 - defect_score,
        'Surface Integrity': 100 - edge_score
    }
    
    return label, desc, recommendation, spoilage_score, breakdown

# ------------------------- Histogram Plot -------------------------
def plot_histogram(image):
    color = ('b', 'g', 'r')
    labels = ('Blue', 'Green', 'Red')
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (col, label) in enumerate(zip(color, labels)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, label=label, alpha=0.7, linewidth=2)
        ax.set_xlim([0, 256])
    ax.set_title('RGB Color Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel Intensity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ------------------------- HSV Histogram -------------------------
def plot_hsv_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    channels = ['Hue', 'Saturation', 'Value']
    colors = ['orange', 'green', 'blue']
    
    for i, (ax, channel, color) in enumerate(zip(axes, channels, colors)):
        hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, linewidth=2)
        ax.set_title(f'{channel} Distribution', fontweight='bold')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

# ------------------------- Defect Statistics Plot -------------------------
def plot_defect_stats(defect_sizes):
    if not defect_sizes:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram of defect sizes
    ax1.hist(defect_sizes, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax1.set_title('Defect Size Distribution', fontweight='bold')
    ax1.set_xlabel('Defect Area (pixels²)')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(defect_sizes, vert=True)
    ax2.set_title('Defect Size Statistics', fontweight='bold')
    ax2.set_ylabel('Defect Area (pixels²)')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

# ------------------------- Streamlit App -------------------------
st.set_page_config(page_title="Advanced Fruit Spoilage Detector", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🍎 Advanced Fruit Spoilage Detection System")
st.markdown("### AI-Powered Quality Analysis with Multi-Filter Processing")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    show_all_filters = st.checkbox("Show All Filters", value=False)
    show_masks = st.checkbox("Show Detection Masks", value=True)
    show_clustering = st.checkbox("Show Color Clustering", value=True)
    n_clusters = st.slider("Number of Color Clusters", 3, 8, 5)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    This system uses advanced computer vision techniques to detect fruit spoilage:
    
    - **11 Spatial Filters** for texture analysis
    - **Multi-method Spoilage Detection**
    - **Color Space Analysis** (RGB, HSV, LAB)
    - **Defect Localization & Quantification**
    - **K-means Color Clustering**
    """)

uploaded_file = st.file_uploader("📤 Upload a fruit image for analysis", type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Could not load the image. Please upload a valid file.")
    else:
        # Display original image
        col_orig1, col_orig2 = st.columns([1, 1])
        with col_orig1:
            st.markdown("### 📸 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Quick stats
        with col_orig2:
            st.markdown("### 📊 Image Information")
            st.markdown(f"""
            <div class="highlight">
            <b>Dimensions:</b> {image.shape[1]} × {image.shape[0]} pixels<br>
            <b>Total Pixels:</b> {image.shape[0] * image.shape[1]:,}<br>
            <b>Color Channels:</b> {image.shape[2]}<br>
            <b>File Name:</b> {uploaded_file.name}
            </div>
            """, unsafe_allow_html=True)
        
        # Progress indicator
        with st.spinner('🔄 Processing image with advanced filters...'):
            # Apply filters
            filters = spatial_filtering(image)
            
            # Detect spoilage regions
            combined_mask, brown_mask, dark_mask, deviation_mask, texture_mask, contours = detect_spoilage_regions(image)
            
            # Analyze defects
            defect_stats = analyze_defects(image, contours, combined_mask)
            
            # Color and texture analysis
            color_metrics = analyze_color_texture(image)
            
            # Classification
            label, desc, recommendation, spoilage_score, breakdown = classify_spoilage(color_metrics, defect_stats)
            
            # Draw defect overlay
            blended_overlay, bbox_overlay = draw_defect_overlay(image, contours)
            
            # Color clustering
            if show_clustering:
                segmented, centers, counts, cluster_labels = color_clustering(image, n_clusters)
        
        st.success('✅ Analysis Complete!')
        
        # ==================== MAIN RESULTS ====================
        st.markdown("---")
        st.markdown("## 🎯 Analysis Results")
        
        # Classification Result
        st.markdown(f"### {label}")
        st.markdown(f"<div class='highlight'>{desc}</div>", unsafe_allow_html=True)
        st.markdown(f"**💡 Recommendation:** {recommendation}")
        
        # Spoilage Score
        col_score1, col_score2 = st.columns([2, 1])
        with col_score1:
            st.markdown("### 📈 Overall Spoilage Score")
            st.progress(int(spoilage_score))
            st.markdown(f"<p class='big-font'>Score: {spoilage_score:.1f}/100</p>", unsafe_allow_html=True)
        
        with col_score2:
            st.markdown("### 🎨 Quality Breakdown")
            for metric, value in breakdown.items():
                st.metric(metric, f"{value:.1f}%")
        
        # ==================== DEFECT DETECTION ====================
        st.markdown("---")
        st.markdown("## 🔍 Spoilage Region Detection")
        
        col_def1, col_def2 = st.columns(2)
        with col_def1:
            st.markdown("### 🎯 Defects Highlighted")
            st.image(cv2.cvtColor(blended_overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col_def2:
            st.markdown("### 📦 Bounding Boxes")
            st.image(cv2.cvtColor(bbox_overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Defect Statistics
        st.markdown("### 📊 Defect Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("🔴 Defect Count", f"{defect_stats['defect_count']}")
        col_stat2.metric("📏 Total Defect Area", f"{defect_stats['defect_area']:,} px²")
        col_stat3.metric("📊 Spoiled Area %", f"{defect_stats['defect_percentage']:.2f}%")
        col_stat4.metric("📐 Avg Defect Size", f"{defect_stats['avg_defect_size']:.0f} px²")
        
        # Defect size distribution
        if defect_stats['defect_sizes']:
            st.markdown("### 📈 Defect Size Analysis")
            defect_fig = plot_defect_stats(defect_stats['defect_sizes'])
            if defect_fig:
                st.pyplot(defect_fig)
        
        # ==================== DETECTION MASKS ====================
        if show_masks:
            st.markdown("---")
            st.markdown("## 🎭 Detection Masks (Multi-Method)")
            col_mask1, col_mask2, col_mask3 = st.columns(3)
            
            with col_mask1:
                st.image(brown_mask, caption="🟤 Brown/Dark Spot Mask", use_column_width=True)
                st.image(dark_mask, caption="⚫ Low Brightness Mask", use_column_width=True)
            
            with col_mask2:
                st.image(deviation_mask, caption="🌈 Color Deviation Mask", use_column_width=True)
                st.image(texture_mask, caption="🔲 Texture Roughness Mask", use_column_width=True)
            
            with col_mask3:
                st.image(combined_mask, caption="✅ Combined Defect Mask", use_column_width=True)
                # Create colored overlay
                colored_mask = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB), 
                        caption="🌡️ Heatmap Visualization", use_column_width=True)
        
        # ==================== SPATIAL FILTERS ====================
        if show_all_filters:
            st.markdown("---")
            st.markdown("## 🔬 Spatial Filter Analysis (11 Filters)")
            
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            
            with col_f1:
                st.image(filters['blurred'], caption="1️⃣ Gaussian Blur", use_column_width=True)
                st.image(filters['sobel'], caption="5️⃣ Sobel Edge", use_column_width=True)
                st.image(filters['adaptive_thresh'], caption="9️⃣ Adaptive Threshold", use_column_width=True)
            
            with col_f2:
                st.image(filters['bilateral'], caption="2️⃣ Bilateral Filter", use_column_width=True)
                st.image(filters['laplacian'], caption="6️⃣ Laplacian Edge", use_column_width=True)
                st.image(filters['unsharp'], caption="🔟 Unsharp Mask", use_column_width=True)
            
            with col_f3:
                st.image(filters['median'], caption="3️⃣ Median Filter", use_column_width=True)
                st.image(filters['edges'], caption="7️⃣ Canny Edges", use_column_width=True)
                st.image(filters['variance_map'], caption="1️⃣1️⃣ Variance Map", use_column_width=True)
            
            with col_f4:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), caption="4️⃣ Grayscale", use_column_width=True)
                st.image(filters['tophat'], caption="8️⃣ Top Hat (Bright)", use_column_width=True)
                st.image(filters['blackhat'], caption="8️⃣ Black Hat (Dark)", use_column_width=True)
        
        # ==================== COLOR ANALYSIS ====================
        st.markdown("---")
        st.markdown("## 🎨 Color & Texture Metrics")
        
        col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)
        col_met1.metric("🌈 Mean Hue", f"{color_metrics['mean_hue']:.1f}")
        col_met2.metric("💎 Mean Saturation", f"{color_metrics['mean_sat']:.1f}")
        col_met3.metric("💡 Mean Brightness", f"{color_metrics['mean_val']:.1f}")
        col_met4.metric("🔆 Mean Lightness", f"{color_metrics['mean_lightness']:.1f}")
        col_met5.metric("📊 Texture Variance", f"{color_metrics['lap_var']:.1f}")
        
        col_met6, col_met7, col_met8, col_met9, col_met10 = st.columns(5)
        col_met6.metric("📈 Hue Std Dev", f"{color_metrics['std_hue']:.1f}")
        col_met7.metric("📈 Saturation Std Dev", f"{color_metrics['std_sat']:.1f}")
        col_met8.metric("📈 Brightness Std Dev", f"{color_metrics['std_val']:.1f}")
        col_met9.metric("🎯 Edge Density", f"{color_metrics['edge_density']:.2f}%")
        col_met10.metric("✨ Color Uniformity", f"{color_metrics['color_uniformity']:.1f}%")
        
        # ==================== HISTOGRAMS ====================
        st.markdown("---")
        st.markdown("## 📊 Color Distribution Analysis")
        
        col_hist1, col_hist2 = st.columns(2)
        
        with col_hist1:
            st.markdown("### RGB Histogram")
            hist_fig = plot_histogram(image)
            st.pyplot(hist_fig)
        
        with col_hist2:
            st.markdown("### HSV Histogram")
            hsv_fig = plot_hsv_histogram(image)
            st.pyplot(hsv_fig)
        
        # ==================== COLOR CLUSTERING ====================
        if show_clustering:
            st.markdown("---")
            st.markdown("## 🎨 K-Means Color Clustering Analysis")
            
            col_clust1, col_clust2 = st.columns(2)
            
            with col_clust1:
                st.markdown("### Segmented Image")
                st.image(cv2.cvtColor(segmented.astype(np.uint8), cv2.COLOR_BGR2RGB), 
                        use_column_width=True)
            
            with col_clust2:
                st.markdown("### Dominant Colors")
                
                # Create color palette
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Sort by frequency
                sorted_indices = np.argsort(counts)[::-1]
                
                for i, idx in enumerate(sorted_indices):
                    color_rgb = centers[idx][::-1] / 255  # Convert BGR to RGB and normalize
                    percentage = (counts[idx] / np.sum(counts)) * 100
                    
                    ax.barh(i, percentage, color=color_rgb, edgecolor='black', linewidth=2)
                    ax.text(percentage + 1, i, f'{percentage:.1f}%', va='center', fontweight='bold')
                
                ax.set_xlabel('Percentage of Image', fontsize=12, fontweight='bold')
                ax.set_ylabel('Color Cluster', fontsize=12, fontweight='bold')
                ax.set_title('Dominant Color Distribution', fontsize=14, fontweight='bold')
                ax.set_yticks(range(n_clusters))
                ax.set_yticklabels([f'Color {i+1}' for i in range(n_clusters)])
                ax.grid(axis='x', alpha=0.3)
                fig.tight_layout()
                
                st.pyplot(fig)
                
                # Color cluster details
                st.markdown("### Color Cluster Details")
                cluster_data = []
                for i, idx in enumerate(sorted_indices):
                    b, g, r = centers[idx]
                    percentage = (counts[idx] / np.sum(counts)) * 100
                    cluster_data.append({
                        'Cluster': f'Color {i+1}',
                        'RGB': f'({r}, {g}, {b})',
                        'Percentage': f'{percentage:.2f}%',
                        'Pixel Count': f'{counts[idx]:,}'
                    })
                
                df = pd.DataFrame(cluster_data)
                st.dataframe(df, use_container_width=True)
        
        # ==================== DOWNLOAD SECTION ====================
        st.markdown("---")
        st.markdown("## 💾 Download Results")
        
        col_down1, col_down2, col_down3 = st.columns(3)
        
        with col_down1:
            # Save annotated image
            _, buffer = cv2.imencode('.png', blended_overlay)
            st.download_button(
                label="📥 Download Annotated Image",
                data=buffer.tobytes(),
                file_name="spoilage_detected.png",
                mime="image/png"
            )
        
        with col_down2:
            # Save mask
            _, buffer_mask = cv2.imencode('.png', combined_mask)
            st.download_button(
                label="📥 Download Defect Mask",
                data=buffer_mask.tobytes(),
                file_name="defect_mask.png",
                mime="image/png"
            )
        
        with col_down3:
            # Save report
            report = f"""
FRUIT SPOILAGE DETECTION REPORT
================================

Image: {uploaded_file.name}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

CLASSIFICATION
--------------
Status: {label}
Spoilage Score: {spoilage_score:.2f}/100
Description: {desc}
Recommendation: {recommendation}

DEFECT STATISTICS
-----------------
Defect Count: {defect_stats['defect_count']}
Total Defect Area: {defect_stats['defect_area']:,} pixels²
Spoiled Area Percentage: {defect_stats['defect_percentage']:.2f}%
Average Defect Size: {defect_stats['avg_defect_size']:.0f} pixels²
Maximum Defect Size: {defect_stats['max_defect_size']:.0f} pixels²

COLOR METRICS
-------------
Mean Hue: {color_metrics['mean_hue']:.2f}
Mean Saturation: {color_metrics['mean_sat']:.2f}
Mean Brightness: {color_metrics['mean_val']:.2f}
Mean Lightness (LAB): {color_metrics['mean_lightness']:.2f}
Color Uniformity: {color_metrics['color_uniformity']:.2f}%

TEXTURE METRICS
---------------
Laplacian Variance: {color_metrics['lap_var']:.2f}
Edge Density: {color_metrics['edge_density']:.2f}%

QUALITY BREAKDOWN
-----------------
"""
            for metric, value in breakdown.items():
                report += f"{metric}: {value:.2f}%\n"
            
            st.download_button(
                label="📥 Download Analysis Report",
                data=report,
                file_name="spoilage_report.txt",
                mime="text/plain"
            )

else:
    # Landing page
    st.markdown("""
    ## 🚀 Welcome to Advanced Fruit Spoilage Detection
    
    Upload an image to begin comprehensive quality analysis using:
    
    ### 🔬 Analysis Features:
    - ✅ **11 Advanced Spatial Filters** (Gaussian, Bilateral, Median, Sobel, Laplacian, Canny, Top Hat, Black Hat, Adaptive Threshold, Unsharp Mask, Variance Map)
    - ✅ **Multi-Method Spoilage Detection** (Color-based, Texture-based, Statistical deviation)
    - ✅ **Precise Defect Localization** with bounding boxes and overlays
    - ✅ **Quantitative Metrics** (defect count, area, percentage)
    - ✅ **Color Space Analysis** (RGB, HSV, LAB)
    - ✅ **K-Means Color Clustering** to identify dominant colors and abnormalities
    - ✅ **Comprehensive Quality Scoring** with detailed breakdown
    - ✅ **Downloadable Reports** and annotated images
    
    ### 📤 Supported Formats:
    PNG, JPG, JPEG, BMP
    
    ### 🎯 Use Cases:
    - Quality control in food processing
    - Retail freshness assessment
    - Supply chain monitoring
    - Consumer safety verification
    """)
    
    # Example images section
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results:")
    st.info("""
    - Use clear, well-lit images
    - Ensure the fruit fills most of the frame
    - Avoid heavy shadows or reflections
    - Use high-resolution images for more accurate detection
    """)
