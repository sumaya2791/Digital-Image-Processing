import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Page config
st.set_page_config(layout="wide")
st.title("🍎 Fruit Spoilage Detection & Histogram Analysis")

# Constants
def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def detect_spoilage_area(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return mask, dist

# Upload multiple images
uploaded_files = st.file_uploader("Upload daily fruit images (PNG or JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    images = []
    filenames = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        image = image.resize((512, 512))
        img_np = np.array(image)
        images.append(img_np)
        filenames.append(file.name)

    # Histogram Comparison
    st.subheader("📊 Histogram Comparison Over Days")
    fig, ax = plt.subplots()
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = calculate_histogram(gray)
        ax.plot(hist, label=filenames[i])
    ax.set_title("Histogram Comparison Over Days")
    ax.legend()
    st.pyplot(fig)

    # Spoilage Detection for each image
    st.subheader("🧪 Spoilage Detection Visualization")

    for i, img in enumerate(images):
        mask, dist = detect_spoilage_area(img)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Original Image", use_column_width=True)
        with col2:
            st.image(mask, caption="Spoilage Mask", use_column_width=True, clamp=True)
        with col3:
            st.image(dist, caption="Distance Transform", use_column_width=True, clamp=True)

        st.markdown(f"**Day/Image:** {filenames[i]}")
        st.markdown("---")
else:
    st.info("Upload 1 or more fruit images to start analysis.")
