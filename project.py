import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Function to check allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Spatial filtering function
def spatial_filtering(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
    return blurred, edges, morph, adaptive_thresh

# Histogram plot function
def plot_histogram(image):
    color = ('b', 'g', 'r')
    plt.figure(figsize=(8, 4))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return plt

st.title("Fruit Spoilage - Filter & Histogram Viewer")

uploaded_file = st.file_uploader("Upload an image", type=list(ALLOWED_EXTENSIONS))
if uploaded_file and allowed_file(uploaded_file.name):
    # Read image as OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)
    
    # Apply filters
    blurred, edges, morph, adaptive_thresh = spatial_filtering(image)
    
    # Show filtered images
    st.subheader("Blurred Image")
    st.image(blurred, use_column_width=True, clamp=True)
    
    st.subheader("Edges Detected (Canny)")
    st.image(edges, use_column_width=True, clamp=True)
    
    st.subheader("Morphological Top Hat")
    st.image(morph, use_column_width=True, clamp=True)
    
    st.subheader("Adaptive Thresholding")
    st.image(adaptive_thresh, use_column_width=True, clamp=True)
    
    # Plot and show histogram
    st.subheader("Color Histogram")
    fig = plot_histogram(image)
    st.pyplot(fig)
else:
    st.info("Please upload an image file of type: " + ", ".join(ALLOWED_EXTENSIONS))
