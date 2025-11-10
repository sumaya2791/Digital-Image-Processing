import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =================== Utility ===================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

TARGET_SIZE = (400, 400)

def resize_fixed(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

# =================== Homomorphic Filter ===================
def homomorphic(image_gray):
    img = image_gray.astype(np.float32) / 255
    img_log = np.log1p(img)

    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img_gray.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.float32)
    r = 30
    cv2.circle(mask, (ccol, crow), r, 0, -1)

    filtered = dft_shift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.exp(np.real(img_back)) - 1
    img_back = np.uint8(np.clip(img_back * 255, 0, 255))

    return img_back

# =================== Defect Filters ===================
def detect_dark_spots(gray):
    log = cv2.GaussianBlur(gray, (5,5), 0)
    log = cv2.Laplacian(log, cv2.CV_64F)
    log = cv2.convertScaleAbs(log)
    _, mask = cv2.threshold(log, 25, 255, cv2.THRESH_BINARY)
    return mask

def detect_wrinkles(gray):
    gabor_kernel = cv2.getGaborKernel((25,25), 4.0, np.pi/2, 10.0, 0.5, 0, cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8U, gabor_kernel)
    _, mask = cv2.threshold(filtered, 40, 255, cv2.THRESH_BINARY)
    return mask

def detect_mold(gray):
    blurred = cv2.GaussianBlur(gray, (9,9), 0)
    dog = cv2.GaussianBlur(gray,(0,0),3) - cv2.GaussianBlur(gray,(0,0),1)
    dog = cv2.convertScaleAbs(dog)
    _, mask = cv2.threshold(dog, 20, 255, cv2.THRESH_BINARY)
    return mask

def detect_color_loss(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mean_L = np.mean(L)
    delta = cv2.absdiff(L, mean_L.astype(np.uint8))
    _, mask = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)
    return mask

# =================== Spoilage Score ===================
def compute_spoilage_score(mask):
    total_pixels = mask.size
    spoiled_pixels = np.count_nonzero(mask)
    ratio = spoiled_pixels / total_pixels

    score = int(ratio * 100)

    if score < 15:
        return "🍏 Fresh", "Light defects", score
    elif score < 35:
        return "🍊 Slightly Spoiled", "Moderate surface defects", score
    else:
        return "🍂 Heavily Spoiled", "Large spoilage patches", score

# =================== Histogram ===================
def plot_histogram(image):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(('b','g','r')):
        hist = cv2.calcHist([image], [i], None, [256], [0,256])
        ax.plot(hist, color=col)
    ax.set_xlim(0,256)
    fig.tight_layout()
    return fig

# =================== Streamlit App ===================
st.set_page_config(page_title="Smart Fruit Spoilage Detector", layout="wide")
st.title("🍎 Smart Fruit Spoilage Detection System — Improved")

uploaded = st.file_uploader("Upload a fruit image", type=list(ALLOWED_EXTENSIONS))

if uploaded and allowed_file(uploaded.name):
    file_bytes = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not load image.")
    else:
        image = resize_fixed(image)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = resize_fixed(gray)

        # Homomorphic for smooth illumination
        hm = homomorphic(gray)

        # Defect maps
        dark = detect_dark_spots(hm)
        wrinkle = detect_wrinkles(hm)
        mold = detect_mold(hm)
        colorfade = detect_color_loss(image)

        combined_mask = cv2.bitwise_or(dark, wrinkle)
        combined_mask = cv2.bitwise_or(combined_mask, mold)
        combined_mask = cv2.bitwise_or(combined_mask, colorfade)

        # Display filters (same size)
        st.markdown("### 🔍 Defect Filters")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.image(resize_fixed(hm), caption="Illumination Normalized", use_column_width=True)
        c2.image(resize_fixed(dark), caption="Dark Spots", use_column_width=True)
        c3.image(resize_fixed(wrinkle), caption="Wrinkles", use_column_width=True)
        c4.image(resize_fixed(mold), caption="Mold", use_column_width=True)
        c5.image(resize_fixed(colorfade), caption="Color Fading", use_column_width=True)

        st.markdown("### ✅ Combined Spoilage Mask")
        st.image(resize_fixed(combined_mask), use_column_width=True)

        # Score
        label, desc, score = compute_spoilage_score(combined_mask)
        st.subheader(f"Result: {label}")
        st.info(desc)
        st.progress(score)

        st.markdown("### 📊 Color Histogram")
        st.pyplot(plot_histogram(image))

else:
    st.info("Upload a png/jpg/jpeg/bmp image.")
