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
    """
    image_gray: single-channel uint8 image.
    Returns uint8 illumination-corrected image.
    """
    # ensure float in [0,1]
    img = image_gray.astype(np.float32) / 255.0
    # log domain
    img_log = np.log1p(img)

    # frequency transform
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image_gray.shape
    crow, ccol = rows // 2, cols // 2

    # create a simple high-pass mask (suppress low freq near center)
    mask = np.ones((rows, cols), np.float32)
    r = max(15, min(rows, cols) // 10)
    cv2.circle(mask, (ccol, crow), r, 0, -1)

    filtered = dft_shift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    # exponentiate and scale back
    img_exp = np.expm1(img_back)  # inverse of log1p
    # sometimes values go out of range; normalize safely
    img_exp = np.nan_to_num(img_exp, nan=0.0, posinf=0.0, neginf=0.0)
    img_exp = img_exp - img_exp.min()
    if img_exp.max() > 0:
        img_exp = img_exp / img_exp.max()

    img_out = np.uint8(np.clip(img_exp * 255.0, 0, 255))
    return img_out

# =================== Defect Filters ===================
def detect_dark_spots(gray):
    bl = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(bl, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    _, mask = cv2.threshold(lap, 25, 255, cv2.THRESH_BINARY)
    return mask

def detect_wrinkles(gray):
    k = cv2.getGaborKernel((25, 25), 4.0, np.pi / 2, 10.0, 0.5, 0, cv2.CV_32F)
    filtered = cv2.filter2D(gray, ddepth=cv2.CV_8U, kernel=k)
    _, mask = cv2.threshold(filtered, 40, 255, cv2.THRESH_BINARY)
    return mask

def detect_mold(gray):
    # Difference of Gaussians
    g1 = cv2.GaussianBlur(gray, (3, 3), 0)
    g2 = cv2.GaussianBlur(gray, (9, 9), 0)
    dog = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(dog, 15, 255, cv2.THRESH_BINARY)
    return mask

def detect_color_loss(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mean_L = np.mean(L).astype(np.uint8)
    delta = cv2.absdiff(L, mean_L)
    _, mask = cv2.threshold(delta, 18, 255, cv2.THRESH_BINARY)
    return mask

# =================== Spoilage Score ===================
def compute_spoilage_score(mask):
    total_pixels = mask.size
    spoiled_pixels = np.count_nonzero(mask)
    ratio = spoiled_pixels / total_pixels if total_pixels > 0 else 0.0
    score = int(np.clip(ratio * 100, 0, 100))
    if score < 15:
        return "🍏 Fresh", "Minor defects detected", score
    elif score < 35:
        return "🍊 Slightly Spoiled", "Moderate defects present", score
    else:
        return "🍂 Heavily Spoiled", "Significant spoilage detected", score

# =================== Histogram ===================
def plot_histogram(image):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(('b', 'g', 'r')):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
    ax.set_xlim(0, 256)
    fig.tight_layout()
    return fig

# =================== Streamlit App ===================
st.set_page_config(page_title="Smart Fruit Spoilage Detector", layout="wide")
st.title("🍎 Smart Fruit Spoilage Detection System — Improved")

uploaded = st.file_uploader("Upload a fruit image", type=list(ALLOWED_EXTENSIONS))

if uploaded and allowed_file(uploaded.name):
    buf = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode image.")
    else:
        image = resize_fixed(image)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original (resized)", use_column_width=True)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # illumination correction
        hm = homomorphic(gray)

        # detect defects on illumination-normalized image
        dark = detect_dark_spots(hm)
        wrinkle = detect_wrinkles(hm)
        mold = detect_mold(hm)
        colorfade = detect_color_loss(image)

        # combine masks
        combined = np.zeros_like(dark)
        combined = cv2.bitwise_or(combined, dark)
        combined = cv2.bitwise_or(combined, wrinkle)
        combined = cv2.bitwise_or(combined, mold)
        combined = cv2.bitwise_or(combined, colorfade)

        # optional smoothing/cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # ensure same size display
        disp_hm = resize_fixed(hm)
        disp_dark = resize_fixed(dark)
        disp_wrinkle = resize_fixed(wrinkle)
        disp_mold = resize_fixed(mold)
        disp_colorfade = resize_fixed(colorfade)
        disp_combined = resize_fixed(combined)

        st.markdown("### 🔍 Defect Filters (uniform size)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.image(disp_hm, caption="Illumination Normalized", use_column_width=True)
        c2.image(disp_dark, caption="Dark Spots", use_column_width=True)
        c3.image(disp_wrinkle, caption="Wrinkles", use_column_width=True)
        c4.image(disp_mold, caption="Mold", use_column_width=True)
        c5.image(disp_colorfade, caption="Color Fading", use_column_width=True)

        st.markdown("### ✅ Combined Spoilage Mask")
        st.image(disp_combined, caption="Combined Mask", use_column_width=True)

        label, desc, score = compute_spoilage_score(combined)
        st.subheader(f"Result: {label}")
        st.info(desc)
        st.progress(score)

        st.markdown("### 📊 Color Histogram")
        st.pyplot(plot_histogram(image))

else:
    st.info("Upload a png/jpg/jpeg/bmp image.")
