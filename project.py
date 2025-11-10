import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

ALLOWED_EXTENSIONS = {'png','jpg','jpeg','bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------- Spatial Filters -------------------------
def spatial_filtering(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray,(5,5),0)

    sobelx = cv2.Sobel(blurred,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(blurred,cv2.CV_64F,0,1,ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sobelx,sobely))

    laplacian = cv2.convertScaleAbs(cv2.Laplacian(blurred,cv2.CV_64F))

    edges = cv2.Canny(blurred,50,150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    morph = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel)

    adaptive_thresh = cv2.adaptiveThreshold(
        blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,11,2
    )

    # CLAHE contrast
    clahe_obj = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    clahe_img = clahe_obj.apply(gray)

    # Otsu threshold
    _, otsu = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Gabor filter
    gabor_kernel = cv2.getGaborKernel((21,21),4.0,np.pi/4,10.0,0.5,0,ktype=cv2.CV_32F)
    gabor = cv2.filter2D(gray,CV_8UC3,gabor_kernel)

    return blurred,sobel,laplacian,edges,morph,adaptive_thresh,clahe_img,otsu,gabor

# ------------------------- Color & Texture Analysis -------------------------
def analyze_color_texture(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    mean_hue = np.mean(h)
    mean_sat = np.mean(s)

    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    mean_lightness = np.mean(l)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray,cv2.CV_64F).var()

    # GLCM roughness proxy: local variance
    local_var = cv2.Laplacian(gray,cv2.CV_64F).var()

    return mean_hue,mean_sat,mean_lightness,lap_var,local_var

# ------------------------- Spoilage Classification -------------------------
def classify_spoilage(mean_hue,mean_sat,mean_lightness,lap_var,local_var):
    color_score = (mean_sat/255)*100
    texture_score = np.clip((lap_var+local_var)/150,0,100)

    spoilage_score = (100 - color_score)*0.5 + texture_score*0.5

    if spoilage_score < 30:
        label = "Fresh"
        desc = "Color stable. Texture smooth."
    elif spoilage_score < 60:
        label = "Slightly Spoiled"
        desc = "Color dullness or mild texture roughness."
    else:
        label = "Heavily Spoiled"
        desc = "Low saturation and high texture deformation."

    return label,desc,spoilage_score

# ------------------------- Histogram -------------------------
def plot_histogram(image):
    fig,ax = plt.subplots(figsize=(8,4))
    for i,col in enumerate(('b','g','r')):
        hist = cv2.calcHist([image],[i],None,[256],[0,256])
        ax.plot(hist,color=col)
    ax.set_xlim([0,256])
    fig.tight_layout()
    return fig

# ------------------------- Streamlit -------------------------
st.set_page_config(page_title="Smart Fruit Spoilage Detector",layout="wide")
st.title("Smart Fruit Spoilage Detection")

uploaded_file = st.file_uploader("Upload image",type=list(ALLOWED_EXTENSIONS))

if uploaded_file and allowed_file(uploaded_file.name):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()),dtype=np.uint8)
    image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

    if image is None:
        st.error("Invalid image.")
    else:
        st.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),caption='Original',use_column_width=True)

        outputs = spatial_filtering(image)
        (blurred,sobel,laplacian,edges,morph,adaptive,clahe_img,otsu,gabor_img) = outputs

        st.markdown("### Spatial Filters")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.image(blurred,caption="Gaussian Blur",use_column_width=True)
            st.image(sobel,caption="Sobel",use_column_width=True)
            st.image(clahe_img,caption="CLAHE",use_column_width=True)
        with c2:
            st.image(laplacian,caption="Laplacian",use_column_width=True)
            st.image(edges,caption="Canny",use_column_width=True)
            st.image(otsu,caption="Otsu Threshold",use_column_width=True)
        with c3:
            st.image(morph,caption="Top Hat",use_column_width=True)
            st.image(adaptive,caption="Adaptive Threshold",use_column_width=True)
            st.image(gabor_img,caption="Gabor Texture",use_column_width=True)

        mean_h,mean_s,mean_l,lap_var,local_var = analyze_color_texture(image)
        label,desc,score = classify_spoilage(mean_h,mean_s,mean_l,lap_var,local_var)

        st.markdown("### Color and Texture Analysis")
        a,b,c,d,e = st.columns(5)
        a.metric("Hue",f"{mean_h:.2f}")
        b.metric("Saturation",f"{mean_s:.2f}")
        c.metric("Lightness",f"{mean_l:.2f}")
        d.metric("Laplacian Var",f"{lap_var:.2f}")
        e.metric("Local Var",f"{local_var:.2f}")

        st.markdown(f"### Result: {label}")
        st.info(desc)
        st.progress(int(np.clip(score,0,100)))

        st.markdown("### Histogram")
        fig = plot_histogram(image)
        st.pyplot(fig)

else:
    st.info("Upload a valid image.")
