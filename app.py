import streamlit as st
import cv2
import numpy as np
import os
from splitter.opencv_splitter import split_image_contour, split_image_xycut

st.set_page_config(page_title="Figure Split Tester", layout="wide")

st.title("Figure Split Tester 🔍")

os.makedirs("outputs/crops", exist_ok=True)
os.makedirs("outputs/debug", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

st.sidebar.header("Testing Parameters")
mode = st.sidebar.selectbox("Select Method", ["OpenAI Smart Split", "XY-Cut (Whitespace)", "OpenCV Contour", "SAM"])

if mode == "OpenCV Contour":
    st.sidebar.subheader("Contour Configuration")
    preprocess_mode = st.sidebar.radio("Preprocessing Mode", ["Threshold Line Art (White Backgrounds)", "Canny Edge (Photos/Complex)"])
    min_area = st.sidebar.slider("Min Area (pixels)", 500, 200000, 4000, step=500)
    dilate_iter = st.sidebar.slider("Dilation Iterations", 0, 10, 2)
    kernel_size_val = st.sidebar.slider("Kernel Size (odd number)", 3, 101, 21, step=2)
    kernel_size = (kernel_size_val, kernel_size_val)

elif mode == "XY-Cut (Whitespace)":
    st.sidebar.subheader("XY-Cut Configuration")
    st.sidebar.info("Best for technical drawings. Cuts the image along horizontal/vertical white spaces.")
    min_area = st.sidebar.slider("Min Area (pixels)", 500, 200000, 4000, step=500)
    min_gap = st.sidebar.slider("Min Whitespace Gap (pixels)", 5, 200, 20, step=5)
    noise_threshold = st.sidebar.slider("Noise Threshold (pixel sum)", 0, 50, 5)
    dilate_kernel = st.sidebar.slider("Pre-merge Kernel (0 to disable)", 0, 21, 5)

elif mode == "OpenAI Smart Split":
    st.sidebar.subheader("OpenAI Configuration")
    st.sidebar.info("Uses GPT-4o to semantically group distinct machine sub-diagrams!")
    openai_min_area = st.sidebar.slider("Initial Min Area", 50, 5000, 100, step=50)

uploaded_file = st.file_uploader("Upload Figure Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Original Image")
    st.image(image, channels="BGR", caption="Original Image")

    intermediate_img = None

    if mode == "OpenCV Contour":
        boxes = split_image_contour(image, min_area=min_area, kernel_size=kernel_size, dilate_iter=dilate_iter, preprocess_mode=preprocess_mode)
    elif mode == "XY-Cut (Whitespace)":
        boxes = split_image_xycut(image, min_area=min_area, min_gap=min_gap, noise_threshold=noise_threshold, dilate_kernel=dilate_kernel)
    elif mode == "OpenAI Smart Split":
        from splitter.openai_splitter import split_image_openai
        with st.spinner("AI is determining semantic bounding box layouts..."):
            boxes, intermediate_img = split_image_openai(image, min_area=openai_min_area)
    else:
        st.info("SAM splitting not yet implemented.")
        boxes = []

    if intermediate_img is not None:
        st.markdown("### Step 1: Base Contour Generation (Sent to AI)")
        st.image(intermediate_img, channels="BGR", caption="Intermediate Numbered Boxes")

    if boxes:
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        debug_img = image.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, f"Crop {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        st.subheader("Detected Regions (Final)")
        st.image(debug_img, channels="BGR", caption="Detected Regions")

        st.markdown(f"### Cropped Images ({len(boxes)} detected)")
        
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, (x, y, w, h) in enumerate(boxes):
            crop = image[y:y+h, x:x+w]
            with cols[i % num_cols]:
                st.image(crop, channels="BGR", caption=f"Crop {i+1} ({w}x{h})")
                
    else:
        st.warning("No regions detected. Try adjusting parameters.")
