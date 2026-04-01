import streamlit as st
import cv2
import numpy as np
import os
from splitter.opencv_splitter import split_image

st.set_page_config(page_title="Figure Split Tester", layout="wide")

st.title("Figure Split Tester 🔍")

# Let's create output directories on startup if they don't exist
os.makedirs("outputs/crops", exist_ok=True)
os.makedirs("outputs/debug", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

st.sidebar.header("Testing Parameters")
mode = st.sidebar.selectbox("Select Method", ["OpenCV", "SAM"])

if mode == "OpenCV":
    st.sidebar.subheader("OpenCV Configuration")
    preprocess_mode = st.sidebar.radio("Preprocessing Mode", ["Threshold Line Art (White Backgrounds)", "Canny Edge (Photos/Complex)"])
    st.sidebar.info("Tip: Use 'Threshold Line Art' for technical drawings to group separate figures better.")
    min_area = st.sidebar.slider("Min Area (pixels)", 500, 200000, 4000, step=500)
    dilate_iter = st.sidebar.slider("Dilation Iterations", 0, 10, 2)
    kernel_size_val = st.sidebar.slider("Kernel Size (odd number)", 3, 101, 21, step=2)
    kernel_size = (kernel_size_val, kernel_size_val)

uploaded_file = st.file_uploader("Upload Figure Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Original Image")
    st.image(image, channels="BGR", caption="Original Image", use_container_width=True)

    if mode == "OpenCV":
        boxes = split_image(image, min_area=min_area, kernel_size=kernel_size, dilate_iter=dilate_iter, preprocess_mode=preprocess_mode)
    else:
        st.info("SAM splitting not yet implemented.")
        boxes = []

    if boxes:
        # Sort boxes to show them in somewhat logical order (top-to-bottom, left-to-right)
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        # Draw bounding boxes
        debug_img = image.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, f"Crop {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        st.subheader("Detected Regions")
        st.image(debug_img, channels="BGR", caption="Detected Regions", use_container_width=True)

        st.markdown(f"### Cropped Images ({len(boxes)} detected)")
        
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, (x, y, w, h) in enumerate(boxes):
            crop = image[y:y+h, x:x+w]
            with cols[i % num_cols]:
                st.image(crop, channels="BGR", caption=f"Crop {i+1} ({w}x{h})", use_container_width=True)
                
    else:
        st.warning("No regions detected. Try adjusting parameters.")
