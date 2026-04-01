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

elif mode in ["OpenAI Smart Split", "SAM"]:
    st.sidebar.subheader("AI Configuration")
    st.sidebar.info("Uses GPT-4o to find Semantic Centers of machines.")
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
        with st.spinner("AI is allocating ink to semantic clusters..."):
            result = split_image_openai(image, min_area=openai_min_area)
            if len(result) == 3:
                boxes, intermediate_img, crops = result
            else:
                boxes, intermediate_img = result
                crops = None
    elif mode == "SAM":
        from splitter.sam_splitter import split_image_sam
        with st.spinner("Meta SAM Neural Segmentation running..."):
            result = split_image_sam(image, min_area=openai_min_area)
            if len(result) == 3:
                boxes, intermediate_img, crops = result
            else:
                boxes, intermediate_img = result
                crops = None
    else:
        boxes = []

    if intermediate_img is not None:
        st.markdown("### Step 1: Semantic Center Extraction by GPT-4o")
        st.image(intermediate_img, channels="BGR", caption="Green=Machine Centers, Red=Noise Centers")

    if boxes:
        if 'crops' in locals() and crops is not None and len(crops) == len(boxes):
            combined = sorted(zip(boxes, crops), key=lambda b: (b[0][1], b[0][0]))
            boxes, crops = zip(*combined)
            boxes = list(boxes)
            crops = list(crops)
        else:
            boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
            crops = None
        
        debug_img = image.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, f"Crop {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        st.subheader("Detected Regions (Final Boundaries)")
        st.image(debug_img, channels="BGR", caption="Detected Regions (Extracted Ink Bounds)")

        st.markdown(f"### Cropped Isolated Images ({len(boxes)} components)")
        
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, (x, y, w, h) in enumerate(boxes):
            if crops is not None and i < len(crops):
                crop = crops[i]
                channels_arg = "BGRA" if len(crop.shape) == 3 and crop.shape[2] == 4 else "BGR"
            else:
                crop = image[y:y+h, x:x+w]
                channels_arg = "BGR"
            with cols[i % num_cols]:
                st.image(crop, channels=channels_arg, caption=f"Organic Extraction {i+1} ({w}x{h})")
                
    else:
        st.warning("No regions detected. Try adjusting parameters.")
