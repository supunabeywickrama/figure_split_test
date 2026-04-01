import streamlit as st
import cv2
import numpy as np
import os


st.set_page_config(page_title="Figure Split Tester", layout="wide")

st.title("Figure Split Tester 🔍")

os.makedirs("outputs/crops", exist_ok=True)
os.makedirs("outputs/debug", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

st.sidebar.header("AI Configuration")
st.sidebar.info("This tool exclusively uses the state-of-the-art Meta SAM + GPT-4o Agentic Pipeline.")
sam_min_area = st.sidebar.slider("Noise Filter Min Area", 50, 5000, 300, step=50)

uploaded_file = st.file_uploader("Upload Figure Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Original Image")
    st.image(image, channels="BGR", caption="Original Image")

    intermediate_img = None

    from splitter.sam_splitter import split_image_sam
    with st.spinner("Executing Intelligent GPT-4o + SAM Extraction..."):
        result = split_image_sam(image, min_area=sam_min_area)
        if len(result) == 3:
            boxes, intermediate_img, crops = result
        else:
            boxes, intermediate_img = result
            crops = None

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
