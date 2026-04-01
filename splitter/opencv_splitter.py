import cv2

def split_image(image, min_area=4000, kernel_size=(7,7), dilate_iter=2, preprocess_mode="Canny"):
    """
    Splits the image into individual sub-figures using OpenCV contour detection.
    
    Args:
        image: Original BGR image array
        min_area: Minimum area for bounding box to be considered
        kernel_size: Size of structuring element for dilation
        dilate_iter: Number of dilation iterations. Lower means less merging of separate sub-figures.
        preprocess_mode: Strategy for feature extraction ("Canny" or "Threshold Line Art")
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if "Threshold" in preprocess_mode:
        # Ideal for Line Art: Convert to binary where lines are white and background is black
        _, base_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        # Optionally close gaps inside drawings before dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        base_img = cv2.morphologyEx(base_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        base_img = cv2.Canny(blur, 50, 150)

    # Dilate the features. 
    # NOTE: High iterations or large kernel size could combine separate images in the same figure.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(base_img, kernel, iterations=dilate_iter)

    # Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    # Using bounding rect to find valid crops
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > min_area:
            boxes.append((x, y, w, h))

    return boxes
