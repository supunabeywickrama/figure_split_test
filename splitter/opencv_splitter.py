import cv2
import numpy as np

def split_image_contour(image, min_area=4000, kernel_size=(7,7), dilate_iter=2, preprocess_mode="Canny"):
    """Contour-based splitting logic (Original)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if "Threshold" in preprocess_mode:
        _, base_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        base_img = cv2.morphologyEx(base_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        base_img = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(base_img, kernel, iterations=dilate_iter)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > min_area:
            boxes.append((x, y, w, h))
    return boxes

def split_image_xycut(image, min_area=4000, min_gap=20, noise_threshold=5, dilate_kernel=5):
    """
    XY-Cut algorithm specifically designed for documents.
    Splits the image by finding horizontal and vertical whitespace.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    if dilate_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel, dilate_kernel))
        binary = cv2.dilate(binary, kernel, iterations=1)
        
    boxes = []
    
    def cut(x, y, w, h):
        if w * h < min_area:
            return
            
        roi = binary[y:y+h, x:x+w]
        
        # Horizontal gaps
        h_proj = np.sum(roi > 0, axis=1)
        h_gaps = []
        in_gap = False
        gap_start = 0
        for i, val in enumerate(h_proj):
            if val <= noise_threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    in_gap = False
                    if (i - gap_start) >= min_gap:
                        h_gaps.append(('h', gap_start, i, i - gap_start))
        if in_gap and (h - gap_start) >= min_gap:
            h_gaps.append(('h', gap_start, h, h - gap_start))
            
        # Vertical gaps
        v_proj = np.sum(roi > 0, axis=0)
        v_gaps = []
        in_gap = False
        gap_start = 0
        for i, val in enumerate(v_proj):
            if val <= noise_threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    in_gap = False
                    if (i - gap_start) >= min_gap:
                        v_gaps.append(('v', gap_start, i, i - gap_start))
        if in_gap and (w - gap_start) >= min_gap:
            v_gaps.append(('v', gap_start, w, w - gap_start))
            
        all_gaps = h_gaps + v_gaps
        if all_gaps:
            # Split at the largest gap found
            all_gaps.sort(key=lambda g: g[3], reverse=True)
            best_gap = all_gaps[0]
            
            gap_type = best_gap[0]
            start = best_gap[1]
            end = best_gap[2]
            mid = (start + end) // 2
            
            if gap_type == 'h':
                cut(x, y, w, mid)
                cut(x, y + mid, w, h - mid)
            else:
                cut(x, y, mid, h)
                cut(x + mid, y, w - mid, h)
            return

        # Tightly bound the region
        rows = np.any(roi > 0, axis=1)
        cols = np.any(roi > 0, axis=0)
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            final_w = cmax - cmin + 1
            final_h = rmax - rmin + 1
            if final_w * final_h >= min_area:
                boxes.append((x + cmin, y + rmin, final_w, final_h))
                
    cut(0, 0, image.shape[1], image.shape[0])
    return boxes
