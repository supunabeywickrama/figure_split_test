import cv2
import numpy as np
import os
from splitter.openai_splitter import split_image_openai, ask_openai_centers, encode_image

def split_image_sam(image, min_area=300):
    try:
        from ultralytics import SAM
    except ImportError:
        print("Ultralytics library not found. Please wait for pip installation.")
        return [], None, []

    h, w = image.shape[:2]

    # Step 1: Use the hyper-stable OpenAI K-Means Voronoi Splitting 
    # to structurally group the detached lines (like the machine base rails) and calculate
    # the exact boundary of the ENTIRE machine block. 
    # This prevents SAM from grabbing only a single tiny sub-plate or segmenting empty space!
    boxes, voronoi_debug, _ = split_image_openai(image, min_area=min_area)
    
    if not boxes:
        return [], None, []

    debug_img = image.copy()
    
    # Optional: Re-fetch centers just to draw them cleanly on our SAM debug image
    b64_image = encode_image(image)
    llm_centers = ask_openai_centers(b64_image)
    if llm_centers:
        for pt in llm_centers:
            cx = max(0, min(w-1, int(pt.get("x", 0) / 1000.0 * w)))
            cy = max(0, min(h-1, int(pt.get("y", 0) / 1000.0 * h)))
            color = (0, 0, 255) if pt.get("is_noise") else (0, 255, 0)
            cv2.circle(debug_img, (cx, cy), 15, color, -1)
            cv2.putText(debug_img, pt.get("label", ""), (cx+20, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Step 2: Boot up Meta's Segment Anything Model!
    model = SAM('mobile_sam.pt')
    
    final_boxes = []
    crops = []
    
    for (x, y, box_w, box_h) in boxes:
        xmin, ymin = x, y
        xmax, ymax = x + box_w, y + box_h
        
        # Pass the Voronoi Bound as a "Bounding Box Prompt" into SAM!
        # SAM will now look at this strict geographical zone and intrinsically find the 
        # LARGEST neural object inside of it, forcing it to trace the ENTIRE machine shell!
        try:
            results = model(image, bboxes=[xmin, ymin, xmax, ymax], retina_masks=True, verbose=False)
        except Exception:
            results = model(image, bboxes=[[xmin, ymin, xmax, ymax]], retina_masks=True, verbose=False)
        
        if not results or not results[0].masks:
            continue
            
        mask_array = results[0].masks.data[0].cpu().numpy()
        
        if mask_array.shape[:2] != (h, w):
            mask_resized = cv2.resize(mask_array, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_array
            
        poly_mask = (mask_resized > 0).astype(np.uint8) * 255
        
        # Fix for Technical Drawings: SAM often only segments the thin black lines,
        # leaving the inside of the machine hollow/checkered. We must "cover the area".
        # 1. Lightly expand lines to fuse them together
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fused_mask = cv2.dilate(poly_mask, kernel, iterations=1)
        
        # 2. Extract ONLY the outermost boundary enclosing the machine
        contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Create a SOLID filled geometry (completely white paper background inside the boundary)
        solid_poly_mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            cv2.drawContours(solid_poly_mask, contours, -1, 255, -1)
        
        # Verify mask isn't empty
        valid_points = np.column_stack(np.where(solid_poly_mask > 0))
        if len(valid_points) == 0: continue
        
        # Neural Transparent RGBA Snippet Extraction
        crop_rgba = np.zeros((box_h, box_w, 4), dtype=np.uint8)
        
        orig_snippet = image[ymin:ymax, xmin:xmax]
        orig_rgba = cv2.cvtColor(orig_snippet, cv2.COLOR_BGR2BGRA)
        
        local_mask = solid_poly_mask[ymin:ymax, xmin:xmax]
        mask_idx = local_mask == 255
        
        crop_rgba[mask_idx] = orig_rgba[mask_idx]
        
        crops.append(crop_rgba)
        final_boxes.append((x, y, box_w, box_h))
        
        # Draw the solid organic wrapping boundary
        if contours:
            cv2.drawContours(debug_img, contours, -1, (255, 0, 255), 3)

    return final_boxes, debug_img, crops
