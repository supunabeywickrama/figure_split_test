import cv2
import numpy as np
import base64
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def ask_openai_centers(base64_image):
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    prompt = (
        "You are an expert layout analyzer. Analyze this technical drawing which contains multiple distinct machine diagrams.\n"
        "Your task: Identify the exact center points of each distinct machine diagram AND any major text blocks (titles, headers, footers).\n"
        "Return a JSON array of objects:\n"
        " - \"x\": normalized x coordinate (0 to 1000)\n"
        " - \"y\": normalized y coordinate (0 to 1000)\n"
        " - \"is_noise\": boolean (true if text block/title/footer, false if actual machine diagram)\n"
        " - \"label\": a 1-3 word description\n"
        "DO NOT group separate machines. There are exactly 5 main machine/detail illustrations. Ensure 'is_noise': false for all 5.\n"
        "DO NOT use markdown."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}]}],
            max_tokens=600, temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"): content = content.replace("```json", "", 1)
        if content.endswith("```"): content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        print(f"OpenAI error: {e}")
        return []

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

def split_image_openai(image, min_area=300):
    if cKDTree is None:
        return [], None, []
        
    h, w = image.shape[:2]
    
    b64_image = encode_image(image)
    llm_centers = ask_openai_centers(b64_image)
    
    if not llm_centers:
        return [], None, []
        
    debug_img = image.copy()
    centers_px = []
    is_noise_list = []
    
    for pt in llm_centers:
        cx = max(0, min(w-1, int(pt.get("x", 0) / 1000.0 * w)))
        cy = max(0, min(h-1, int(pt.get("y", 0) / 1000.0 * h)))
        centers_px.append([cy, cx])
        is_noise_list.append(pt.get("is_noise", False))
        color = (0, 0, 255) if pt.get("is_noise") else (0, 255, 0)
        cv2.circle(debug_img, (cx, cy), 15, color, -1)
        cv2.putText(debug_img, pt.get("label", ""), (cx+20, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 1. Page Border Removal! Removes large vertical bounding lines 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw > 0.8 * w or ch > 0.8 * h:
            cv2.drawContours(binary, [c], -1, 0, -1)
            
    points = np.column_stack(np.where(binary > 0))
    if len(points) == 0 or len(centers_px) < 2:
        return [], None, []
        
    tree = cKDTree(np.array(centers_px))
    _, cluster_labels = tree.query(points)
    
    final_boxes = []
    crops = []
    
    for i in range(len(centers_px)):
        if is_noise_list[i]: continue
            
        cluster_points = points[cluster_labels == i]
        if len(cluster_points) == 0: continue
            
        cluster_mask = np.zeros((h, w), dtype=np.uint8)
        cluster_mask[cluster_points[:, 0], cluster_points[:, 1]] = 255
        
        # Thick dilation to fuse the primary machine structure into one coherent blob
        # and create a smooth, beautiful 30px organic border
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        mask_dilated = cv2.dilate(cluster_mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
            
        # Select the single largest blob! This brilliantly eliminates disconnected text/numbers (like page numbers)
        largest_contour = max(contours, key=cv2.contourArea)
        
        cx, cy, bw, bh = cv2.boundingRect(largest_contour)
        if bw * bh < min_area: continue
            
        # Draw this organic contour to form our perfect clipping mask
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(poly_mask, [largest_contour], -1, 255, -1)
        
        valid_points = np.column_stack(np.where(poly_mask > 0))
        ymin, ymax = valid_points[:, 0].min(), valid_points[:, 0].max()
        xmin, xmax = valid_points[:, 1].min(), valid_points[:, 1].max()
        
        box_w, box_h = xmax - xmin, ymax - ymin
        
        if box_w * box_h >= min_area:
            final_boxes.append((xmin, ymin, box_w, box_h))
            
            # Create a transparent RGBA image
            crop_rgba = np.zeros((box_h, box_w, 4), dtype=np.uint8)
            
            orig_snippet = image[ymin:ymax, xmin:xmax]
            orig_rgba = cv2.cvtColor(orig_snippet, cv2.COLOR_BGR2BGRA)
            
            local_mask = poly_mask[ymin:ymax, xmin:xmax]
            mask_idx = local_mask == 255
            
            crop_rgba[mask_idx] = orig_rgba[mask_idx]
            
            crops.append(crop_rgba)
            
            # Draw organic rounded polygon on debug_img
            cv2.drawContours(debug_img, [largest_contour], -1, (255, 0, 255), 3)

    return final_boxes, debug_img, crops
