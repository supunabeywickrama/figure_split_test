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

def get_base_boxes(image, min_area=100, kernel_size=(5,5), dilate_iter=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=dilate_iter)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > min_area:
            boxes.append((x, y, w, h))
    return boxes

def ask_openai_for_bounding_boxes(base64_image):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
        
    client = OpenAI(api_key=api_key)
    
    prompt = (
        "Act as an expert Computer Vision Layout Analysis Agent. Analyze this technical document.\n"
        "It contains several distinct sub-diagrams or illustrations showing mechanical machines and details.\n"
        "Identify the primary visual diagrams. Ignore standalone headers, footers, titles, and pure text paragraphs.\n"
        "Return a JSON array of objects representing the bounding boxes of only the distinct major diagrams. "
        "Include any labels or arrows that clearly belong to the diagram.\n"
        "For each distinct diagram, provide the coordinates normalized between 0 and 1000.\n"
        "Format strictly as:\n"
        "[\n"
        "  {\"ymin\": 100, \"xmin\": 100, \"ymax\": 500, \"xmax\": 500},\n"
        "  ...\n"
        "]\n"
        "IMPORTANT: Do not group two physically separate diagrams (e.g. top and bottom machines) into one box. They MUST be separate bounding boxes.\n"
        "Return ONLY the valid JSON array string. Nothing else."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=600,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "", 1)
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
            
        boxes_json = json.loads(content)
        return boxes_json
    except Exception as e:
        print(f"OpenAI error/parsing error: {e}")
        return []

def intersect_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)

def snap_to_opencv(opencv_boxes, llm_boxes, img_w, img_h):
    cv_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in opencv_boxes]
    final_merged = []
    
    for lbox in llm_boxes:
        ymin = int(lbox.get("ymin", 0) / 1000.0 * img_h)
        xmin = int(lbox.get("xmin", 0) / 1000.0 * img_w)
        ymax = int(lbox.get("ymax", 1000) / 1000.0 * img_h)
        xmax = int(lbox.get("xmax", 1000) / 1000.0 * img_w)
        
        l_rect = [xmin, ymin, xmax, ymax]
        l_area = (xmax - xmin) * (ymax - ymin)
        if l_area == 0: continue
        
        group_cv_boxes = []
        for cvb in cv_boxes:
            cx = (cvb[0] + cvb[2]) / 2
            cy = (cvb[1] + cvb[3]) / 2
            
            i_area = intersect_area(cvb, l_rect)
            cv_area = (cvb[2]-cvb[0]) * (cvb[3]-cvb[1])
            
            if (cv_area > 0 and i_area / cv_area > 0.4) or (xmin - 20 <= cx <= xmax + 20 and ymin - 20 <= cy <= ymax + 20):
                group_cv_boxes.append(cvb)
                
        if group_cv_boxes:
            min_x = min([b[0] for b in group_cv_boxes])
            min_y = min([b[1] for b in group_cv_boxes])
            max_x = max([b[2] for b in group_cv_boxes])
            max_y = max([b[3] for b in group_cv_boxes])
            final_merged.append((min_x, min_y, max_x - min_x, max_y - min_y))
            
    return final_merged

def split_image_openai(image, min_area=100, kernel_size=(5,5), dilate_iter=1):
    h, w = image.shape[:2]
    
    # 1. Ask GPT-4o for conceptual spatial bounding boxes on the raw image
    b64_image = encode_image(image)
    llm_boxes = ask_openai_for_bounding_boxes(b64_image)
    
    # 2. Get pixel-perfect base boxes from OpenCV
    cv_boxes = get_base_boxes(image, min_area, kernel_size, dilate_iter)
    
    # 3. Snap and Merge
    if llm_boxes:
        print(f"OpenAI returned {len(llm_boxes)} bounding boxes.")
        final_boxes = snap_to_opencv(cv_boxes, llm_boxes, w, h)
        
        # Create visual debug image
        debug_img = image.copy()
        
        # Draw LLM "Conceptual" boxes in Light Blue
        for lbox in llm_boxes:
            ymin = int(lbox.get("ymin", 0) / 1000.0 * h)
            xmin = int(lbox.get("xmin", 0) / 1000.0 * w)
            ymax = int(lbox.get("ymax", 1000) / 1000.0 * h)
            xmax = int(lbox.get("xmax", 1000) / 1000.0 * w)
            cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), (255, 150, 0), 2)
            cv2.putText(debug_img, "LLM Core Boundary", (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            
        return final_boxes, debug_img
    else:
        print("OpenAI bounding box prediction failed, returning fallback.")
        return [], None
