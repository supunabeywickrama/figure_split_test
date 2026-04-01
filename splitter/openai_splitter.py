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
            
    # Sort boxes top-to-bottom, left-to-right approximately
    boxes = sorted(boxes, key=lambda b: (b[1] // 50, b[0]))
    return boxes

def draw_numbered_boxes(image, boxes):
    debug_img = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(debug_img, (x, y-th-4), (x+tw+2, y+2), (0,0,0), -1)
        cv2.putText(debug_img, text, (x+1, y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    return debug_img

def ask_openai_to_group(base64_image):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
        
    client = OpenAI(api_key=api_key)
    
    prompt = (
        "This is an image of a technical drawing with several distinct sub-diagrams. "
        "I have overlaid red bounding boxes, each labeled with a bright yellow ID number on a black background. "
        "Your task is to identify the distinct major sub-diagrams and group the IDs of the bounding boxes that belong to each one. "
        "Rules:\n"
        "1. Texts, labels, and arrows that point to a specific sub-diagram should be grouped with it.\n"
        "2. The final output must be ONLY a valid JSON array of arrays of integers representing the grouped IDs.\n"
        "3. Ensure all relevant IDs are assigned to a group.\n"
        "4. Do not output any markdown formatting like ```json or any explanations, JUST the correct JSON array string.\n"
        "Example output: [[0, 1, 2], [3], [4, 5, 6]]"
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
            max_tokens=300,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        # Sanitizing markdown if it returns it
        if content.startswith("```json"):
            content = content.replace("```json", "", 1)
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
            
        groups = json.loads(content)
        return groups
    except Exception as e:
        print(f"OpenAI error/parsing error: {e}")
        return []

def merge_grouped_boxes(boxes, groups):
    merged_boxes = []
    for group in groups:
        if not group: continue
        group_boxes = [boxes[idx] for idx in group if idx < len(boxes)]
        if not group_boxes: continue
        
        min_x = min([b[0] for b in group_boxes])
        min_y = min([b[1] for b in group_boxes])
        max_x = max([b[0]+b[2] for b in group_boxes])
        max_y = max([b[1]+b[3] for b in group_boxes])
        
        merged_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
    return merged_boxes

def split_image_openai(image, min_area=100, kernel_size=(5,5), dilate_iter=1):
    # 1. Get initial components
    boxes = get_base_boxes(image, min_area, kernel_size, dilate_iter)
    if not boxes:
        return [], None
        
    # 2. Draw IDs
    debug_intermediate = draw_numbered_boxes(image, boxes)
    
    # 3. Call OpenAI for semantic grouping
    b64_image = encode_image(debug_intermediate)
    groups = ask_openai_to_group(b64_image)
    
    # 4. Merge grouped boxes
    if groups:
        print(f"OpenAI successfully grouped into {len(groups)} regions.")
        final_boxes = merge_grouped_boxes(boxes, groups)
    else:
        print("OpenAI grouping failed, returning intermediate boxes fallback.")
        final_boxes = boxes 
        
    return final_boxes, debug_intermediate
