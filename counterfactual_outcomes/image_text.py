import cv2
import numpy as np

def hstack_frames_consolidated(img1, img2, text1, text2):
    
    # --- Step 1: Define Constants and Calculate Required Caption Heights ---
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    padding = 20                 # Vertical padding around text
    text_color = (0, 0, 0)       # Black text (BGR)
    bg_color = (255, 255, 255)   # White background (BGR)

    # Calculate required height for text 1
    (text_w1, text_h1), baseline1 = cv2.getTextSize(text1, font, font_scale, font_thickness)
    caption_height1 = text_h1 + baseline1 + (padding * 2)
    
    # Calculate required height for text 2
    (text_w2, text_h2), baseline2 = cv2.getTextSize(text2, font, font_scale, font_thickness)
    caption_height2 = text_h2 + baseline2 + (padding * 2)
    
    # Determine the maximum height to ensure clean concatenation
    max_caption_height = max(caption_height1, caption_height2)

    # --- Step 2: Handle Original Image Height Mismatch ---
    
    if img1.shape[0] != img2.shape[0]:
        # Resize to the smaller height, maintaining aspect ratio
        min_original_height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * min_original_height / img1.shape[0]), min_original_height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * min_original_height / img2.shape[0]), min_original_height))

    # --- Step 3: Process and Caption Images (Using a local function for clean repetition) ---
    
    def caption_and_pad(img, text, text_w, text_h, baseline):
        img_h, img_w = img.shape[:2]
        
        # 3a. Create the border (padding the bottom to max_caption_height)
        img_with_border = cv2.copyMakeBorder(
            img, 
            0,                     # top
            max_caption_height,    # bottom (uses the calculated max height)
            0, 
            0, 
            cv2.BORDER_CONSTANT, 
            value=bg_color
        )
        
        # 3b. Calculate coordinates to center text in the new white bar
        
        # X: Center text horizontally
        text_x = (img_w - text_w) // 2
        
        # Y: Center text vertically within the white bar
        # Start at original image height (img_h) + half the bar height + half the text height - baseline offset
        text_y = img_h + (max_caption_height // 2) + (text_h // 2) - baseline
        
        # 3c. Draw the text
        cv2.putText(
            img_with_border, 
            text, 
            (text_x, text_y), 
            font, 
            font_scale, 
            text_color, 
            font_thickness, 
            cv2.LINE_AA
        )
        return img_with_border

    # Process both images using the local function
    final_img1 = caption_and_pad(img1, text1, text_w1, text_h1, baseline1)
    final_img2 = caption_and_pad(img2, text2, text_w2, text_h2, baseline2)

    # --- Step 4: Horizontal Concatenation ---
    
    # Since final_img1 and final_img2 have the same total height, hconcat works perfectly.
    combined_img = cv2.hconcat([final_img1, final_img2])
    
    return combined_img
