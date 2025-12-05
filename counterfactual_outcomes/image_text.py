import cv2
import numpy as np

def add_description_below(image_path, text, output_path):
    # 1. Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not load image.")
        return

    # 2. Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0)       # Black text (BGR)
    bg_color = (255, 255, 255)   # White background (BGR)
    padding = 20                 # Extra space around text

    # 3. Calculate text size to determine border height
    # getTextSize returns ((width, height), baseline)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # 4. create the bottom border
    # The new section needs to be tall enough for the text + padding
    new_height = text_height + baseline + (padding * 2)
    
    # copyMakeBorder arguments: src, top, bottom, left, right, borderType, value
    img_with_border = cv2.copyMakeBorder(
        img, 
        0,              # Top
        new_height,     # Bottom (add space here)
        0,              # Left
        0,              # Right
        cv2.BORDER_CONSTANT, 
        value=bg_color
    )

    # 5. Calculate X and Y coordinates to center the text
    # X: Center text horizontally
    image_width = img.shape[1]
    text_x = (image_width - text_width) // 2
    
    # Y: Position text in the new white space
    # Start at original height + padding + text height
    text_y = img.shape[0] + padding + text_height

    # 6. Draw the text
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

    # 7. Save or Display
    cv2.imwrite(output_path, img_with_border)
    print(f"Image saved to {output_path}")
    
    # Optional: Display window
    # cv2.imshow("Result", img_with_border)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# --- Usage ---
add_description_below('input_image.jpg', 'Figure 1: Analysis Result', 'output_result.jpg')
