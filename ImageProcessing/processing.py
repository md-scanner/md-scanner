import cv2
import numpy as np
import os

def load_images(path):
    # Load and preprocess images
    images = os.listdir(path)
    images = [f for f in images if f.endswith('.jpg')]

    loaded_images = []
    for img in images:
        image = cv2.imread(os.path.join(path, img))
        loaded_images.append(image)

    return loaded_images

def resize_image(image, scale_percent):
    # Resize the image
    image = cv2.imread(image)
    h, w = image.shape[:2]
    new_width = int(w * scale_percent / 100)
    new_height = int(h * scale_percent / 100)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def apply_gaussian_blur(image):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5,5), 0)
    return blurred_image

def find_contours(image):
    # Find and sort contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours

def transform_perspective(image, contours):
    # Perform perspective transformation
    target = None

    # Loop that extracts the boundary contours of the page
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break
    
    if target is not None:
        target = target.reshape((4, 2))
        approx = np.zeros((4, 2), dtype=np.float32)

        add = target.sum(1)
        approx[0] = target[np.argmin(add)]
        approx[2] = target[np.argmax(add)]

        diff = np.diff(target, axis=1)
        approx[1] = target[np.argmin(diff)]
        approx[3] = target[np.argmax(diff)]

        h, w = image.shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        op = cv2.getPerspectiveTransform(approx, pts)
        dst = cv2.warpPerspective(image, op, (w, h))
        return dst
    else:
        return None
    
def save_warped_image(image, output_path):
    # Save the deblurred image
    cv2.imwrite(output_path, image)

def main():

    input_path = "images/"
    output_path = "output/"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    #images = load_images(input_path)
    images = os.listdir(input_path)
    images = [f for f in images if f.endswith('.jpg')]
    
    for img in images:
        resized_image = resize_image(os.path.join(input_path, img), 50)
        blurred_image = apply_gaussian_blur(resized_image)
        contours = find_contours(blurred_image)
        warped_image = transform_perspective(resized_image, contours)
        
        if warped_image is not None:
            
            save_warped_image(warped_image, os.path.join(output_path, f"warped_{img}"))

if __name__ == "__main__":
    main()