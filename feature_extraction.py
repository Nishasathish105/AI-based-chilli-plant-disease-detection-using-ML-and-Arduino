import cv2
import numpy as np

def extract_features(image_path):

    # Read image
    img = cv2.imread(image_path)

    # Resize
    img = cv2.resize(img, (128,128))

    # -------- IMAGE CLEANING --------
    # Denoise
    img = cv2.GaussianBlur(img,(5,5),0)

    # Convert to LAB for contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    # CLAHE (improves clarity)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen image
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img,-1,kernel)

    # -------- FEATURE EXTRACTION --------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_color = cv2.mean(hsv)[:3]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    edge_density = np.sum(edges)/(128*128)

    features = list(mean_color)+[edge_density]

    return np.array(features)
