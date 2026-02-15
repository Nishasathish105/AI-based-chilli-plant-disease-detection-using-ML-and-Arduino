import cv2
import joblib
import numpy as np
from feature_extraction import extract_features

model = joblib.load("best_model.pkl")
cap = cv2.VideoCapture(0)

print("Press C to capture plant")
print("ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # green plant mask
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # clean mask
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # biggest green region = whole plant
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
        plant_region = frame[y:y+h, x:x+w]

    cv2.imshow("Whole Plant Detection", display)

    key = cv2.waitKey(1)

    if key == ord('c') and contours:
        cv2.imwrite("plant.jpg", plant_region)

        features = extract_features("plant.jpg")
        prediction = model.predict([features])[0]

        if prediction == 0:
            print("🌿 HEALTHY PLANT")
        else:
            print("🍂 DISEASED PLANT")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
