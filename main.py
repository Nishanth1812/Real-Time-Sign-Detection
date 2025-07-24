import cv2
import time
import numpy as np
from math import ceil
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Constants
w_cam, h_cam = 640, 480
offset = 30
img_size = 200

# Load model and class labels
model = load_model("asl_alphabet_final_model.keras")
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Preprocess image before prediction
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=-1)  # shape: (32, 32, 1)
    img = img.astype("float32") / 255.0
    img_array = np.expand_dims(img, axis=0)  # shape: (1, 32, 32, 1)
    return img_array

# Get prediction from model
def get_prediction(img):
    processed = preprocess_image(img)
    pred = model.predict(processed, verbose=0)
    class_id = np.argmax(pred)
    confidence = np.max(pred) * 100
    print(class_names[class_id])
    return class_names[class_id], confidence

# Initialize video stream and hand detector
stream = cv2.VideoCapture(0)
stream.set(3, w_cam)
stream.set(4, h_cam)
detector = HandDetector(maxHands=1)

# For stabilizing predictions
prev_pred = ""
same_pred_count = 0
stable_pred = ""
prev_time = 0

while True:
    success, img = stream.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Crop and center hand in white 200x200 background
            img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            bg_img = np.ones((img_size, img_size, 3), np.uint8) * 255
            a_ratio = h / w

            if a_ratio > 1:
                k = img_size / h
                calc_width = ceil(k * w)
                resized_img = cv2.resize(img_crop, (calc_width, img_size))
                w_gap = ceil((img_size - calc_width) / 2)
                bg_img[:, w_gap:calc_width + w_gap] = resized_img
            else:
                k = img_size / w
                calc_height = ceil(k * h)
                resized_img = cv2.resize(img_crop, (img_size, calc_height))
                h_gap = ceil((img_size - calc_height) / 2)
                bg_img[h_gap:calc_height + h_gap, :] = resized_img

            # Predict
            pred, confidence = get_prediction(bg_img)

            if confidence > 70:
                if pred == prev_pred:
                    same_pred_count += 1
                else:
                    same_pred_count = 0
                    prev_pred = pred

                if same_pred_count >= 10:
                    stable_pred = pred
                    same_pred_count = 0

            # Show prediction near hand
            if stable_pred:
                cv2.putText(img, f"{stable_pred}", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 150, 0), 3)

    except Exception as e:
        pass

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-5)
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    # Display image
    cv2.imshow("ASL Detection", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Optional: Slow down to improve stability
    time.sleep(0.03)

# Cleanup
stream.release()
cv2.destroyAllWindows()
