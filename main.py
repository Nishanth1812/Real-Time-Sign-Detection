import cv2
import time
import numpy as np
from math import ceil
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import threading

# ==== Configurations ====
w_cam, h_cam = 640, 480
offset = 30
bg_size = 200
MODEL_INPUT_SIZE = 32
PREDICT_EVERY_N_FRAMES = 5  # Predict every 5 frames

# ==== Load trained model ====
model = load_model("NEW_MODEL_EXTENDED.keras")
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# ==== Hand detector ====
detector = HandDetector(maxHands=1)

# ==== Preprocess function ====
def preprocess_image(img_array):
    """Convert to grayscale, resize, normalize, and add dimensions."""
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # shape (32, 32, 1)
    img_array = np.expand_dims(img_array, axis=0)   # shape (1, 32, 32, 1)
    return img_array

# ==== Prediction function ====
def get_prediction(img_array):
    processed = preprocess_image(img_array)
    pred = model.predict(processed, verbose=0)
    class_id = np.argmax(pred)
    confidence = np.max(pred) * 100
    return class_names[class_id], confidence

# ==== Shared state ====
latest_frame = None
stable_pred = ""
lock = threading.Lock()
last_confident_time = time.time()

# ==== Prediction thread ====
def prediction_worker():
    global latest_frame, stable_pred, last_confident_time
    frame_counter = 0

    while True:
        frame_counter += 1

        # Predict every N frames
        if frame_counter % PREDICT_EVERY_N_FRAMES == 0 and latest_frame is not None:
            with lock:
                frame = latest_frame.copy()

            # Detect hand
            hands, _ = detector.findHands(frame, draw=False)

            if hands:
                # Crop hand region
                x, y, w, h = hands[0]['bbox']
                y1, y2 = max(0, y - offset), min(h_cam, y + h + offset)
                x1, x2 = max(0, x - offset), min(w_cam, x + w + offset)
                img_crop = frame[y1:y2, x1:x2]

                if img_crop.size != 0:
                    # Create white background (centered hand)
                    bg_img = np.ones((bg_size, bg_size, 3), np.uint8) * 255
                    aspect_ratio = h / w

                    if aspect_ratio > 1:
                        k = bg_size / h
                        new_w = ceil(k * w)
                        resized = cv2.resize(img_crop, (new_w, bg_size))
                        w_gap = ceil((bg_size - new_w) / 2)
                        bg_img[:, w_gap:w_gap + new_w] = resized
                    else:
                        k = bg_size / w
                        new_h = ceil(k * h)
                        resized = cv2.resize(img_crop, (bg_size, new_h))
                        h_gap = ceil((bg_size - new_h) / 2)
                        bg_img[h_gap:h_gap + new_h, :] = resized

                    pred, confidence = get_prediction(bg_img)
                else:
                    pred, confidence = get_prediction(frame)
            else:
                # No hand detected → clear prediction
                stable_pred = ""
                continue


            # Update stable prediction if confidence ≥ 50%
            if confidence >= 50:
                print(f"Prediction: {pred}, Confidence: {confidence:.2f}%")
                stable_pred = pred
                last_confident_time = time.time()

        # Clear prediction if no confident prediction in 0.5 sec
        if time.time() - last_confident_time > 0.5:
            stable_pred = ""

        time.sleep(0.01)  # Prevent CPU overuse

# ==== Start prediction thread ====
threading.Thread(target=prediction_worker, daemon=True).start()

# ==== Camera stream ====
stream = cv2.VideoCapture(0)
stream.set(3, w_cam)
stream.set(4, h_cam)

prev_time = 0

while True:
    success, img = stream.read()
    if not success:
        break

    # Share latest frame with prediction thread
    with lock:
        latest_frame = img.copy()

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-5)
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    # Display stable prediction
    if stable_pred:
        cv2.putText(img, stable_pred, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 150, 0), 4)

    cv2.imshow("ASL Detection", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
