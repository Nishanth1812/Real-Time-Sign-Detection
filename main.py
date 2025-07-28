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
MODEL_INPUT_SIZE = 64
PREDICT_INTERVAL = 0.05  # Predict every 50ms

# ==== Load trained model ====
model = load_model("ANOTHER_SOME_CHANGES_NEW_MODEL.keras")
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# ==== Hand detector ====
detector = HandDetector(maxHands=1)

# ==== Preprocess function ====
def preprocess_image(img_array):
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
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
typed_word=""
last_letter=""
letter_start_time=time.time()
word_ready=False
final_word= ""



# ==== Prediction thread ====
def prediction_worker():
    global latest_frame, stable_pred, last_confident_time
    global typed_word,last_letter,letter_start_time
    last_pred_time = 0
    global word_ready, final_word
    
    while True:
        # Ensure frame exists and time interval met
        if latest_frame is not None and (time.time() - last_pred_time >= PREDICT_INTERVAL):
            last_pred_time = time.time()

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
                    # Create white background
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
                        new_w = bg_size 
                        resized = cv2.resize(img_crop, (bg_size, new_h))
                        h_gap = ceil((bg_size - new_h) / 2)
                        w_gap=0
                        bg_img = cv2.copyMakeBorder(
                            resized,
                            top=h_gap, bottom=bg_size - h_gap - new_h,
                            left=w_gap, right=bg_size - w_gap - new_w,
                            borderType=cv2.BORDER_CONSTANT,
                            value=[255, 255, 255]  # white padding
                        )

                    pred, confidence = get_prediction(bg_img)
                else:
                    pred, confidence = get_prediction(frame)
            else:
                stable_pred = ""
                continue

            # Word Prediction 
            if confidence >= 60 and not word_ready:
                print(f"Prediction: {pred}, Confidence: {confidence:.2f}%")
                stable_pred = pred
                last_confident_time = time.time()
                
                if pred!=last_letter:
                    last_letter=pred
                    letter_start_time=time.time()
                else:
                    if time.time()-letter_start_time>1.0:
                        if pred=="space":
                            word_ready=True
                            final_word= typed_word.strip()
                            print("Final Word:",final_word)
                            letter_start_time=time.time()
                            
                        elif pred=="del":
                            typed_word=typed_word[:-1]
                            
                        elif pred not in ['nothing']:
                            typed_word+=pred
                            last_letter=""
                            letter_start_time=time.time()
                        

        # Clear stale prediction after 0.7 seconds
        if time.time() - last_confident_time > 0.7:
            stable_pred = ""

        time.sleep(0.01)  

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

    # Share latest frame
    with lock:
        latest_frame = img.copy()

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-5)
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow FPS

    # Display prediction (cyan color)
    if stable_pred and not word_ready:
        cv2.putText(img, f"Typing: {typed_word}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(img, stable_pred, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
        print("Typing:", typed_word)

    elif word_ready:
        cv2.putText(img, f"Detected Word: {final_word}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        print("Final Word:", final_word)

        

    cv2.imshow("ASL Detection", img)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        typed_word = ""
        final_word = ""
        word_ready = False
        print("Word Reset.")
        
    elif key == ord('n'):  
        typed_word = ""
        final_word = ""
        word_ready = False
        last_letter = ""
        print("Ready for next word...")

stream.release()
cv2.destroyAllWindows()
