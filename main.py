import cv2
import time
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from math import *
from tensorflow.keras.models import load_model


# defining the constants

prev_time=0
curr_time=0
w_cam,h_cam=640,480 # used to define the size of the video capture window
offset=30
img_size=200
letter_count=0

# loading the model and putting labels 
model=load_model("best_model.keras")
class_names=[] 

# Defining functions for preprocessing the image and getting the prediction of the model

def preprocess(img):
    return img

def get_prediction(img):
    pred=model.predict(preprocess(img))
    class_id=np.argmax(pred)
    confidence=np.max(pred)*100
    pred_class=class_names[class_id]
    
    return pred_class,confidence


# Main 

# Setting up the video stream and hand detector

stream=cv2.VideoCapture(0)
stream.set(3,w_cam)
stream.set(4,h_cam)
detector=HandDetector(maxHands=1)

# Main Loop
while True:
    succes,img=stream.read()
    hands,img=detector.findHands(img)
    
    # Using a try except block to avoid errors
    
    try:
        if hands:
            hand=hands[0]   
            x,y,w,h=hand['bbox']
            
            # Cropping the image to get only the bounding box and a little buffer area
            img_crop=img[y-offset:y+h+offset,x-offset:x+w+offset]
            
            # creating a white image to be used as background
            bg_img=np.ones((img_size,img_size,3),np.uint8)*255 
            
            # getting the aspect ratio of the bounding box
            a_ratio=h/w 
            
            # Based on the a_ratio we will overlay the cropped image in the center of the  background image to maintain consistency in size 
            
            if a_ratio>1:
                k=img_size/h
                calc_width=ceil(k*w)
                resized_img=cv2.resize(img_crop,(calc_width,img_size))
                w_gap=ceil((img_size-calc_width)/2)
                bg_img[0:resized_img.shape[0],w_gap:calc_width+w_gap]=resized_img
            else:
                k=img_size/w
                calc_height=ceil(k*w)
                resized_img=cv2.resize(img_crop,(img_size,calc_height))                    
                h_gap=ceil((img_size-calc_height)/2)
                bg_img[h_gap:calc_height+h_gap,0:resized_img.shape[0]]=resized_img 
                
            # Getting the prediction of the model
            
            pred,confidence=get_prediction(bg_img)
            x=""
            if pred:
                letter_count+=1
            
            if letter_count>=50:
                if pred=='blank':
                    cv2.putText(img, f"{x} ({int(confidence)}%)", (x-70, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,150,0), 2)
                    x=""
                else:
                    x=x+pred
                letter_count=0
    except Exception as e:
        pass 
    
    
    # Displaying the fps on screen
    
    curr_time=time.time()
    fps=1/(curr_time-prev_time)
    prev_time=curr_time
    
    cv2.putText(img,str(f"FPS:{int(fps)}"),(20,80),cv2.FONT_HERSHEY_COMPLEX,1,(150,90,0),2) 
    
    # Displaying the final image
    
    cv2.imshow("Image",img)
    cv2.waitkey(1)  
