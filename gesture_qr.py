#using this for webcam help: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

#used this for hand detection: https://www.youtube.com/watch?v=RRBXVu5UE-U
#                              https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

#ACTING AS THE WEBCAMS FROM NORO SCREEN
import cv2 as cv

import mediapipe as mp #hand detection by Google
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision import HandLandmarkerResult
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
# from mediapipe.tasks.python.core import BaseOptions
from mediapipe.tasks.python.core import base_options

import time #to make QR code popup temporary
 

#init webcam class
webcam = cv.VideoCapture(1) #'1'= index of cameras -- this case, my default computer camera. 
#don't know how to make it be multiple cameras in the case for Noro (3 cameras)

#not important: checking if camera is opened
if not webcam.isOpened(): #if camera is not opened --> exit
    print("Cannot open camera")
    exit()

#defining QR code things
qrcode_is_shown = False #starts as qrcode hidden
qrcode_shown_start_time = 0 #starts at 0 seconds. going to last for 15 seconds (can be changed obviously)


#initializing hand detection things
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils #to see the red/green dots/lines on hand later on

# BaseOptions = mp.tasks.BaseOptions
BaseOptions = base_options.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

#init Hand class + params
# hand = mp_hands.HandLandmarker(
#     max_num_hands = 1, #1 hand here since the hand detection is only for 1 index finger up
#     min_detection_confidence=0.7, #random value. dont know what it does exactly
#     min_tracking_confidence=0.5 #random value. dont know what it does exactly
# )



# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):

    #actual hand detection process
    if result.hand_landmarks: #if at least 1 hand is detected:
        for hand_landmarks in result.hand_landmarks:
            # print(hand_landmarks) #prints coordinates of each landmark on the hand
            #print(result.multi_hand_landmarks)
            # mp_drawing.draw_landmarks(new_RBG_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #draws the green/red dots/lines on the hand
    
            #checking if index finger is up
            if hand_landmarks[8].y < hand_landmarks[5].y: #[8] is the index finger tip, 
                #[5] is the index finger bottom. #also, point 0,0 is top left corner so thats why '<' and not '>'
                
                #if index tip is higher than bottom, then index finger is up
                print("Index finger is up")

                '''DO QR CODE POPUP HERE'''


#params for hand detection model
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'), #hand detection model in root dir
    running_mode=VisionRunningMode.LIVE_STREAM, #using live stream mode since its going to be real time feed during calls
    result_callback=print_result,
    num_hands=50) #50 hands max scannable (can be changed). thinking of it like in a conference room. Up to 25 people

#actually init hand detection model
with HandLandmarker.create_from_options(options) as landmarker:
    #MAIN CAMERA LOOP
    while True: #camera on until user presses 'q'
        ret, frame = webcam.read() #ret: boolean value(True if camera gives a frame, False if not)
        #frame: the actual individual frame from the video that were seeing ?
    
        if not ret: #if camera didn't give a frame --> exit
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #opencv uses BGR, mediapipe uses RGB. fixing order from BGR to RGB:
        new_RBG_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #convert opencv image to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=new_RBG_frame) #doing this bc mediapipe doesn't use numpy frames like opencv does
            #mp.ImageFormat.SRGB = 'RBG' simply

        #real time feed so we need to give the timestamp of each individual frame since were using 'live stream' mode
        timestamp_ms = int(time.time() * 1000) #timestamp in milliseconds
        landmarker.detect_async(mp_image, timestamp_ms) # processes and detects hands in video frame


        #actual hand detection process
        # result = hand.detect_async(new_RBG_frame)
        # if result.multi_hand_landmarks: #if at least 1 hand is detected:
        #     for hand_landmarks in result.multi_hand_landmarks:
        #         # print(hand_landmarks) #prints coordinates of each landmark on the hand
        #         #print(result.multi_hand_landmarks)
        #         mp_drawing.draw_landmarks(new_RBG_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #draws the green/red dots/lines on the hand
        
        #         #checking if index finger is up
        #         if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y: #[8] is the index finger tip, 
        #             #[5] is the index finger bottom. #also, point 0,0 is top left corner so thats why '<' and not '>'
                    
        #             #if index tip is higher than bottom, then index finger is up
        #             print("Index finger is up")



        #displaying frames in a window
        cv.imshow('frame', new_RBG_frame) #displays frames in a window(thats what imshow does: opens a new window)
        if cv.waitKey(1) == ord('q'): #exit if user presses 'q'
            break

 
webcam.release()
cv.destroyAllWindows()






