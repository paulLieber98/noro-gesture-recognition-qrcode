#using this for webcam help: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

#used this for hand detection: https://www.youtube.com/watch?v=RRBXVu5UE-U
#                              https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

#ACTING AS THE WEBCAMS FROM NORO SCREEN
import cv2 as cv
import mediapipe as mp #hand detection by Google
 
webcam = cv.VideoCapture(1) #'1'= index of cameras -- this case, my default computer camera. 
#don't know how to make it be multiple cameras in the case for Noro (3 cameras)

#initializing hand detection things
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils #to see the red/green dots/lines on hand later on

#init Hand class + params
hand = mp_hands.Hands(
    max_num_hands = 1, #1 hand here since the hand detection is only for 1 index finger up
    min_detection_confidence=0.7, #random value. dont know what it does exactly
    min_tracking_confidence=0.5 #random value. dont know what it does exactly
)

if not webcam.isOpened(): #if camera is not opened --> exit
    print("Cannot open camera")
    exit()

#main video loop
while True: #camera on until user presses 'q'
    ret, frame = webcam.read() #ret: boolean value(True if camera gives a frame, False if not)
    #frame: the actual individual frame from the video that were seeing ?
 
    if not ret: #if camera didn't give a frame --> exit
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #opencv uses BGR, mediapipe uses RGB. fixing order from BGR to RGB:
    new_RBG_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hand.process(new_RBG_frame)
    if result.multi_hand_landmarks: #if multiple hands are detected
        for hand_landmarks in result.multi_hand_landmarks:
            print(hand_landmarks) #prints coordinates of each landmark on the hand
            mp_drawing.draw_landmarks(new_RBG_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #draws the green/red dots/lines on the hand

    cv.imshow('frame', new_RBG_frame) #displays frames in a window(thats what imshow does: opens a new window)
    if cv.waitKey(1) == ord('q'): #exit if user presses 'q'
        break
 
webcam.release() 
cv.destroyAllWindows()






