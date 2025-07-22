#using: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
import cv2 as cv
 
webcam = cv.VideoCapture(0) #0 argument= my default computer camera. don't know how to make it be multiple cameras in the
#case for Noro (3 cameras)
if not webcam.isOpened(): #if camera is not opened --> exit
    print("Cannot open camera")
    exit()
while True: #camera on until user presses 'q'
    ret, frame = webcam.read() #ret: boolean value(True if camera gave a frame, False if not)
    #frame: the actual individual frame from the video that were seeing ?
 
    if not ret: #if camera didn't give a frame --> exit
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('frame', frame) #displays frames in a window(thats what imshow does: opens a new window)
    if cv.waitKey(1) == ord('q'): #exit if user presses 'q'
        break
 
webcam.release() 
cv.destroyAllWindows()

