#using this for webcam help: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

#used this for hand detection: https://www.youtube.com/watch?v=RRBXVu5UE-U
#                              https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

#used this for segmentation model: https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter

#other: https://www.geeksforgeeks.org/python/python-opencv-cv2-cvtcolor-method/

#ACTING AS THE WEBCAMS FROM NORO SCREEN
import cv2 as cv

import mediapipe as mp #models from Google

#hand detection model
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision import HandLandmarkerResult
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
# from mediapipe.tasks.python.core import BaseOptions
from mediapipe.tasks.python.core import base_options

#for segmentation model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List
from mediapipe import Image
from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions

import time #to make QR code popup temporary

import qrcode

import numpy as np
 

#INIT WEBCAM CLASS
webcam = cv.VideoCapture(1)  #'1'= index of cameras -- this case, my default computer web-camera. 


#--------------------------------
#DEFINING QR CODE THINGS
#--------------------------------
qrcode_is_shown = False #starts as qrcode hidden
qrcode_shown_start_time = 0 #starts at 0 seconds. going to last for 15 seconds (can be changed obviously)
should_show_qrcode = False #starts as False. going to be True if index finger is up

#GENERATE QR CODE
def generate_qr():
    qr = qrcode.make('https://www.noro.co/') #replace link with whatever the actual link is for the qrcode-Noro-screen remote
    qr_converted = qr.convert('RGB') 
    qr_array = cv.cvtColor(np.array(qr_converted), cv.COLOR_RGB2BGR) #convert to opencv image: takes only numpy array + BGR
    return qr_array


#--------------------------------
#DEFINING HAND DETECTION MODEL
#--------------------------------
BaseOptions = base_options.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# HAND DETECTION CALLBACK
def print_result_hand_detection(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #actual hand detection process
    if result.hand_landmarks: #if at least 1 hand is detected:
        for hand_landmarks in result.hand_landmarks:
            finger_threshold = 0.05 # increase to make detection stricter
            if (
                hand_landmarks[8].y < hand_landmarks[5].y - finger_threshold and 
                hand_landmarks[12].y > hand_landmarks[9].y + finger_threshold and 
                hand_landmarks[16].y > hand_landmarks[13].y + finger_threshold and 
                hand_landmarks[20].y > hand_landmarks[17].y + finger_threshold): 

                global should_show_qrcode, qrcode_is_shown, qrcode_shown_start_time
                if qrcode_is_shown == False: #if qrcode is not shown, then show it
                    qrcode_is_shown = True
                    qrcode_shown_start_time = time.time()
                    should_show_qrcode = True



#PARAMS FOR HAND DETECTION MODEL
options_hand_detection = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'), #hand detection model in root dir
    running_mode=VisionRunningMode.LIVE_STREAM, #using live stream mode since its going to be real time feed during calls
    result_callback=print_result_hand_detection,
    num_hands=50) #50 hands max scannable (can be changed). thinking of it like in a conference room. Up to 25 people


#--------------------------------
#DEFINING SEGMENTATION MODEL
#--------------------------------
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# SEGMENTATION CALLBACK
# def print_result_segmentation(result: List[Image], output_image: Image, timestamp_ms: int):
#     # print(f'result FROM SEGMENTATION: {result}')
#     global last_person_mask, last_person_prob
#     if result.category_mask: #if there is a person in the frame (category mask tells us if there is a person: person vs background)
#         mask = result.category_mask.numpy_view() #converting mediapipe image to numpy array for Opencv
#         last_person_mask = mask #updating the most recent person mask
def print_result_segmentation(result: List[Image], output_image: Image, timestamp_ms: int):
    global last_person_mask, last_person_prob
    #category (argmax) mask → one integer per pixel:
    #   0 = background, 1 = person
    if result.category_mask:
        last_person_mask = result.category_mask.numpy_view()

    # Confidence masks → per-class probabilities (float in [0, 1]) for each pixel.
    # Selfie Segmenter usually returns 2 channels: [background_prob, person_prob].
    # We prefer the person probability channel (index 1) when available.
    if getattr(result, "confidence_masks", None):
        cms = result.confidence_masks
        idx = 1 if len(cms) > 1 else 0  # pick person channel if present, else fallback
        last_person_prob = cms[idx].numpy_view()  # float mask: person probability per pixel


#PARAMS FOR SEGMENTATION MODEL
options_segmentation = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='selfie_segmenter.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result_segmentation,
    output_category_mask=True,
    output_confidence_masks=True
)


#--------------------------------
#DEFINING OBJECT DETECTION MODEL (solely for putting rectangles around people. not using anymore bc now using segmentation model for drawing boxes)
#--------------------------------
BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.Detection
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode.LIVE_STREAM

# OBJECT DETECTION CALLBACK
def print_result_object_detection(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    global detected_people
    detected_people = []
    if result.detections:
        for detection in result.detections:
            # Check if the detected object is a person
            if detection.categories[0].category_name.lower() == 'person':
                # Get bounding box coordinates
                bbox = detection.bounding_box
                detected_people.append({
                    'x': bbox.origin_x,
                    'y': bbox.origin_y,
                    'width': bbox.width,
                    'height': bbox.height,
                    'confidence': detection.categories[0].score
                })

#PARAMS FOR OBJECT DETECTION MODEL
options_object_detection = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result_object_detection)

#GLOBAL VARS
last_person_mask = None #most recent segmented person mask. used later on
binary_mask = None 
last_person_prob = None


#--------------------------------
# MAIN LOOP
#--------------------------------
with HandLandmarker.create_from_options(options_hand_detection) as handmodel: #hand detection model
    with ImageSegmenter.create_from_options(options_segmentation) as segmenter: #segmentation model
        with ObjectDetector.create_from_options(options_object_detection) as objectmodel: #object detection model (only for rectangle outlines)

            #MAIN CAMERA LOOP
            while True: #camera on until user presses 'q'
                ret, frame = webcam.read() #ret: boolean value(True if camera gives a frame, False if not)
                # frame: the actual individual frame from the video that were seeing (opencv image)
                if not ret: #if camera didn't give a frame --> exit
                    print("Can't receive frame (stream end?). Exiting ...")
                    break


                #DRAW BOXES AROUND PEOPLE (OBJECT DETECTION MODEL)
                detected_people = [] # Store detected people globally
                for person in detected_people:
                    # Draw red rectangle around person
                    cv.rectangle(frame, 
                            (person['x'], person['y']), 
                            (person['x'] + person['width'], person['y'] + person['height']), 
                            (0, 0, 255), 2)  #red color, thickness 2


                # opencv uses BGR, mediapipe uses RGB. fixing order from BGR to RGB:
                new_RBG_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                #convert opencv image to mediapipe image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=new_RBG_frame) #doing this bc mediapipe doesn't use numpy frames like opencv does
                          #mp.ImageFormat.SRGB is the same thing as 'RBG' simply

                zoom_view = None #will be used to hold the zoomed crop we show in a seperate window
                
                if last_person_mask is not None: #if person is detected in most recent frame
                    # print('Person detected by segmentation model')

                    # Prefer the probability mask (smoother, less noisy). If it's not available yet,
                    # fall back to the category/label mask so the app still works.
                    if last_person_prob is not None:
                        src_mask = last_person_prob
                    else:
                        src_mask = last_person_mask


                    #binary mask: CONVERTS INTO BLACK AND WHITE IMAGE FOR OPENCV (255 = white, 0 = black)
                    #if a value is a float in between 0 and 1, threshold at 0.5
                    #if an integer of either {0,1} or {0,255}, any non-zero is a person (duh)
                    if np.issubdtype(src_mask.dtype, np.floating):
                        binary_mask = (src_mask > 0.5).astype(np.uint8) * 255
                    else:
                        binary_mask = (src_mask > 0).astype(np.uint8) * 255



                    # #RESIZING MASK TO MATCH WEBCAM FRAME SIZE
                    # resized_mask = cv.resize(last_person_mask, (frame.shape[1], frame.shape[0]))

                    # # mask_u8 = resized_mask.astype(np.uint8)

                    # # #CONVERTING MASK TO BINARY MASK (people = 255 pixels, background = 0)
                    # binary_mask = (mask_u8  > 0).astype(np.uint8) * 255 #255 to scale up every non-zero number to 255 for full distinction
                    # # #resized_mask == 1: if a pixel is equal to 1 (which means its a person), then its a person. period.
                    # # #.astype(np.uint8) to convert numbers to 8-bit integer(only non-negative integers and 0's.) Opencv expects integers
                    


                    #clean up mask to remove noise and close small holes/gaps
                    kernel = np.ones((3, 3), np.uint8) # 3x3 square structuring element
                    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel, iterations=1) # remove small noise
                    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel, iterations=2) # fill small holes/gaps

                    # Connected components → group white pixels into blobs (people candidates).
                    # Returns:
                    # - num_labels: number of blobs (incl. background)
                    # - labels: image assigning a blob id to each pixel
                    # - stats: per-blob stats: [x, y, w, h, area]
                    # - _: centroids (unused)
                    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_mask, connectivity = 8) #EXPLANATION

                    people_boxes = []
                    H, W = frame.shape[:2]
                    #far back people still pass
                    min_area_px = max(150, int(0.0003 * W * H)) #0.03% of frame . floor at 150 px
                    
                    #EXPLAINNNN!!!
                    for i in range(1, num_labels): # 0 is background 
                        x = int(stats[i, cv.CC_STAT_LEFT])
                        y = int(stats[i, cv.CC_STAT_TOP])
                        w = int(stats[i, cv.CC_STAT_WIDTH])
                        h = int(stats[i, cv.CC_STAT_HEIGHT])
                        area = int(stats[i, cv.CC_STAT_AREA])

                        #filter noise + the 'whole frame' scan failure case
                        if area < min_area_px:
                            continue

                        #dropping components that make the box fit the entire frame (dont wnat that)
                        # Guard against the classic failure where the mask is almost the whole frame.
                        # If a component touches both left+right edges or top+bottom edges, or is >90% of frame area,
                        # we skip it so we don't draw a full-frame rectangle.
                        touch_left = x<=2
                        touch_top = y <=2 
                        touch_right = (x+w) >= (W - 3)
                        touch_bottom = (y+h) >= (H - 3)
                        if (touch_left and touch_right) or (touch_top and touch_bottom):
                            continue
                        if area > 0.90 * ( W * H):
                            continue

                        #to include hands slightly outside the body mask/box
                        # Add 5% padding so hands slightly outside the person silhouette are still inside the crop.
                        hands_slight_outside = int(0.05 * max(w, h))
                        x1 = max(0, x - hands_slight_outside)
                        y1 = max(0, y - hands_slight_outside)
                        x2 = min(W, x + w + hands_slight_outside)
                        y2 = min(H, y + h + hands_slight_outside)

                        people_boxes.append({'x': x1,
                                            'y': y1,
                                            'width': x2- x1,
                                            'height': y2- y1})

                    # #STORING CONTOURS OF PEOPLE IN VARIABLE (edges of people)
                    # contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    # #finding contours (edges) of people with .findContours() bc of our binary mask 0-255 earlier
                    # #ignoring hierarchy(second value) bc we dont need it, hence the ', _'
                    # #cv.RETR_EXTERNAL: only finds outermost contours
                    # #cv.CHAIN_APPROX_SIMPLE: makes contour data more efficient (removes unnecessary points)

                    # #CROPPING PEOPLE AND ZOOMING IN ON THEM
                    # if contours:#if there are contours (which means there is a person)
                    #     for contour in contours:
                    #             x, y, w, h = cv.boundingRect(contour) #finds every contour for people
                    #             #x, y: coords for top left corner of box outlining person (mask)
                    #             #w, h: width and height of box outlining person (mask)
                                
                    #             if w * h < min_area_px: #skipping over small boxes
                    #                 continue

                    #             # # Clamp to image bounds
                    #             # x = max(0, x); y = max(0, y)
                    #             # x2 = min(frame.shape[1], x + w); y2 = min(frame.shape[0], y + h)

                    #             #skip failed cases: one huge contour spanning the whole frame
                    #             if x<= 1 and y<=1 and (x + w) >= (W - 1) and (y + h) >= H - 2:
                    #                 continue
                    #             if (w* h) >0.95 * (W * H): #EXPLAIN THE 0.95
                    #                 continue

                    #DRAWING BOXES AROUND PEOPLE
                    for box in people_boxes:
                        cv.rectangle(frame,
                        (box['x'], box['y']),
                        (box['x'] + box['width'], box['y'] + box['height']),
                        (0, 255, 0), 2)

                else:
                    people_boxes = []
                    frame_to_use = frame #if no person is detected, then use the original frame

                #real time feed so we need to give the timestamp of each individual frame since were using 'live stream' mode
                timestamp_ms = int(time.time() * 1000) #timestamp in milliseconds

                segmenter.segment_async(mp_image, timestamp_ms) # SEGMENTATION
                # objectmodel.detect_async(mp_image, timestamp_ms) # OBJECT DETECTION
                # handmodel.detect_async(mp_image, timestamp_ms) # processes and detects hands in video frame | HAND


                # run hand detection per-person ROI (cropped + upscaled) so distant hands are larger
                if 'people_boxes' in locals() and people_boxes:
                    for i, person in enumerate(people_boxes):
                        px1 = person['x']; py1 = person['y']
                        px2 = px1 + person['width']; py2 = py1 + person['height']
                        roi = frame[py1:py2, px1:px2]
                        if roi.size == 0:
                            continue
                        # upscale/zoom crop to make hands larger for the model
                        roi_up = cv.resize(roi, (frame.shape[1], frame.shape[0]))
                        roi_rgb = cv.cvtColor(roi_up, cv.COLOR_BGR2RGB)
                        mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
                        handmodel.detect_async(mp_roi, timestamp_ms + i) #unique timestamp per ROI
                else:
                    # fallback so gestures still work if no person box is found
                    handmodel.detect_async(mp_image, timestamp_ms)




                # #flip so not inverted (dont know why it does this by default)
                # frame_to_use = cv.flip(frame_to_use, 1)
                # cv.imshow('frame', frame_to_use) #displays frames in a window(thats what imshow does: opens a new window)
                # show full frame in main window; zoom in a separate window
                frame_main = cv.flip(frame, 1)
                cv.imshow('frame', frame_main)
                # if zoom_view is not None:
                #     cv.imshow('zoom', cv.flip(zoom_view, 1))
                # else:
                #     cv.destroyWindow('zoom')

                #displaying qrcode if it should be shown
                if should_show_qrcode:
                    qrcode_is_shown = True
                    qrcode_shown_start_time = time.time()
                    qrcode_generated = generate_qr()
                    cv.imshow('QR Code', qrcode_generated)
                    print("QR Code is being shown")
                    should_show_qrcode = False #resetting to False so that it doesnt show again

                #temporary popup for qrcode
                if qrcode_is_shown and time.time() - qrcode_shown_start_time >= 10: # 10 seconds is the time qrcode is shown for
                    #time.time() = current time - the start time gives us the total time qrcode has been shown
                    qrcode_is_shown = False
                    qrcode_shown_start_time = 0
                    cv.destroyWindow('QR Code')

                if cv.waitKey(1) == ord('q'): #exit if user presses 'q'
                    break

   
webcam.release()

cv.destroyAllWindows()






