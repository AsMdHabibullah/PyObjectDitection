import cv2 as cv
from detect import *
# import numpy as np


# Create object traker instance
tracker = ObjectTracker()

# Load the video
cap = cv.VideoCapture('./videos/fish-swimming1.mp4')
# Create background mask
obj_detect = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

while(cap.isOpened()):
    # Part 01. Object detection
    # Create object frame
    ret, frame = cap.read()

    # Height and Whidth
    height, widht, _ = frame.shape

    # Extract region of interst
    roi= frame[140: 440, 400: 700]


    # Create object mask
    msk = obj_detect.apply(roi)

    # Clean the mask
    _, mask = cv.threshold(msk, 254, 255, cv.THRESH_BINARY)
    # Create object contours
    contours, _ =cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # detections object array
    detections= []

    for cnt in contours:
        # Calculate the area
        area = cv.contourArea(cnt)
        if area > 100:
            # cv.drawContours(roi, [cnt], -1, (0, 225, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            detections.append([x, y, w, h])

    # Part 02. Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(roi, (str(id)), (x, y - 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # print(boxes_ids)

    # cv.imshow("Roi", roi)
    cv.imshow("Mask", mask)
    cv.imshow("Frame", frame)

    key = cv.waitKey(30)

    if key == 27:
        break

cap.release()
cv.destroyAllWindows()