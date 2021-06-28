import csv
import time
import cv2 as cv
from detect import *
# import numpy as np


# Define initial time
initial_time= time.time()

# Create a initiall list for create a csv file
lists_data = []

# Create object traker instance
tracker = ObjectTracker()

# Load the video
cap = cv.VideoCapture('./videos/fish.mp4')
# Create background mask
obj_detect = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

# Detect speed of the object
def speed_cal(time):
    #Here i converted m to Km and second to hour then divison to reach Speed in this form (KM/H)
    try:
        speed = (9.144*3600)/(time*1000)
        return speed
    except ZeroDivisionError:
        print(5)

# Create the csv and push the all data
with open('objdata.csv', 'w', newline='') as f:
    HEADERS = ['objid', 'xcordinate', 'ycordinate', 'width', 'height', 'speed']
    thewriter = csv.DictWriter(f, fieldnames=HEADERS)
    thewriter.writeheader()

    # Itaration for new frames
    while(cap.isOpened()):
        # Part 01. Object detection
        # Create object frame
        ret, frame = cap.read()

        # Height and Whidth
        height, widht, _ = frame.shape

        # Extract region of interst
        roi= frame[100: 400, 400: 800]


        # Create object mask
        # msk = obj_detect.apply(roi)
        msk = obj_detect.apply(frame)

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
            # Part 03. Speed calculation
            speed = speed_cal(time.time() - initial_time)
            # lists_data.append([id, x, y, w, h, speed])

            # Part 04. Create CSV file
            new_data = {'objid': id, 'xcordinate': x,
                        'ycordinate': y, 'width': w, 'height': h, 'speed': speed}
            thewriter.writerow(new_data)
            # print(new_data)

            # print("x", x)
            # print("y", y)
            # print("w", w)
            # print("h", h)
            # print("speed", speed)

            cv.putText(frame,  f'Fish {id} and speed {speed}',
                    (x, y - 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # print(boxes_ids)

        # cv.imshow("Roi", roi)
        cv.imshow("Mask", mask)
        cv.imshow("Frame", frame)

        # Write csv
        # for lst in lists_data:
            # print(lst)
            # id, x, y, w, h, spd = lst

            # new_data = {'objid': id, 'xcordinate': x,
            #             'ycordinate': y, 'width': w, 'height': h, 'speed': spd}

            # print(new_data)
            # thewriter.writerows()
        
        key = cv.waitKey(30)
        if key == 27:
            break

    
# # Create the csv and push the all data
# with open('objdata.csv', 'w', newline='') as f:
#     fieldnames = ['obj-id', 'x-cordinate', 'y-cordinate', 'width', 'height']
#     thewriter = csv.DictWriter(f, fieldnames=fieldnames)
#     thewriter.writeheader()
#     print(lists_data)
#     # for lst in lists_data:
#     #     print(lst)
#         # thewriter.writerows({
#         #     'obj-id': id,
#         #     'x-cordinate': x,
#         #     'y-cordinate': y,
#         #     'width': w,
#         #     'height': h
#         # })


cap.release()
cv.destroyAllWindows()
