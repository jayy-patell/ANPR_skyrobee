from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# load models
try:
    coco_model = YOLO('../models/yolov8s.pt')
    license_plate_detector = YOLO('../models/license_plate_detector.pt')
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)


# load video
cap = cv2.VideoCapture('../videos/sample.mp4')
if not cap.isOpened():
    print("Error opening video file")
    exit(1)

vehicles = [2, 3, 5, 7]  # car, bus, truck, motorbike

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        print(f"Processing frame {frame_nmr}")

        # detect vehicles
        try:
            detections = coco_model(frame)[0]
            detections.save('../detections.jpg')
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            continue
        
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        try:
            track_ids = mot_tracker.update(np.asarray(detections_))
        except Exception as e:
            print(f"Error in vehicle tracking: {e}")
            continue

        # detect license plates
        try:
            license_plates = license_plate_detector(frame)[0]
            license_plates.save('../license_plates.jpg')
        except Exception as e:
            print(f"Error in license plate detection: {e}")
            continue

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, score: {score}')

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # cv2.imshow('original_crop', license_plate_crop)
                # cv2.imshow('threshold', license_plate_crop_thresh)
                # cv2.waitKey(0)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, './test.csv')