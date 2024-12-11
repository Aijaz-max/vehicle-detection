import numpy as np
import time
import cv2
from scipy.spatial import KDTree
import os


list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416 


LABELS = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
weightsPath = 'yolov3.weights'
configPath = 'yolov3.cfg'
inputVideoPath = 'input_video.mp4'
outputVideoPath = 'output_video.avi'
preDefinedConfidence = 0.5
preDefinedThreshold = 0.4
USE_GPU = False


COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


def displayVehicleCount(frame, vehicle_count):
    cv2.putText(frame, f'Detected Vehicles and Objects: {vehicle_count}', (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)


def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
    x1_line, y1_line, x2_line, y2_line = line_coordinates
    if x_mid_point >= x1_line and x_mid_point <= x2_line + 5 and y_mid_point >= y1_line and y_mid_point <= y2_line + 5:
        return True
    return False


def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if current_time > start_time:
        os.system('cls' if os.name == 'nt' else 'clear')  # 
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames


def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{LABELS[classIDs[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 255, 0), thickness=2)


def initializeVideoWriter(video_width, video_height, videoStream):
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps, (video_width, video_height), True)

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf  # Initializing the minimum distance
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:
            continue
        tree = KDTree(coordinate_list)
        temp_dist, index = tree.query([(centerX, centerY)])
        if temp_dist < dist:
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]
    if dist > (max(width, height) / 2):
        return False
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True


def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centerX = x + (w // 2)
            centerY = y + (h // 2)
            if LABELS[classIDs[i]] in list_of_vehicles:
                current_detections[(centerX, centerY)] = vehicle_count
                if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections):
                    vehicle_count += 1
    previous_frame_detections.append(current_detections)
    if len(previous_frame_detections) > FRAMES_BEFORE_CURRENT:
        previous_frame_detections.pop(0)
    return vehicle_count