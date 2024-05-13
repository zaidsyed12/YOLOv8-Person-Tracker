import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

def track_people(video_file, model, area1, area2):
    st.write("Processing...")

    cap = cv2.VideoCapture(video_file)

    # Read the coco.txt file
    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")

    count = 0
    tracker = Tracker()

    people_enter = {}
    counter1 = []

    people_exit = {}
    counter2 = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame)
        a = results[0].boxes.boxes
        px = pd.DataFrame(a).astype("float")
        list = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'person' in c:
                list.append([x1, y1, x2, y2])
        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox

            results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results >= 0:
                people_exit[id] = (x4, y4)

            if id in people_exit:
                results1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
                if results1 >= 0:
                    if counter2.count(id) == 0:
                        counter2.append(id)

            results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results2 >= 0:
                people_enter[id] = (x4, y4)

            if id in people_enter:
                results3 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
                if results3 >= 0:
                    if counter1.count(id) == 0:
                        counter1.append(id)

    cap.release()

    st.write("Processing complete!")
    st.write(f'People Entered: {len(counter1)}')
    st.write(f'People Exited: {len(counter2)}')

def detect_objects_on_uploaded_video(uploaded_video, model, area1, area2):
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    track_people(video_path, model, area1, area2)
def main():
    st.title("Object Detection and Person Tracking on Uploaded Video")

    model = YOLO('yolov8s.pt')
    st.write(" Polygon shape coordinates:")
    st.write(" 1      -->         4")
    st.write(" 2      -->         3")

    # Take user input for area1 coordinates
    st.subheader("Area 1 (Exiting) Coordinates")

    area1_x1 = st.number_input("Enter x1 coordinate for area1", value=0)
    area1_y1 = st.number_input("Enter y1 coordinate for area1", value=0)
    area1_x2 = st.number_input("Enter x2 coordinate for area1", value=0)
    area1_y2 = st.number_input("Enter y2 coordinate for area1", value=0)
    area1_x3 = st.number_input("Enter x3 coordinate for area1", value=0)
    area1_y3 = st.number_input("Enter y3 coordinate for area1", value=0)
    area1_x4 = st.number_input("Enter x4 coordinate for area1", value=0)
    area1_y4 = st.number_input("Enter y4 coordinate for area1", value=0)
    area1 = [(area1_x1, area1_y1), (area1_x2, area1_y2), (area1_x3, area1_y3), (area1_x4, area1_y4)]

    # Take user input for area2 coordinates
    st.subheader("Area 2 (Entering) Coordinates")
    area2_x1 = st.number_input("Enter x1 coordinate for area2", value=0)
    area2_y1 = st.number_input("Enter y1 coordinate for area2", value=0)
    area2_x2 = st.number_input("Enter x2 coordinate for area2", value=0)
    area2_y2 = st.number_input("Enter y2 coordinate for area2", value=0)
    area2_x3 = st.number_input("Enter x3 coordinate for area2", value=0)
    area2_y3 = st.number_input("Enter y3 coordinate for area2", value=0)
    area2_x4 = st.number_input("Enter x4 coordinate for area2", value=0)
    area2_y4 = st.number_input("Enter y4 coordinate for area2", value=0)
    area2 = [(area2_x1, area2_y1), (area2_x2, area2_y2), (area2_x3, area2_y3), (area2_x4, area2_y4)]

    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file is not None:
        detect_objects_on_uploaded_video(uploaded_file, model, area1, area2)

if __name__ == "__main__":
    main()
