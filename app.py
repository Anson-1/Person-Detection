import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import streamlit as st

# Function to process video frames
def process_frame(frame, model, box_annotator):
    # Run YOLOv8 inference
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)

    # Filter for "person" class (typically ID 0 in YOLO)
    person_class_id = 0
    person_detections = detections[detections.class_id == person_class_id]

    # Count the number of persons
    person_count = len(person_detections)

    # Annotate detections on the frame
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in person_detections
    ]
    annotated_frame = box_annotator.annotate(
        scene=frame, 
        detections=person_detections, 
        labels=labels
    )

    # Add person count to the frame
    cv2.putText(
        annotated_frame,
        f"Person Count: {person_count}",
        (20, 50),  # Position on the frame (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # Font scale
        (0, 255, 0),  # Font color (green)
        2,  # Thickness
        cv2.LINE_AA
    )

    return annotated_frame

# Function to resize frame for vertical orientation
def resize_to_vertical(frame, target_width=480):
    """
    Resize the frame to fit a vertical (portrait) aspect ratio.
    The width is fixed, and the height is dynamically adjusted to maintain the aspect ratio.
    """
    height, width, _ = frame.shape
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    resized_frame = cv2.resize(frame, (target_width, target_height))
    return resized_frame

# Streamlit app
def main():
    st.title("Person Detection with Vertical Camera Feed")
    st.sidebar.title("Settings")

    # Model selection
    model_choice = st.sidebar.selectbox("Select YOLO Model", ["yolov8n.pt"])
    model = YOLO(model_choice)
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    # Input for camera feed URL
    st.sidebar.write("Enter the camera feed URL (e.g., IP Webcam app URL):")
    camera_url = st.sidebar.text_input(
        "Camera URL", 
        "http://192.168.0.151:8080/video"  # Default IP camera example
    )

    # Start video processing
    stframe = st.empty()  # Placeholder for the video feed

    # Access the camera feed from the user-provided URL
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        st.error("Unable to access the camera feed. Please check the URL.")
        return

    # Process the video feed frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to retrieve frame. Exiting...")
            break

        # Resize frame for vertical orientation
        frame = resize_to_vertical(frame, target_width=480)

        # Process the frame
        annotated_frame = process_frame(frame, model, box_annotator)

        # Convert BGR to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        stframe.image(annotated_frame, channels="RGB", use_container_width=False)

    cap.release()

if __name__ == "__main__":
    main()

