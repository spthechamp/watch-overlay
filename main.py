import streamlit as st
import importlib
import cv2
import numpy as np
import tempfile
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cvzone
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from av import VideoFrame  # Needed for converting frames in webcam processing

# Created by @spthechamp

# -------------------------------
# Helper Class for Hand Detection
# -------------------------------
class HandLandmarkerStreamlit:
    def __init__(self, model_path, watch_path):
        self.model_path = model_path
        # Initialize the detector options.
        self.base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(self.options)
        
        # Load the watch image (with transparency)
        self.watch_image = cv2.imread(watch_path, cv2.IMREAD_UNCHANGED)
        if self.watch_image is None:
            st.warning(f"Warning: Unable to load watch image from {watch_path}")
    
    def process_frame(self, frame):
        """
        Processes a single frame (BGR image) and returns the annotated frame.
        """
        height, width, _ = frame.shape
        # Convert frame to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        annotated_frame = frame.copy()
        
        handedness_list = result.handedness
        hand_landmarks_list = result.hand_landmarks

        if handedness_list:
            for i in range(len(handedness_list)):
                handedness = handedness_list[i][0].category_name
                landmarks = hand_landmarks_list[i]
                keypoints_x = [lm.x * width for lm in landmarks]
                keypoints_y = [lm.y * height for lm in landmarks]
                
                if handedness == 'Left':
                    # Draw left-hand landmarks
                    for j in range(len(keypoints_x)):
                        cv2.circle(annotated_frame, (int(keypoints_x[j]), int(keypoints_y[j])), 4, (0, 255, 0), -2)
                    x_min = int(min(keypoints_x))
                    y_min = int(min(keypoints_y))
                    x_max = int(max(keypoints_x))
                    y_max = int(max(keypoints_y))
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                    cv2.putText(annotated_frame, "Left", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                else:
                    # Right hand processing
                    wrist_x = keypoints_x[0]
                    wrist_y = keypoints_y[0]
                    rt_coord = (keypoints_x[2], keypoints_y[2])
                    ri_coord = (keypoints_x[5], keypoints_y[5])
                    rm_coord = (keypoints_x[9], keypoints_y[9])
                    rr_coord = (keypoints_x[13], keypoints_y[13])
                    rl_coord = (keypoints_x[17], keypoints_y[17])
                    
                    rt_rl = rt_coord[0] - rl_coord[0]
                    # Calculate angle for rotation
                    tan_theta = (rm_coord[1] - wrist_y) / (rm_coord[0] - wrist_x) if (rm_coord[0] - wrist_x) != 0 else 0
                    theta = np.arctan(tan_theta)
                    theta_degrees = np.degrees(theta)
                    
                    # Decide orientation and view
                    if wrist_y > ri_coord[1] and wrist_y > rm_coord[1] and wrist_y > rr_coord[1]:
                        hand_orientation = 'straight'
                        hand_view = 'palm' if rt_rl > 0 else 'dorsum'
                    else:
                        hand_orientation = 'rotated'
                        hand_view = 'dorsum' if rt_rl > 0 else 'palm'
                    
                    # If the hand view is dorsum and the watch image is available, overlay it
                    if hand_view == 'dorsum' and self.watch_image is not None:
                        x_min = int(min(keypoints_x))
                        y_min = int(min(keypoints_y))
                        x_max = int(max(keypoints_x))
                        y_max = int(max(keypoints_y))
                        # Resize the watch image to half the width of the hand bounding box
                        watch_size = (int((x_max - x_min) / 2), int((x_max - x_min) / 2))
                        watch_resized = cv2.resize(self.watch_image, watch_size)
                        
                        # Define position: centered at the wrist.
                        fit_xmin = int(wrist_x - watch_resized.shape[1] // 2)
                        fit_ymin = int(wrist_y - watch_resized.shape[0] // 2)
                        fit_xmax = fit_xmin + watch_resized.shape[1]
                        fit_ymax = fit_ymin + watch_resized.shape[0]
                        
                        # Check boundaries before overlaying
                        if fit_xmin >= 0 and fit_ymin >= 0 and fit_xmax <= width and fit_ymax <= height:
                            M = cv2.getRotationMatrix2D((watch_resized.shape[1] // 2, watch_resized.shape[0] // 2), -theta_degrees, 1)
                            watch_rotated = cv2.warpAffine(watch_resized, M, (watch_resized.shape[1], watch_resized.shape[0]))
                            annotated_frame = cvzone.overlayPNG(annotated_frame, watch_rotated, (fit_xmin, fit_ymin))
                    else:
                        # Draw right-hand landmarks normally.
                        for k in range(len(keypoints_x)):
                            cv2.circle(annotated_frame, (int(keypoints_x[k]), int(keypoints_y[k])), 4, (0, 255, 0), -2)
                        x_min = int(min(keypoints_x))
                        y_min = int(min(keypoints_y))
                        x_max = int(max(keypoints_x))
                        y_max = int(max(keypoints_y))
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                        cv2.putText(annotated_frame, "Right", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return annotated_frame

# -------------------------------
# Video Processor for Webcam
# -------------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self, detector):
        self.detector = detector

    def recv(self, frame):
        # Convert incoming frame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        annotated_img = self.detector.process_frame(img)
        # Convert the NumPy array back to VideoFrame
        return VideoFrame.from_ndarray(annotated_img, format="bgr24")

# -------------------------------
# Main Streamlit App
# -------------------------------

# Set paths for model and watch image
MODEL_PATH = "hand_landmarker.task"  # change to your model file path
WATCH_IMAGE_PATH = "FREE-Watch-Clipart.png"         # ensure this image exists

# Initialize our hand detector
detector = HandLandmarkerStreamlit(MODEL_PATH, WATCH_IMAGE_PATH)

# Sidebar navigation
st.sidebar.title("Hand Landmark Detection and Watch Overlay App")
app_mode = st.sidebar.selectbox("Choose the page", ["Image", "Video", "WebCam"])

if app_mode == "Image":
    st.title("Hand Landmark Detection and Watch Overlay on Images")
    st.write("NOTE: Wrist watch will only be overlayed on the dorsal part of your right hand.")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Read image file as numpy array.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Error: Unable to read the image file.")
        else:
            st.image(image, channels="BGR", caption="Original Image")
            if st.button("Process Image"):
                annotated_image = detector.process_frame(image)
                st.image(annotated_image, channels="BGR", caption="Processed Image")
                # Save to a temporary file for download
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(temp_file.name, annotated_image)
                with open(temp_file.name, "rb") as file:
                    st.download_button(label="Download Processed Image",
                                       data=file.read(),
                                       file_name="processed_image.jpg",
                                       mime="image/jpeg")
                os.unlink(temp_file.name)

elif app_mode == "Video":
    st.title("Hand Landmark Detection and Watch Overlay on Video")
    st.write("NOTE: Wrist watch will only be overlayed on the dorsal part of your right hand.")
    
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        if st.button("Process Video"):
            # Save the uploaded video to a temporary file.
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Error: Unable to open the video file.")
            else:
                # Get video properties.
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25  # Use a default FPS if not available.
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if width == 0 or height == 0:
                    st.error("Error: Invalid video dimensions.")
                    cap.release()
                    os.unlink(tfile.name)
                    st.stop()

                # Define the output video path.
                out_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
                
                # Use 'mp4v' as the codec which is more widely supported.
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

                # Test if VideoWriter opened successfully.
                if not out.isOpened():
                    st.error("Error: Could not create video writer. Please check codec and video properties.")
                    cap.release()
                    os.unlink(tfile.name)
                    st.stop()

                frame_count = 0
                progress_bar = st.progress(0)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated_frame = detector.process_frame(frame)
                    out.write(annotated_frame)
                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                cap.release()
                out.release()

                # Wait briefly to ensure the file is fully written.
                time.sleep(2)
                
                if not os.path.exists(out_video_path):
                    st.error(f"Error: Processed video file not found at {out_video_path}")
                else:
                    with open(out_video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    
                    if len(video_bytes) == 0:
                        st.error("Error: Processed video file is empty.")
                    else:
                        st.success("Video processing completed!")
                        st.download_button(label="Download Processed Video",
                                           data=video_bytes,
                                           file_name="processed_video.mp4",
                                           mime="video/mp4")
            os.unlink(tfile.name)


elif app_mode == "WebCam":
    st.title("Live Webcam Hand Landmark Detection and Watch Overlay")
    st.write("This page uses your webcam to display live hand landmark detection and watch overlay.")
    st.write("NOTE: You might need to allow webcam access in your browser. Wrist watch will only be overlayed on the dorsal part of your right hand.")
    webrtc_streamer(key="example", video_processor_factory=lambda: VideoProcessor(detector))
