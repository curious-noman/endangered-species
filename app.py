import streamlit as st
import numpy as np
from PIL import Image
from modular import predict_image
import time
import json
import os
from datetime import datetime, timedelta
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Title section (top box)
with st.container():
    st.title("Endangered Animal Recognition System")
    st.write(
        "Your camera will be continuously on. Output will only be provided when an animal is detected and accuracy is above 90%.")

# Initialize session state for logs
if 'detection_logs' not in st.session_state:
    st.session_state.detection_logs = []

if 'last_detections' not in st.session_state:
    st.session_state.last_detections = {}

if 'current_detection' not in st.session_state:
    st.session_state.current_detection = None

# Species info file path
SPECIES_INFO_FILE = "database/endangered.json"


# Load species information
def load_species_info():
    try:
        if os.path.exists(SPECIES_INFO_FILE):
            with open(SPECIES_INFO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            st.warning(f"Species info file not found: {SPECIES_INFO_FILE}")
            return {}
    except Exception as e:
        st.error(f"Error loading species info: {e}")
        return {}


# Get species details
def get_species_details(class_name):
    species_info = load_species_info()

    # Try direct match first
    if class_name in species_info:
        return species_info[class_name]

    # If no direct match, try case-insensitive match or partial match
    for species, info in species_info.items():
        if class_name.lower() in species.lower() or species.lower() in class_name.lower():
            return info

    # If no match found, return default values
    return {
        "scientific_name": "Not available",
        "status": "Unknown"
    }


# Function to save detection to log
def log_detection(class_name, category, confidence_score):
    # Check if we're in cooldown period for this species
    current_time = datetime.now()

    # Get the last detection time for this species from session state
    last_detection_time = st.session_state.last_detections.get(class_name)

    # Define cooldown period (in seconds)
    cooldown_period = 30  # 30 seconds cooldown

    # If we have a previous detection time and we're still in cooldown period, skip logging
    if last_detection_time and (current_time - last_detection_time).total_seconds() < cooldown_period:
        # We're in cooldown, don't log
        return None, cooldown_period - int((current_time - last_detection_time).total_seconds())

    # Not in cooldown, proceed with logging
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "class_name": class_name,
        "category": category,
        "confidence_score": float(confidence_score)
    }

    # Add to session state logs
    st.session_state.detection_logs.append(log_entry)

    # Update last detection time
    st.session_state.last_detections[class_name] = current_time

    return timestamp, 0


# Function to get status color and icon
def get_status_display(status):
    status_lower = status.lower()
    if "extinct" in status_lower:
        return "🔴", "#ff0000"  # Red
    elif "critically" in status_lower:
        return "⚠️", "#ff4500"  # Orange-Red
    elif "endangered" in status_lower:
        return "⚠️", "#ff8c00"  # Dark Orange
    elif "vulnerable" in status_lower:
        return "⚠️", "#ffd700"  # Gold
    elif "near" in status_lower and "threatened" in status_lower:
        return "⚠️", "#ffff00"  # Yellow
    elif "least" in status_lower and "concern" in status_lower:
        return "✅", "#90ee90"  # Light Green
    else:
        return "❓", "#808080"  # Gray


# WebRTC video processor class with improved error handling
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_process_time = time.time()
        self.cooldown_info = None
        self.frame_buffer = None
        self.last_valid_frame = None
        logger.info("VideoProcessor initialized")

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="rgb24")

            # Store the valid frame
            self.last_valid_frame = img.copy()

            # Process frames at a lower rate to reduce CPU load (every 300ms)
            # Increased from 200ms to improve stability
            current_time = time.time()
            if current_time - self.last_process_time > 0.3:
                # Convert to PIL Image for processing
                pil_image = Image.fromarray(img)

                try:
                    # Get prediction
                    class_name, category, confidence_score = predict_image(pil_image)

                    # Only process and display if not Environment/Human and confidence > 90%
                    if class_name and not (class_name.endswith("Human") or class_name.endswith(
                            "Environment")) and confidence_score >= 0.90:
                        # Get species details
                        species_details = get_species_details(class_name)
                        scientific_name = species_details.get("scientific_name", "Not available")
                        status = species_details.get("status", "Unknown")

                        # Get status icon and color
                        status_icon, status_color = get_status_display(status)

                        # Log the detection
                        timestamp, cooldown_remaining = log_detection(class_name, category, confidence_score)

                        if timestamp:
                            # Not in cooldown, show detection
                            cooldown_message = "✅ New detection logged!"
                            log_status = f"✅ Logged at {timestamp}"
                        else:
                            # In cooldown, show cooldown message
                            cooldown_message = f"⏳ Cooldown: {class_name} recently logged. New log in {cooldown_remaining}s"
                            log_status = "⚠️ Not logged - in cooldown period"

                        # Update session state with current detection
                        st.session_state.current_detection = {
                            "class_name": class_name,
                            "category": category,
                            "confidence_score": confidence_score,
                            "scientific_name": scientific_name,
                            "status": status,
                            "status_icon": status_icon,
                            "status_color": status_color,
                            "cooldown_message": cooldown_message,
                            "log_status": log_status
                        }
                except Exception as e:
                    logger.error(f"Error in prediction: {e}")

                self.last_process_time = current_time

            return av.VideoFrame.from_ndarray(img, format="rgb24")
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # Return the last valid frame if we have one, or the original frame
            if self.last_valid_frame is not None:
                return av.VideoFrame.from_ndarray(self.last_valid_frame, format="rgb24")
            return frame


# Layout: Create two columns for the bottom section
col1, col2 = st.columns([1, 2])  # 1:2 ratio for width

# Left smaller box (column 1)
with col1:
    st.subheader("Detection Results")

    # This is where you'll display your results
    results_placeholder = st.empty()

    # Add a placeholder for species details
    species_details_placeholder = st.empty()

    # Add a cooldown indicator
    cooldown_placeholder = st.empty()

# Right larger box (column 2)
with col2:
    st.subheader("Camera Feed")
    # Add diagnostic info for users
    st.info("Camera initialization will take a moment. If you see a red loading indicator, please wait.")

    # Configure WebRTC with multiple STUN servers for better connectivity
    rtc_configuration = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]}
    )

    # Use try-except to handle WebRTC errors gracefully
    try:
        # Create the WebRTC streamer with optimized settings
        webrtc_ctx = webrtc_streamer(
            key="endangered-animal-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 30}
                },
                "audio": False
            },
            async_processing=True,
        )

        # Check WebRTC state
        if webrtc_ctx.state.playing:
            st.success("Camera is active and streaming!")
        else:
            st.warning(
                "Camera is inactive. Click 'START' to begin detection. If it doesn't start, try refreshing the page.")

    except Exception as e:
        st.error(f"Camera initialization error: {str(e)}")
        st.info(
            "Try refreshing the page, checking your browser permissions, or using a different browser (Chrome recommended).")

# Display detection results
if st.session_state.current_detection is not None and webrtc_ctx.state.playing:
    detection = st.session_state.current_detection

    # Display the detection result
    label_text = f"**{detection['class_name']}**\n🟢 **Category:** {detection['category']}\n📊 **Confidence Score:** {detection['confidence_score'] * 100:.2f}%"
    results_placeholder.markdown(label_text)

    # Show cooldown status
    cooldown_placeholder.info(detection['cooldown_message'])

    # Create species details content
    species_details_html = f"""
    <div style="padding-top: 10px; padding-bottom: 10px;">
        <hr>
        <h3>🔍 Species Details</h3>
        <table style="width: 100%;">
            <tr>
                <td style="width: 40%;"><strong>Common Name:</strong></td>
                <td><strong>{detection['class_name']}</strong></td>
            </tr>
            <tr>
                <td><strong>Scientific Name:</strong></td>
                <td><em>{detection['scientific_name']}</em></td>
            </tr>
            <tr>
                <td><strong>Conservation Status:</strong></td>
                <td>{detection['status_icon']} <span style='color:{detection['status_color']}'>{detection['status']}</span></td>
            </tr>
            <tr>
                <td><strong>Log Status:</strong></td>
                <td>{detection['log_status']}</td>
            </tr>
            <tr>
                <td><strong>Confidence Score:</strong></td>
                <td><span style='color:{detection['status_color']}'>{detection['confidence_score'] * 100:.2f}%</span></td>
            </tr>
        </table>
    </div>
    """

    # Update the species details placeholder
    species_details_placeholder.markdown(species_details_html, unsafe_allow_html=True)