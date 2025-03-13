import streamlit as st
import cv2
import numpy as np
from PIL import Image
from modular import predict_image
import time
import json
import os
from datetime import datetime, timedelta

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Log file path
LOG_FILE = "logs/detection_logs.json"

# Species info file path (make sure this points to the correct location)
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


# Initialize log file if it doesn't exist or is invalid
def initialize_log_file():
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                file_content = f.read().strip()
                if not file_content:
                    # File exists but is empty
                    with open(LOG_FILE, "w") as f:
                        json.dump([], f)
                else:
                    # Check if content is valid JSON
                    try:
                        json.loads(file_content)
                    except json.JSONDecodeError:
                        # Invalid JSON, reset file
                        with open(LOG_FILE, "w") as f:
                            json.dump([], f)
        else:
            # File doesn't exist, create it
            with open(LOG_FILE, "w") as f:
                json.dump([], f)
    except Exception as e:
        st.error(f"Error initializing log file: {e}")
        # Try again with error handling
        try:
            with open(LOG_FILE, "w") as f:
                json.dump([], f)
        except Exception as e:
            st.error(f"Failed to create log file: {e}")


# Call the initialization function
initialize_log_file()


# Function to save detection to log file with cooldown check
def log_detection(class_name, category, confidence_score):
    # Check if we're in cooldown period for this species
    current_time = datetime.now()

    # Get the last detection time for this species from session state
    last_detection_key = f"last_detection_{class_name}"
    last_detection_time = st.session_state.get(last_detection_key)

    # Define cooldown period (in seconds)
    cooldown_period = 30  # 30 seconds cooldown

    # If we have a previous detection time and we're still in cooldown period, skip logging
    if last_detection_time and (current_time - last_detection_time).total_seconds() < cooldown_period:
        # We're in cooldown, don't log
        return None

    # Not in cooldown, proceed with logging
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Read existing logs with error handling
        try:
            with open(LOG_FILE, "r") as f:
                file_content = f.read().strip()
                if not file_content:
                    logs = []
                else:
                    logs = json.loads(file_content)
        except json.JSONDecodeError:
            logs = []

        # Add new log
        log_entry = {
            "timestamp": timestamp,
            "class_name": class_name,
            "category": category,
            "confidence_score": float(confidence_score)
        }

        logs.append(log_entry)

        # Save updated logs
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=4)

        # Update session state with current time for this species
        st.session_state[last_detection_key] = current_time

    except Exception as e:
        st.error(f"Error logging detection: {e}")

    return timestamp


# Initialize session state for tracking last detections if it doesn't exist
if 'detection_history' not in st.session_state:
    st.session_state['detection_history'] = {}


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


# Title section (top box)
with st.container():
    st.title("Endangered Animal Recognition System")
    st.write(
        "Your camera will be continuously on. Output will only be provided when an animal is detected and accuracy is above 90%.")

st.subheader("Detection Results")
st.subheader("Camera Feed")
# Create two columns for the bottom section
col1, col2 = st.columns([1, 2])  # 1:2 ratio for width

# Left smaller box (column 1)
with col1:

    # This is where you'll display your results
    results_placeholder = st.empty()

    # Add a placeholder for species details (not a container)
    species_details_placeholder = st.empty()

    # Add a cooldown indicator
    cooldown_placeholder = st.empty()


# Right larger box (column 2)
with col2:


    # This is where your camera feed will go
    camera_placeholder = st.empty()

    # Camera start button
    run_camera = st.checkbox("Start Camera")


    if run_camera:
        cap = cv2.VideoCapture(0)  # Start the camera

        # Optimize camera settings to reduce delay
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Increase FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size

        if not cap.isOpened():  # Show error if camera couldn't be opened
            st.error("Camera couldn't be opened! Another program might be using it.")
            run_camera = False  # End the loop

        last_process_time = time.time()
        display_frame = None

        while run_camera:
            ret, frame = cap.read()

            # If image can't be captured from camera, wait for a while and try again
            if not ret:
                st.warning("Can't get image from camera! Check the connection.")
                time.sleep(0.1)  # Reduced wait time
                continue  # Continue the loop

            # Process frames at a lower rate to reduce CPU load (every 200ms)
            current_time = time.time()
            if current_time - last_process_time > 0.1:
                # Convert image to color format
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(image)

                class_name, category, confidence_score = predict_image(img_pil)

                # Only process and display if not Environment/Human and confidence > 90%
                if class_name and not (class_name.endswith("Human") or class_name.endswith("Environment")) and confidence_score >= 0.90:


                    # Get species details
                    species_details = get_species_details(class_name)
                    scientific_name = species_details.get("scientific_name", "Not available")
                    status = species_details.get("status", "Unknown")

                    # Get status icon and color
                    status_icon, status_color = get_status_display(status)

                    # Check if animal is in cooldown period
                    last_detection_key = f"last_detection_{class_name}"
                    last_detection_time = st.session_state.get(last_detection_key)
                    current_datetime = datetime.now()

                    # Define cooldown period (in seconds)
                    cooldown_period = 30  # 30 seconds cooldown

                    # If we have a previous detection and we're still in cooldown, display cooldown message
                    if last_detection_time and (
                            current_datetime - last_detection_time).total_seconds() < cooldown_period:
                        # Calculate remaining cooldown time
                        remaining = cooldown_period - int((current_datetime - last_detection_time).total_seconds())
                        cooldown_placeholder.info(f"⏳ Cooldown: {class_name} recently logged. New log in {remaining}s")

                        # Still display the animal but don't log it
                        label_text = f"**{class_name}**\n🟢 **Sınıfı:** {category}\n📊 **Güven Skoru:** {confidence_score * 100:.2f}%\n⚠️ **Not logged - in cooldown period**"
                        log_status = "⚠️ Not logged - in cooldown period"

                    else:
                        # Log the detection and get timestamp
                        timestamp = log_detection(class_name, category, confidence_score)

                        if timestamp:  # If logging was successful (not in cooldown)
                            label_text = f"**{class_name}**\n🟢 **Sınıfı:** {category}\n📊 **Güven Skoru:** {confidence_score * 100:.2f}%\n⏱️ **Zaman:** {timestamp}"
                            cooldown_placeholder.success("✅ New detection logged!")
                            log_status = f"✅ Logged at {timestamp}"


                    # Display the detection result
                    results_placeholder.markdown(label_text)

                    # Create species details content
                    species_details_html = f"""
                    <div style="padding-top: 10px; padding-bottom: 10px;">
                        <hr>
                        <h3>🔍 Species Details</h3>
                        <table style="width: 100%;">
                            <tr>
                                <td style="width: 40%;"><strong>Common Name:</strong></td>
                                <td><strong>{class_name}</strong></td>
                            </tr>
                            <tr>
                                <td><strong>Scientific Name:</strong></td>
                                <td><em>{scientific_name}</em></td>
                            </tr>
                            <tr>
                                <td><strong>Conservation Status:</strong></td>
                                <td>{status_icon} <span style='color:{status_color}'>{status}</span></td>
                            </tr>
                            <tr>
                                <td><strong>Log Status:</strong></td>
                                <td>{log_status}</td>
                            </tr>
                            <tr>
                                <td><strong>Confidence Score:</strong></td>
                                <td><span style='color:{status_color}'>{confidence_score * 100:.2f}%</span></td>
                            </tr>
                        </table>
                    </div>
                    """

                    # Update the species details placeholder
                    species_details_placeholder.markdown(species_details_html, unsafe_allow_html=True)

                    # # Add label to the frame (simplified for display)
                    # cv2.putText(frame, f"{class_name} - {confidence_score * 100:.2f}%",
                    #             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                display_frame = frame
                last_process_time = current_time

            # Always display the most recent processed frame
            if display_frame is not None:
                camera_placeholder.image(display_frame, channels="BGR")

            # Add a small sleep to prevent hogging the CPU
            time.sleep(0.01)

        cap.release()  # Release the camera
        cv2.destroyAllWindows()