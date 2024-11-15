import cv2
import os
import streamlit as st


# Set up Streamlit app title and description
st.title("Real-Time Face Tracking with Trace")
st.write("Sử dụng mô hình DNN để track khuôn mặt realtime")

# Load DNN model files for face detection
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"

# Check if files exist
if not os.path.isfile(model_file) or not os.path.isfile(config_file):
    st.error("Model file or config file not found. Please check the paths.")
else:
    # Load the DNN model
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    
    # Start the webcam
    cap = cv2.VideoCapture(0)

    # List to store the center positions of detected faces for tracing
    trace_points = []

    # Stream the video
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read() #ret = True nếu đọc khung ảnh thành công
        
        if not ret:
            st.write("Unable to fetch frame from the camera. Please check the camera connection.")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Get the frame dimensions
        (h, w) = frame.shape[:2]
        
        # Prepare the frame for the DNN model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        # Perform face detection
        net.setInput(blob)
        detections = net.forward()
        
        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * [w, h, w, h] #i là thứ tự khuôn mặt, lấy khuôn mặt từ 3 đến 6
                (x, y, x1, y1) = box.astype("int")
                
                # Calculate center of the detected face
                center_x = int((x + x1) / 2)
                center_y = int((y + y1) / 2)
                
                # Append center point to trace points list
                trace_points.append((center_x, center_y))
                
                # Draw bounding box and confidence score
                cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        
        # Draw the trace
        for j in range(1, len(trace_points)):
            if trace_points[j - 1] is None or trace_points[j] is None:
                continue
            cv2.line(frame, trace_points[j - 1], trace_points[j], (0, 255, 0), 2)

        # Limit trace length to the last 50 points
        if len(trace_points) > 50:
            trace_points.pop(0)
        
        # Convert the frame to RGB format and display it in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")
        
        # Break if Streamlit session is interrupted
        if st.session_state.get("stop"):
            break

    # Release the capture and close everything
    cap.release()
