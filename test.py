'''
PyPower Projects
Age, Gender, and Emotion Detection Using AI with Gradio Interface
'''

# USAGE: python combined_detection_gradio.py

import cv2 as cv
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time
import gradio as gr

# Load face detector (DNN-based)
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Load age and gender models (Caffe)
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageNet = cv.dnn.readNetFromCaffe(ageProto, ageModel)
genderNet = cv.dnn.readNetFromCaffe(genderProto, genderModel)

# Load emotion model (Keras)
emotion_classifier = load_model('./Emotion_Detection.h5')

# Define labels
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Global webcam state
webcam_state = {'running': False}

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def process_frame(frame, padding=20):
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        cv.putText(frameFace, "No face detected", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        result_text = "No face detected"
        return frameFace, result_text

    result_text = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        face = frame[max(0, y1-padding):min(y2+padding, frame.shape[0]-1), 
                     max(0, x1-padding):min(x2+padding, frame.shape[1]-1)]

        # --- Age and Gender Detection ---
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender_idx = genderPreds[0].argmax()
        gender = genderList[gender_idx]
        gender_confidence = genderPreds[0][gender_idx]
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age_idx = agePreds[0].argmax()
        age = ageList[age_idx]
        age_confidence = agePreds[0][age_idx]

        # --- Emotion Detection ---
        gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        roi_gray = cv.resize(gray, (48, 48), interpolation=cv.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            emotion_preds = emotion_classifier.predict(roi)[0]
            emotion_idx = emotion_preds.argmax()
            emotion = emotion_labels[emotion_idx]
            emotion_confidence = emotion_preds[emotion_idx]
        else:
            emotion = "Unknown"
            emotion_confidence = 0.0

        # Format the result text
        result = (
            f"Gender: {gender}, confidence = {gender_confidence:.3f}\n"
            f"Age: {age}, confidence = {age_confidence:.3f}\n"
            f"Emotion: {emotion}, confidence = {emotion_confidence:.3f}\n"
            f"Time: {time.time() - t:.3f}s\n"
            f"{'-' * 40}"
        )

        # Print to terminal
        print(result)

        # Add to result text for Gradio UI
        result_text.append(result)

        # Display on frame
        label = f"{gender}, {age}, {emotion}"
        cv.putText(frameFace, label, (x1-5, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    return frameFace, "\n".join(result_text)

# Function for image upload
def detect_from_image(image):
    # Convert Gradio image (PIL format) to OpenCV format
    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    result_image, result_text = process_frame(image)
    # Convert back to RGB for Gradio display
    result_image = cv.cvtColor(result_image, cv.COLOR_BGR2RGB)
    return result_image, result_text

# Function for live webcam streaming
def detect_from_webcam():
    cap = None
    last_frame = None
    last_text = "Webcam is not running. Click 'Start Webcam' to begin."
    while True:
        if webcam_state['running']:
            if cap is None:
                cap = cv.VideoCapture(0)
                if not cap.isOpened():
                    last_text = "Error: Could not open webcam."
                    yield None, last_text
                    continue

            ret, frame = cap.read()
            if not ret:
                last_text = "Error: Could not read frame."
                yield None, last_text
                continue

            result_frame, result_text = process_frame(frame)
            # Convert to RGB for Gradio
            result_frame = cv.cvtColor(result_frame, cv.COLOR_BGR2RGB)
            last_frame = result_frame
            last_text = result_text
            yield last_frame, last_text
        else:
            if cap is not None:
                cap.release()
                cap = None
                cv.destroyAllWindows()
            if last_frame is not None:
                yield last_frame, last_text
            else:
                yield None, last_text
            time.sleep(0.1)  # Reduce CPU usage when not running

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Age, Gender, and Emotion Detection")
    gr.Markdown("Upload an image or use the webcam for live detection. Results will be shown below and printed in the terminal.")

    with gr.Tab("Upload Image"):
        image_input = gr.Image(type="pil", label="Upload an Image")
        image_output = gr.Image(type="numpy", label="Result")
        text_output = gr.Textbox(label="Detection Results")
        image_button = gr.Button("Detect")
        image_button.click(
            fn=detect_from_image,
            inputs=image_input,
            outputs=[image_output, text_output]
        )

    with gr.Tab("Live Webcam"):
        webcam_output = gr.Image(type="numpy", label="Live Feed")
        webcam_text = gr.Textbox(label="Detection Results")
        start_button = gr.Button("Start Webcam")
        stop_button = gr.Button("Stop Webcam")

        start_button.click(
            fn=lambda: webcam_state.update({'running': True}),
            inputs=[],
            outputs=[]
        )
        stop_button.click(
            fn=lambda: webcam_state.update({'running': False}),
            inputs=[],
            outputs=[]
        )

        # Use a generator for live streaming
        gr.Interface(
            fn=detect_from_webcam,
            inputs=None,
            outputs=[webcam_output, webcam_text],
            live=True
        )

# Launch the Gradio interface
demo.launch()