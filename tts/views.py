from django.shortcuts import render
from django.http import HttpResponse
from gtts import gTTS
import os

def index(request):
    return render(request,'index.html')

def text_to_speech_view(request):
    if request.method == "POST":
        # Get the text input from the form
        text = request.POST.get("text", "")
        
        if text:
            # Generate speech using gTTS with the provided text
            tts = gTTS(text)
            tts.save("speech.mp3")
            
            # Return the generated MP3 file as an HTTP response
            with open("speech.mp3", "rb") as speech_file:
                response = HttpResponse(speech_file.read(), content_type="audio/mpeg")
                response['Content-Disposition'] = 'attachment; filename="speech.mp3"'
                return response
    return HttpResponse("No text provided.", status=400)


from django.shortcuts import render
from django.http import JsonResponse
import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Listening...")
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Error with the recognition service."

def speech_recognition_view(request):
    if request.method == 'POST':
        text = speech_to_text()
        return JsonResponse({'transcribed_text': text})
    return render(request, 'speech_recognition.html')



def texttospeech(request):
    return render(request,'form.html')

def speechtotext(request):
    return render(request,'form1.html')





from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
import cv2
import mediapipe as mp
import time

# Initialize Mediapipe and other resources globally
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Global variables for controlling gesture detection
gesture_detection_running = False
gesture_text = "No gesture detected"

# Gesture mapping dictionary
gesture_mapping = {
    "Thumbs Up": "Good job",
    "Open Palm": "Hello",
    "Peace Sign": "Victory",
    "One": "Number 1",
    "Fist": "Stop",
    "Two": "Number 2",
    "Three": "Number 3",
    "Four": "Number 4",
    "Five": "Number 5",
    "Rock": "Rock On",
    "Call Me": "Call Me Gesture",
    "Thumbs Down": "Not Good",
    "Okay": "Okay Gesture",
    "Love You": "I Love You Gesture",
    "Crossed Fingers": "Good Luck",
    # Add other gestures here...
}

# Function to interpret hand landmarks and detect gestures
def interpret_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Simple gesture detection logic
    if (index_tip.y < thumb_tip.y and
        middle_tip.y < thumb_tip.y and
        ring_tip.y < thumb_tip.y and
        pinky_tip.y < thumb_tip.y):
        return "Open Palm"

    if thumb_tip.y < index_tip.y and middle_tip.y > index_tip.y:
        return "Thumbs Up"
    if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y:
        return "Peace Sign"
    if all(finger_tip.y > index_tip.y for finger_tip in [middle_tip, ring_tip, pinky_tip]):
        return "One"
    if all(finger_tip.y < thumb_tip.y for finger_tip in [middle_tip, index_tip]):
        return "Fist"
    if thumb_tip.x < index_tip.x and middle_tip.y < thumb_tip.y:
        return "Call Me"
    if index_tip.x < thumb_tip.x and thumb_tip.y < middle_tip.y:
        return "Rock"
    if all(finger_tip.y > thumb_tip.y for finger_tip in [index_tip, middle_tip]):
        return "Thumbs Down"
    if all(finger_tip.y < thumb_tip.y for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "Five"
    if thumb_tip.y > index_tip.y and middle_tip.y > index_tip.y:
        return "Okay"
    if index_tip.x < thumb_tip.x and middle_tip.x > index_tip.x:
        return "Love You"
    if thumb_tip.y > middle_tip.y and middle_tip.y < index_tip.y:
        return "Crossed Fingers"
    if ring_tip.y > index_tip.y and pinky_tip.y < middle_tip.y:
        return "Two"
    if ring_tip.y > index_tip.y and pinky_tip.y < ring_tip.y:
        return "Three"
    return "No recognizable gesture"



#Video stream generator function
def generate_frames():
    global gesture_text
    cap = cv2.VideoCapture(0)

    while gesture_detection_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        detection_results = "No gesture detected"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = interpret_gesture(hand_landmarks)
                if gesture in gesture_mapping:
                    detection_results = f"Gesture: {gesture} - Meaning: {gesture_mapping[gesture]}"
        
        # Overlay detection results on the video frame
        cv2.putText(frame, detection_results, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        gesture_text = detection_results  # Update global gesture text

        # Encode the frame as JPEG and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Stream view to return video frames
def video_feed(request):
     return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
# View to start gesture detection
def start_gesture_detection(request):
    global gesture_detection_running
    gesture_detection_running = True
    return JsonResponse({'status': 'Gesture detection started'})

# View to stop gesture detection
def stop_gesture_detection(request):
    global gesture_detection_running
    gesture_detection_running = False
    return JsonResponse({'status': 'Gesture detection stopped'})

# View to render the main gesture recognition page
def gesture_recognition_page(request):
    return render(request, 'gesture_recognition.html', {'gesture_text': gesture_text})




import os
import cv2
from django.http import JsonResponse
from django.shortcuts import render
from pathlib import Path

# Define BASE_DIR
BASE_DIR = Path(__file__).resolve().parent.parent

def detect_objects(request):
    if request.method == 'POST':  # Only process detection on POST request
        try:
            # Define YOLO paths
            cfg_path = BASE_DIR / "yolo" / "yolov3.cfg"
            weights_path = BASE_DIR / "yolo" / "yolov3.weights"
            names_path = BASE_DIR / "yolo" / "coco.names"

            # Check if files exist
            if not cfg_path.exists() or not weights_path.exists() or not names_path.exists():
                return JsonResponse({'error': 'YOLO configuration, weights, or class names file not found.'})

            # Load YOLO model
            net = cv2.dnn.readNet(str(weights_path), str(cfg_path))
            with open(names_path, "r") as f:
                classes = f.read().strip().split("\n")

            # Access the camera feed
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                return JsonResponse({'error': 'Failed to capture image from camera.'})

            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())

            detection_results = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = scores.argmax()
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        class_name = classes[class_id]
                        detection_results.append(f"Detected: {class_name} with confidence {confidence:.2f}")

            cap.release()
            cv2.destroyAllWindows()

            # Pass the results to the template
            if detection_results:
                return render(request, 'detections.html', {'detections': detection_results})
            else:
                return render(request, 'detections.html', {'message': 'No objects detected with sufficient confidence.'})

        except Exception as e:
            return JsonResponse({'error': str(e)})

    # If it's a GET request, render the button without detection
    return render(request, 'detections.html')

