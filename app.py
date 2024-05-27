from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import time

app = Flask(__name__)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary to translate emotions to Spanish
emotions_translation = {
    'angry': 'enojado',
    'disgust': 'disgustado',
    'fear': 'miedo',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'sorprendido',
    'neutral': 'neutral'
}

def detect_emotion(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # White color for the rectangle

        # Draw mesh of points on face
        for i in range(x, x + w, 10):
            for j in range(y, y + h, 10):
                cv2.circle(frame, (i, j), 1, (255, 0, 0), -1)
                
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion and its confidence
        dominant_emotion = result[0]['dominant_emotion']
        emotion_confidence = result[0]['emotion'][dominant_emotion]  # Get confidence score

        # Translate emotion to Spanish
        emotion_spanish = emotions_translation[dominant_emotion]

        # Draw rectangle around face and label with predicted emotion and confidence
        label_emotion = f"{emotion_spanish}"
        label_confidence = f"{emotion_confidence:.2f}%"
        
        # Draw the labels on the frame with smaller font size
        cv2.putText(frame, label_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red color for emotion
        cv2.putText(frame, label_confidence, (x + w - 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green color for confidence

    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            frame = detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            time.sleep(0.05)  # Delay of 0.05 seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False)
