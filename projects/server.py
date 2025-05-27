from flask import Flask, render_template, Response
import cv2
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

line_position = 250
count_in = 0
count_out = 0
previous_faces = []
face_timers = defaultdict(int)
frame_id = 0
DEBOUNCE_FRAMES = 30

def center_of_rect(x, y, w, h):
    return (x + w // 2, y + h // 2)

def generate_frames():
    global count_in, count_out, previous_faces, frame_id

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_centers = []

        for (x, y, w, h) in faces:
            cx, cy = center_of_rect(x, y, w, h)
            current_centers.append((cx, cy))
            person_id = f"{cx}_{cy}"

            # Draw rectangle and center
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Skip recently counted faces
            if frame_id - face_timers[person_id] < DEBOUNCE_FRAMES:
                continue

            for (pcx, pcy) in previous_faces:
                if pcy < line_position and cy >= line_position:
                    count_in += 1
                    face_timers[person_id] = frame_id
                elif pcy > line_position and cy <= line_position:
                    count_out += 1
                    face_timers[person_id] = frame_id

        previous_faces = current_centers

        # Draw line and counts
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), 2)
        cv2.putText(frame, f'In: {count_in}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Out: {count_out}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame to JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html',
                           in_count=count_in,
                           out_count=count_out,
                           time=datetime.now())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
