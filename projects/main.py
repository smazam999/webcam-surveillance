import cv2
import os
import time

# Create folder to save face images (optional)
os.makedirs("face_data", exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parameters for people counting
line_position = 250
count_in = 0
count_out = 0
previous_faces = []

def center_of_rect(x, y, w, h):
    return (x + w//2, y + h//2)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_centers = []

    for (x, y, w, h) in faces:
        cx, cy = center_of_rect(x, y, w, h)
        current_centers.append((cx, cy))

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Draw center point
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Save face image
        face_img = frame[y:y+h, x:x+w]
        filename = f"face_data/face_{int(time.time())}.jpg"
        cv2.imwrite(filename, face_img)

    # Compare with previous frame centers for simple line crossing logic
    for pc in previous_faces:
        for cc in current_centers:
            if pc[1] < line_position and cc[1] >= line_position:
                count_in += 1
            elif pc[1] > line_position and cc[1] <= line_position:
                count_out += 1

    previous_faces = current_centers

    # Draw counting line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), 2)
    cv2.putText(frame, f'In: {count_in}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Out: {count_out}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Webcam Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
