import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from time import time
import tkinter as tk

# Load the Haar Cascade Classifier and the emotion recognition model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define emotion colors
emotion_colors = {
    'Angry': (0, 0, 255),  # Red
    'Disgust': (0, 255, 0),  # Green
    'Fear': (0, 255, 255),  # Yellow
    'Happy': (255, 0, 0),  # Blue
    'Neutral': (255, 255, 0),  # Cyan
    'Sad': (0, 0, 128),  # Navy
    'Surprise': (128, 0, 128)  # Purple
}

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize a timer for 30 seconds (you can adjust the time)
start_time = time()
max_execution_time = 30

# Initialize a flag to check if the time limit is reached
time_limit_reached = False

# Initialize a list to store achieved emotions
achieved_emotions = []

# Set the confidence threshold for face detection
face_confidence_threshold = 0.3

while not time_limit_reached:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Perform face detection confidence check
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion and get the label
            prediction = emotion_model.predict(roi)[0]
            label_index = prediction.argmax()
            label = emotion_labels[label_index]
            confidence = prediction[label_index]

            # Filter faces based on confidence threshold
            if confidence >= face_confidence_threshold:
                # Display emotion with color
                label_position = (x, y - 10)
                label_color = emotion_colors.get(label, (0, 0, 0))
                cv2.putText(frame, f'{label} ({confidence:.2f})', label_position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                            label_color, 2)

                # Update the list of achieved emotions
                if label not in achieved_emotions:
                    achieved_emotions.append(label)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate elapsed time
    elapsed_time = time() - start_time

    # Display the timer
    timer_text = f'Time Left: {max_execution_time - int(elapsed_time)} seconds'
    cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check if the time limit is reached
    if elapsed_time >= max_execution_time:
        time_limit_reached = True

    cv2.imshow('Emotion Game', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Create a tkinter window for displaying achievements
achievements_window = tk.Tk()
achievements_window.title("Your Achievements")

# Create a label to display the heading
heading_label = tk.Label(achievements_window, text="Your emotions", font=("Arial", 54))
heading_label.pack()

# Create a label to display the achieved emotions
achievements_label = tk.Label(achievements_window, text="\n".join(achieved_emotions), font=("Arial", 24))
achievements_label.pack()

# Calculate missed emotions
missed_emotions = [emotion for emotion in emotion_labels if emotion not in achieved_emotions]

# Create a label to display the missed emotions
missed_label = tk.Label(achievements_window, text="Missed emotions", font=("Arial", 54))
missed_label.pack()

missed_label2 = tk.Label(achievements_window, text="\n".join(missed_emotions), font=("Arial", 24))
missed_label2.pack()

# Run the tkinter main loop for the achievements window
achievements_window.mainloop()
