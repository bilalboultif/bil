import face_recognition
import cv2
import numpy as np
import os
import pickle

# Function to load known faces, usernames, and passwords from file
def load_known_faces():
    known_faces = {}
    if os.path.exists("known_faces.pkl"):
        with open("known_faces.pkl", "rb") as f:
            known_faces = pickle.load(f)
    return known_faces

# Function to save known faces, usernames, and passwords to file
def save_known_faces(known_faces):
    with open("known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)

# Function to register a new user with their face encoding, username, and password
def register_user(face_encoding, username, password):
    known_faces = load_known_faces()
    known_faces[username] = {"face_encoding": face_encoding, "password": password}
    save_known_faces(known_faces)
    print(f"User '{username}' registered successfully.")

# Function to authenticate a user based on their face encoding
def authenticate_user(face_encoding, tolerance=0.6):
    known_faces = load_known_faces()
    for username, data in known_faces.items():
        stored_face_encoding = data["face_encoding"]
        matches = face_recognition.compare_faces([stored_face_encoding], face_encoding, tolerance=tolerance)
        if matches[0]:
            password = data["password"]
            return username, password
    return None, None

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize variables
process_this_frame = True
registering_user = False
username = ""
password = ""

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame to save time
    if process_this_frame:
        if registering_user:
            # Draw text input boxes for username and password
            cv2.putText(frame, "Enter Username:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, username, (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Enter Password:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "*" * len(password), (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # Attempt to authenticate the user
                username, password = authenticate_user(face_encoding)
                if username is not None:
                    print(f"Authenticated as '{username}'. Password: {password}")
                else:
                    print("User not recognized.")

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Check if user is in registration mode
    if registering_user:
        # Capture face image when 'c' is pressed on the keyboard
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            # Take a snapshot of the current frame
            cv2.imwrite("temp_face.jpg", frame)

            # Load the captured image
            img = face_recognition.load_image_file("temp_face.jpg")
            new_face_encoding = face_recognition.face_encodings(img)
            if len(new_face_encoding) > 0:
                new_face_encoding = new_face_encoding[0]

                # Register the new user
                register_user(new_face_encoding, username, password)

                # Clear temporary file
                os.remove("temp_face.jpg")

                # Reset registration mode
                registering_user = False
                username = ""
                password = ""
            else:
                print("No face detected in the image. Try again.")

    # Hit 'r' on the keyboard to enter registration mode
    if cv2.waitKey(1) & 0xFF == ord('r'):
        registering_user = True

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
