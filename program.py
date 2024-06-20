import face_recognition
import cv2
import csv
import numpy as np
import os
from datetime import datetime

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Load the sample pictures and learn how to recognize them
trump_image = face_recognition.load_image_file("photos/donald_trump.jpg")
trump_encoding = face_recognition.face_encodings(trump_image)[0]

bill_image = face_recognition.load_image_file("photos/bill_gate.jpg")
bill_encoding = face_recognition.face_encodings(bill_image)[0]

tom_image = face_recognition.load_image_file("photos/tom_cruise.jpg")
tom_encoding = face_recognition.face_encodings(tom_image)[0]

jeff_image = face_recognition.load_image_file("photos/jeff_bensose.jpg")
jeff_encoding = face_recognition.face_encodings(jeff_image)[0]

elon_image = face_recognition.load_image_file("photos/elon_mask.jpg")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

bilal_image = face_recognition.load_image_file("photos/bilal_boultif.jpg")
bilal_encoding = face_recognition.face_encodings(bilal_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    trump_encoding,
    bill_encoding,
    tom_encoding,
    jeff_encoding,
    elon_encoding,
    bilal_encoding
]

known_face_names = [
    "Donald Trump",
    "Bill Gate",
    "Tom Cruise",
    "Jeff Bensose",
    "Elon Mask",
    "Bilal Boultif"
]

students = known_face_names.copy()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Set the current time
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

# Prepare the CSV file
filename = "21-50-15.csv"  # Use a fixed filename to append data
f = open(filename, 'a+', newline='')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        if name in known_face_names:
            if name not in students:
                students.append(name)
                current_time = datetime.now().strftime("%H:%M:%S")  # Correct time capturing
                lnwriter.writerow([name, current_time])
                print(students)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
f.close()
