import face_recognition
import cv2
import csv
import numpy as np
import os
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import random

app = Flask(__name__)
app.secret_key = 'super secret key'
loadmodel = create_model()
# Path to store photos
PHOTO_PATH = 'photos'
USERS_CSV = 'data.csv'

# Load liveness detection model
liveness_model = load_model('path_to_your_liveness_model.h5')  # Replace with your actual model path


def load_users():
    users = {}
    if os.path.exists(USERS_CSV):
        with open(USERS_CSV, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                users[row['username']] = row
    return users

def save_user(username, password, image_path):
    hashed_password = generate_password_hash(password)
    with open(USERS_CSV, mode='a', newline='') as outfile:
        fieldnames = ['username', 'password', 'image_path']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if os.stat(USERS_CSV).st_size == 0:
            writer.writeheader()
        writer.writerow({'username': username, 'password': hashed_password, 'image_path': image_path})

def update_user(username, password=None, image_path=None):
    users = load_users()
    if username in users:
        if password:
            users[username]['password'] = password
        if image_path:
            users[username]['image_path'] = image_path
        with open(USERS_CSV, mode='w', newline='') as outfile:
            fieldnames = ['username', 'password', 'image_path']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for user in users.values():
                writer.writerow(user)

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(PHOTO_PATH):
        if filename.endswith('.jpg'):
            img_path = os.path.join(PHOTO_PATH, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(filename.split('.')[0])
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

def is_liveness(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize frame to fit model input
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame.astype("float") / 255.0  # Normalize the frame
    frame_array = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    prediction = liveness_model.predict(frame_array)[0]  # Get model prediction
    return prediction[0] > 0.5  # Assume the model outputs probability of being live

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        if ret:
            if is_liveness(frame):
                img_path = os.path.join(PHOTO_PATH, f'{username}.jpg')
                cv2.imwrite(img_path, frame)
                video_capture.release()
                cv2.destroyAllWindows()

                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    encoding = encodings[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(username)
                    save_user(username, password, img_path)
                    return jsonify({'message': 'Registration successful'})
                else:
                    return jsonify({'message': 'No face found in the image'})
            else:
                video_capture.release()
                cv2.destroyAllWindows()
                return jsonify({'message': 'Failed liveness check'})
        else:
            video_capture.release()
            cv2.destroyAllWindows()
            return jsonify({'message': 'Failed to capture image'})

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        if ret:
            if is_liveness(frame):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        if name == username:
                            session['username'] = username
                            return redirect(url_for('welcome'))
        video_capture.release()
        cv2.destroyAllWindows()
        return render_template('login.html', error="Login failed. Face not recognized or failed liveness check.")
    
    return render_template('login.html')

@app.route('/welcome')
def welcome():
    if 'username' in session:
        username = session['username']
        return render_template('welcome.html', username=username)
    else:
        return redirect(url_for('login'))

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'POST':
        session.pop('username', None)
        return redirect(url_for('index'))
    return render_template('logout.html')

if __name__ == '__main__':
    if not os.path.exists(PHOTO_PATH):
        os.makedirs(PHOTO_PATH)
    app.run(debug=True)
