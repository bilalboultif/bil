import face_recognition
import cv2
import csv
import numpy as np
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'super secret key'

# Path to store photos
PHOTO_PATH = 'photos'
USERS_CSV = 'data.csv'


def load_users():
    users = {}
    if os.path.exists(USERS_CSV):
        with open(USERS_CSV, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                users[row['username']] = row
    return users

# Function to save user to CSV
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

# Load known face encodings and names
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

# Helper function to capture and save camera image
def capture_and_save_image(username):
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    if ret:
        # Save the image to photos folder
        img_path = os.path.join(PHOTO_PATH, f'{username}.jpg')
        cv2.imwrite(img_path, frame)
        video_capture.release()
        # Optionally close specific window if needed
        # cv2.destroyWindow("Camera")  # Replace "Camera" with your window name
        return True, img_path
    else:
        video_capture.release()
        return False, None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        success, img_path = capture_and_save_image(username)
        
        if success:
            global known_face_encodings, known_face_names
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(username)
                
                # Save user data to CSV with hashed password
                save_user(username, hashed_password, img_path)
                
                return jsonify({'message': 'Registration successful'})
            else:
                return jsonify({'message': 'No face found in the image'})
        else:
            return jsonify({'message': 'Failed to capture image'})
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        video_capture = cv2.VideoCapture(0)
        try:
            ret, frame = video_capture.read()
            if ret:
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
                            session['username'] = username  # Store username in session
                            return redirect(url_for('welcome'))
        finally:
            video_capture.release()

        return render_template('login.html', error="Login failed. Face not recognized.")
    
    return render_template('login.html')

@app.route('/welcome')
def welcome():
    if 'username' in session:
        username = session['username']
        return render_template('welcome.html', username=username)
    else:
        return render_template('welcome.html')
    
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