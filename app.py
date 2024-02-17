# from flask import Flask, render_template, request, redirect, url_for, flash
# import os
# import cv2
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import time
# from threading import Thread

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a secure key

# # Helper function to create folders if they don't exist
# def create_folders():
#     if not os.path.exists("data/faculty_details"):
#         os.makedirs("data/faculty_details")
#     if not os.path.exists("data/attendance"):
#         os.makedirs("data/attendance")

# # Helper function to capture and save face images
# def capture_face_images(username, name, id):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Create folder for the user if it doesn't exist
#     user_folder = os.path.join("data/faculty_details", f"{name}({id})")
#     if not os.path.exists(user_folder):
#         os.makedirs(user_folder)

#     # Start capturing face images
#     cap = cv2.VideoCapture(0)
#     face_count = 0
#     total_faces_to_capture = 10

#     # Display instructions
#     instruction_text = "Please rotate your face. Starting capture in 3 seconds."
#     print(instruction_text)

#     # Wait for 3 seconds before starting capture
#     time.sleep(3)

#     for _ in range(total_faces_to_capture):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y+h, x:x+w]
#             face_path = os.path.join(user_folder, f"face_{face_count}.jpg")
#             cv2.imwrite(face_path, face_roi)
#             face_count += 1

#         # Display the captured face in a small window
#         cv2.imshow('Capture Face', frame)

#         # Delay between captures (adjust as needed)
#         time.sleep(1)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     if face_count >= total_faces_to_capture:
#         return True
#     else:
#         return False


# # Route for the landing page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route for registration
# @app.route('/registration', methods=['GET', 'POST'])
# def registration():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username == 'hod' and password == 'cse':
#             name = request.form['name']
#             id = request.form['id']
#             if capture_face_images(username, name, id):
#                 flash('Registration successful!', 'success')
#             else:
#                 flash('Failed to capture enough face images!', 'error')
#             return redirect(url_for('index'))
#         else:
#             flash('Invalid credentials!', 'error')
#     return render_template('registration.html')

# def recognize_face():
#     # Load face cascade classifier
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Load LBPH recognizer for face recognition
#     recognizer = cv2.face.LBPHFaceRecognizer_create()

#     # Create a list to store recognized faces
#     recognized_faces = []

#     # Load images and labels for face recognition
#     (images, labels, names, id) = ([], [], {}, 0)
#     for (subdirs, dirs, files) in os.walk("data/faculty_details"):
#         for subdir in dirs:
#             names[id] = subdir
#             subject_path = os.path.join("data/faculty_details", subdir)
#             for filename in os.listdir(subject_path):
#                 path = os.path.join(subject_path, filename)
#                 label = id
#                 images.append(cv2.imread(path, 0))
#                 labels.append(int(label))
#             id += 1

#     # Train the LBPH recognizer
#     recognizer.train(images, np.array(labels))

#     # Capture video from webcam
#     cap = cv2.VideoCapture(0)

#     # Process recognized faces and store attendance details
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             # Predict the label of the face
#             roi_gray = gray[y:y+h, x:x+w]
#             id_, confidence = recognizer.predict(roi_gray)

#             # If the confidence is less than 100, the face is recognized
#             if confidence < 100:
#                 name = names[id_]

#                 # Get current date and time
#                 now = datetime.now()
#                 today_date = now.strftime('%Y-%m-%d')
#                 current_time = now.strftime('%H:%M:%S')

#                 # Process the recognized faces and store attendance details
#                 filename = f"data/attendance/{today_date}.xlsx"

#                 if not os.path.exists(filename):
#                     df = pd.DataFrame(columns=['ID', 'Name', 'In Time', 'Out Time', 'Time Spent'])
#                 else:
#                     df = pd.read_excel(filename)

#                 if name not in df['Name'].values:
#                     new_row = pd.DataFrame({'ID': [df.shape[0] + 1], 'Name': [name], 'In Time': [current_time], 'Out Time': [''], 'Time Spent': ['']})
#                     df = pd.concat([df, new_row], ignore_index=True)
#                 else:
#                     if pd.isna(df.loc[df['Name'] == name, 'Out Time'].iloc[0]):
#                         in_time = df.loc[df['Name'] == name, 'In Time'].iloc[0]
#                     else:
#                         in_time = df.loc[df['Name'] == name, 'In Time'].iloc[0]
#                         out_time = df.loc[df['Name'] == name, 'Out Time'].iloc[0]
#                         time_spent = datetime.strptime(str(out_time), '%H:%M:%S') - datetime.strptime(str(in_time), '%H:%M:%S')
#                         df.loc[df['Name'] == name, 'Time Spent'] = str(time_spent)
#                     recognized_faces.append({'Name': name, 'In Time': in_time, 'Out Time': current_time})
#                     if pd.isna(df.loc[df['Name'] == name, 'Out Time'].iloc[0]):
#                         df.loc[df['Name'] == name, 'Out Time'] = current_time

#                 df.to_excel(filename, index=False)
#             else:
#                 recognized_faces.append("Unknown")

#         cv2.imshow('Face Recognition', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # Check if faces are detected
#         if recognized_faces:
#             # Release the video capture
#             cap.release()
#             cv2.destroyAllWindows()

#             return recognized_faces

#     # Release the video capture
#     cap.release()
#     cv2.destroyAllWindows()

#     return []


# # Route for recognition
# @app.route('/recognition')
# def recognition():
#     recognized_faces = recognize_face()
#     if recognized_faces:
#         return render_template('recognition_result.html', recognized_faces=recognized_faces)
#     else:
#         flash('Failed to recognize faces!', 'error')
#         return redirect(url_for('index'))


# # if __name__ == '__main__':
# #     create_folders()
# #     app.run(debug=True)
# if __name__ == '__main__':
#     create_folders()
#     app.run(host='0.0.0.0', port=8080)

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
from datetime import datetime
import pandas as pd
import numpy as np
import time
from threading import Thread

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key

# Helper function to create folders if they don't exist
def create_folders():
    if not os.path.exists("data/faculty_details"):
        os.makedirs("data/faculty_details")
    if not os.path.exists("data/attendance"):
        os.makedirs("data/attendance")

# Helper function to capture and save face images
def capture_face_images(username, name, id):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create folder for the user if it doesn't exist
    user_folder = os.path.join("data/faculty_details", f"{name}({id})")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Start capturing face images
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        flash('Failed to open camera! Please grant camera permissions.', 'error')
        return False

    face_count = 0
    total_faces_to_capture = 10

    # Display instructions
    instruction_text = "Please rotate your face. Starting capture in 3 seconds."
    print(instruction_text)

    # Wait for 3 seconds before starting capture
    time.sleep(3)

    for _ in range(total_faces_to_capture):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_path = os.path.join(user_folder, f"face_{face_count}.jpg")
            cv2.imwrite(face_path, face_roi)
            face_count += 1

        # Display the captured face in a small window
        cv2.imshow('Capture Face', frame)

        # Delay between captures (adjust as needed)
        time.sleep(1)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if face_count >= total_faces_to_capture:
        return True
    else:
        return False


# Route for the landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route for registration
@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'hod' and password == 'cse':
            name = request.form['name']
            id = request.form['id']
            if capture_face_images(username, name, id):
                flash('Registration successful!', 'success')
            else:
                flash('Failed to capture enough face images!', 'error')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials!', 'error')
    return render_template('registration.html')

def recognize_face():
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load LBPH recognizer for face recognition
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Create a list to store recognized faces
    recognized_faces = []

    # Load images and labels for face recognition
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk("data/faculty_details"):
        for subdir in dirs:
            names[id] = subdir
            subject_path = os.path.join("data/faculty_details", subdir)
            for filename in os.listdir(subject_path):
                path = os.path.join(subject_path, filename)
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1

    # Train the LBPH recognizer
    recognizer.train(images, np.array(labels))

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        flash('Failed to open camera! Please grant camera permissions.', 'error')
        return []

    # Process recognized faces and store attendance details
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Predict the label of the face
            roi_gray = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi_gray)

            # If the confidence is less than 100, the face is recognized
            if confidence < 100:
                name = names[id_]

                # Get current date and time
                now = datetime.now()
                today_date = now.strftime('%Y-%m-%d')
                current_time = now.strftime('%H:%M:%S')

                # Process the recognized faces and store attendance details
                filename = f"data/attendance/{today_date}.xlsx"

                if not os.path.exists(filename):
                    df = pd.DataFrame(columns=['ID', 'Name', 'In Time', 'Out Time', 'Time Spent'])
                else:
                    df = pd.read_excel(filename)

                if name not in df['Name'].values:
                    new_row = pd.DataFrame({'ID': [df.shape[0] + 1], 'Name': [name], 'In Time': [current_time], 'Out Time': [''], 'Time Spent': ['']})
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    if pd.isna(df.loc[df['Name'] == name, 'Out Time'].iloc[0]):
                        in_time = df.loc[df['Name'] == name, 'In Time'].iloc[0]
                    else:
                        in_time = df.loc[df['Name'] == name, 'In Time'].iloc[0]
                        out_time = df.loc[df['Name'] == name, 'Out Time'].iloc[0]
                        time_spent = datetime.strptime(str(out_time), '%H:%M:%S') - datetime.strptime(str(in_time), '%H:%M:%S')
                        df.loc[df['Name'] == name, 'Time Spent'] = str(time_spent)
                    recognized_faces.append({'Name': name, 'In Time': in_time, 'Out Time': current_time})
                    if pd.isna(df.loc[df['Name'] == name, 'Out Time'].iloc[0]):
                        df.loc[df['Name'] == name, 'Out Time'] = current_time

                df.to_excel(filename, index=False)
            else:
                recognized_faces.append("Unknown")

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if faces are detected
        if recognized_faces:
            # Release the video capture
            cap.release()
            cv2.destroyAllWindows()

            return recognized_faces

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

    return []

# Route for recognition
@app.route('/recognition')
def recognition():
    recognized_faces = recognize_face()
    if recognized_faces:
        return render_template('recognition_result.html', recognized_faces=recognized_faces)
    else:
        flash('Failed to recognize faces!', 'error')
        return redirect(url_for('index'))


if __name__ == '__main__':
    create_folders()
    app.run(host='0.0.0.0', port=8080)
