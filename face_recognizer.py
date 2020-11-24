import face_recognition
import cv2
import numpy as np
import os
import glob

faces_encodings = []
faces_names = []

cur_dir = os.getcwd()

path = os.path.join(cur_dir, './faces/')

list_files = [f for f in glob.glob(path + '*.jpg')]

number_files = len(list_files)

names = list_files.copy()

for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])

    names[i] = names[i].replace(cur_dir, "")
    faces_names.append(names[i])

face_locations = []
face_encod = []
face_names = []
process_frame = True

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:

    ret,frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = small_frame[:, :, ::-1]

    if process_frame:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encod = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encod:
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = 'unknown'

            face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]

            face_names.append(name)

    process_frame = not process_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

