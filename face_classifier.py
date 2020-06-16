import face_recognition
import cv2
import face_recognition_knn
import numpy as np

def classify_face(face_img):
    face_img = cv2.resize((face_img), (48, 48))
    try:
        encoded_face = face_recognition.face_encodings(face_img)
        if len(encoded_face) > 0:
            encoded_face = face_recognition.face_encodings(face_img)[0]
    except Exception as e:
        print(e)
        return ["Cannot encode"]

    name = face_recognition_knn.predict([encoded_face], model_path="trained_knn_model.clf")

    return name

if __name__ == '__main__':
    face_img = cv2.imread('face_unknown/Rose4.png', 1)
    print(classify_face(face_img)[0])