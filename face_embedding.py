import cv2
from deepface import DeepFace
def deep_data_extract(img):
    face_data=None
    try:
        embedding = DeepFace.represent(img, model_name='Facenet')
        # print(embedding[0]["embedding"])
        face_data=embedding[0]["embedding"]
    except:
        pass
    return face_data

img=cv2.imread("./t.jpg")
face_data=deep_data_extract(img)
print(face_data)
