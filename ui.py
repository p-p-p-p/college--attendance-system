import cv2
import numpy as np
import mediapipe as mp
import math

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon, model_selection=1)
    def findFace(self, img):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(self.imgRGB)
        faces = []
        cropped_faces = []
        if self.results.detections:
            for detection in self.results.detections:
                key_points = detection.location_data.relative_keypoints

                # Get the location of the left and right eyes
                left_eye = (
                    int(key_points[self.mpFaceDetection.FaceKeyPoint.LEFT_EYE].x * img.shape[1]),
                    int(key_points[self.mpFaceDetection.FaceKeyPoint.LEFT_EYE].y * img.shape[0])
                )
                right_eye = (
                    int(key_points[self.mpFaceDetection.FaceKeyPoint.RIGHT_EYE].x * img.shape[1]),
                    int(key_points[self.mpFaceDetection.FaceKeyPoint.RIGHT_EYE].y * img.shape[0])
                )
                delta_x = right_eye[0] - left_eye[0]
                delta_y = right_eye[1] - left_eye[1]
                alpha = np.degrees(np.arctan2(delta_y, delta_x))

                box = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                x1 = int(box.xmin * img.shape[1])
                y1 = int(box.ymin * img.shape[0])
                x2 = int((box.xmin + box.width) * iw)
                y2 = int((box.ymin + box.height) * ih)
                faces.append([x1, y1, x2, y2])
                # face_image = img[y1:y2, x1:x2]
                face_image = img[y1-50:y2+50, x1-50:x2+50]
                rotated_image = cv2.warpAffine(face_image,
                                cv2.getRotationMatrix2D((face_image.shape[1] / 2, 
                                face_image.shape[0] / 2), alpha+180, 1.0),
                                (face_image.shape[1], face_image.shape[0]))
                #resize face to train facenet model
                resized_image = cv2.resize(rotated_image, (160, 160))
                cv2.imshow("Cropped Face", resized_image)

                # Append the cropped face to the list
                cropped_faces.append(resized_image)
        return  faces,cropped_faces
def rgb_to_bgr(rgb_color):
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    return bgr_color
def drawBox(img, x1, y1, x2, y2, l=30, t=5, rt=1, text="Unknown", id=None,display_id=False,draw_rect=False,color=(2, 240, 228),text_color=(255,255,255)):
    # Define the sci-fi style font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    thickness = 2
    # color = (255, 255, 255)
    color=rgb_to_bgr(color)
    text_color=rgb_to_bgr(text_color)
    # Draw the ID of the detected person on top of the bounding box
    ((id_width, id_height), _) = cv2.getTextSize(str(id), font, fontScale=fontScale, thickness=thickness)
    id_offset_x = x1 + int((x2 - x1 - id_width) / 2)
    id_offset_y = y1 - 35
    if display_id:
        cv2.putText(img, str(id), (id_offset_x, id_offset_y+25), font, fontScale, text_color, thickness)
        # Draw the name of the detected person inside the bounding box
        ((text_width, text_height), _) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=thickness)
        text_offset_x = x1 + int((x2 - x1 - text_width) / 2)
        text_offset_y = y2 + 25
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale, text_color, thickness)
        # Draw box around face
    if draw_rect:
        cv2.rectangle(img, (x1, y1), (x2, y2), color,thickness=rt)
    t=t-3
    face_width = x2 - x1
    face_height = y2 - y1
    # l = int(l * min(face_width, face_height) / 100)-20
    
    # Draw top-left corner
    cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness=t)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness=t)
    # Draw top-right corner
    cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness=t)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness=t)
    # Draw bottom-left corner
    cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness=t)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness=t)
    # Draw bottom-right corner
    cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness=t)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness=t)
    return img

def white_overlay(img):
    white_img = np.ones_like(img) * 255
    alpha = 0.5
    result = cv2.addWeighted(img, alpha, white_img, 1-alpha, 0)
    return result

def fps_display(img,pTime,mid_x):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # text = f'FPS: {int(fps)}'
    text=str(int(fps))
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 3
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = img.shape[1] - text_size[0] - 20
    color=rgb_to_bgr((240, 0, 148))
    cv2.putText(img, text, (mid_x, 45), font, font_scale,color, thickness)
    return img,pTime

def overlay_icon(img, mid_x):
    logo = cv2.imread('./fps.png', cv2.IMREAD_UNCHANGED)
    logo = cv2.resize(logo, (50, 50))

    # Extract the alpha channel and convert to 8-bit unsigned integer
    alpha_channel = logo[:, :, 3]
    alpha_channel = cv2.convertScaleAbs(alpha_channel)

    # Remove the alpha channel from the logo and convert to BGR format
    logo = logo[:, :, :3]
    logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2BGR)

    # Create a mask from the alpha channel and resize it
    mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, (logo.shape[1], logo.shape[0]))

    # Overlay the logo on the image
    x = mid_x - 40
    y = 5
    overlay = img.copy()
    roi = overlay[y:y+logo.shape[0], x:x+logo.shape[1]]
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    roi_fg = cv2.bitwise_and(logo, logo, mask=mask)
    roi_combined = cv2.add(roi_bg, roi_fg)
    overlay[y:y+logo.shape[0], x:x+logo.shape[1]] = roi_combined

    return overlay

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
device = torch.device('cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def data_extract(img):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cropped = mtcnn(img_rgb)
        img_cropped = img_cropped.to(device)
        img_embedding = resnet(img_cropped.unsqueeze(0))
        embedding_np = img_embedding.detach().cpu().numpy()
        embedding_list = embedding_np.tolist()[0]
        # print(embedding_list)
    except:
        embedding_list=None
        
    return embedding_list


def style(frame,pTime):
    img=frame
    try:
        faces,cropped_faces= detector.findFace(frame)
        if len(faces) != 0: 
            data=data_extract(cropped_faces[0])
            print(data)
            x1,y1,x2,y2=faces[0]
            l = int(0.1 * math.sqrt((x2-x1)**2 + (y2-y1)**2))
            img=drawBox(frame, x1-5,y1-5,x2+5,y2+5, l=l, t=5, rt=1, text="Unknown", id=None,display_id=True,draw_rect=False)              
    except:
        img=frame
    overlay = white_overlay(img)
    x1 = 60
    y1 = 60
    x2 = frame.shape[1] - 60
    y2 = frame.shape[0] - 60
    mid_x = (frame.shape[1]) // 2
    roi = frame[y1:y2, x1:x2]
    overlay[y1:y2, x1:x2] = roi
    overlay=drawBox(overlay, x1+5, y1+5, x2-5, y2-5, l=30, t=6, rt=1, text="Unknown", id=None,display_id=False,draw_rect=True)
    overlay,pTime=fps_display(overlay,pTime,mid_x)
    overlay=overlay_icon(overlay,mid_x-20)
    return overlay,pTime




pTime = 0
import time
cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    overlay,pTime=style(frame,pTime)
    cv2.imshow('Webcam', overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()

