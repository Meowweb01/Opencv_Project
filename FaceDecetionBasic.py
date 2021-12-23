import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0

mpFaceDetection = mp.solutions.face_detection
mp.mpDraw = mp.solutions.drawing_utils
FaceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detections in enumerate(results.detections):
            mp.mpDraw.draw_detection(img, detections)
            # print(id, detections)
            # print(detections.score)
            # print(detections.location_data.relative_bounding_box)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)  