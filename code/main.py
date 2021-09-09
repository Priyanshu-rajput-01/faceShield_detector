import cv2
import mediapipe as mp
import time
import numpy as np

# CONSATNTS
HSV = [np.array([136, 87, 111], np.uint8), np.array([180, 255, 255], np.uint8),
       np.array([25, 52, 72], np.uint8)  , np.array([76, 255, 255],  np.uint8), 
       np.array([94, 80, 2], np.uint8)   , np.array([150, 255, 255], np.uint8)]


cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

################################################################################################################################################
def prespective(biggest,img):
    scale = 1
    pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
    highttrace = (biggest[1][1]-biggest[0][1])*scale
    widthtrace = (biggest[2][0]-biggest[0][0])*scale
    pts2 = np.float32([[0, 0],  [0, highttrace], [widthtrace, highttrace], [widthtrace, 0]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthtrace, highttrace))
    return imgWarpColored
################################################################################################################################################


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    id067=id203=(0,0)
    maxma = 0
    mask = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS,drawSpec, drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0, 255))
                if(id == 67):
                    id067 = (x,y)                    
                if(id == 293):
                    id203 = (x,y)
    if id067[1] > id203[1]:
        pt1=id067
        pt2=id203
        id067 = (pt1[0],pt2[1])
        id203 = (pt2[0],pt1[1])

    array = [[id067[0],id067[1]], [id067[0], id203[1]], [id203[0], id203[1]], [id203[0], id067[1]]]
    imageFrame = prespective(array,img)

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    kernal = np.ones((5, 5), "uint8")
    for i in range(3):
        mask.append(cv2.inRange(hsvFrame, HSV[(i*2)], HSV[(i*2) + 1]))   
        m = cv2.dilate(mask[i], kernal)

        red = cv2.bitwise_and(imageFrame, imageFrame, mask = m)

        contours, hierarchy = cv2.findContours(m,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 0, 255), 2)
                cv2.putText(img, "Face Shield detected", (id067[0] + x, id067[1] + y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                maxma = 1

    if(maxma==0):
        cv2.putText(img, "No shield detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0, 255))

    cv2.rectangle(img, id067, id203, (255, 255, 0), 3)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()