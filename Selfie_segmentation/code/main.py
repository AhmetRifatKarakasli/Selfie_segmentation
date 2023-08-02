import cv2
import mediapipe as mp
import numpy as np

mp_drawing=mp.solutions.drawing_utils
mp_selfie_segmentation=mp.solutions.selfie_segmentation

cap=cv2.VideoCapture(0)

with  mp_selfie_segmentation.SelfieSegmentation (model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        succes,image=cap.read()
        if not succes:
            break

        img_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result=selfie_segmentation.process(img_rgb)

        condition=np.stack((result.segmentation_mask,)*3,axis=-1)>0.1
        bg_image=np.zeros(image.shape,dtype=np.uint8)
        bg_image[:]=(0,255,0)
        output_image=np.where(condition,image,bg_image)

        cv2.imshow("pencere",output_image)

        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

cap.release()
cv2.destroyAllWindows()


