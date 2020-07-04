import cv2
import numpy as np
import os
import faceDetection as fr
try:
    faces,faceID=fr.labels_for_training_data('path to training image in different folder with lable')
    recognizer=fr.train_classifier(faces,faceID)
    cv2.iamWrite.recognizer.save('traingData.yml')#saves training data
    recognizer=cv2.face.LBPHFaceRecognizer_create('training.yml')
    recognizer.read()
    name={0:"arun",1:"kholi"}#folder lable name
    print("capturing")
    cap=cv2.videoCapture(0)

    while True:
        test_img=cap.read()
        faces_detected,gray_img=fr.faceDetection(test_img)
        for(x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            resized_img=cv2.resize(test_img,(1000,700))
            cv2.imshow("face Detection",resized_img)
            cv2.waitKey(10)
            for face in faces_detected:
                (x,y,w,h)=face
                roi_gray=gray_img[y:y+w,x:x+h]
                label,confidence=recognizer.predict(roi_gray)
except Exception as e:
    [False,e],[None]
    print("confidence:",confidence)
    print("label",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    # if(confidence>39):
try:
    fr.put_text(test_img,predicted_name,x,y)
    resized_img=cv2.resize(test_img,(1000,700))
    cv2.imshow("resized image",resized_img)
    cv2.waitKey(10)
    cap.release()
    cv2.destroyAllWindows
except Exception as e:
    [False,e],[None]
    
