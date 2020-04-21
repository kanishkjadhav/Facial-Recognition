import cv2
import facial_recog as fr

test_img=cv2.imread('/ce712_project/test_images/obama.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)

faces,faceID=fr.labels_for_training_data('/ce712_project/trainingImages')
face_recognizer=fr.train_classifier(faces,faceID)

name={0:"Priyanka",1:"Kangana",2:"Tara",3:"obama"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    if(confidence>40):
        continue 
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)
#for(x,y,w,h) in faces_detected:
 #   cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
 #   
resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows    



