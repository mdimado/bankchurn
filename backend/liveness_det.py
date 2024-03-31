import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model(r'bankchurn/backend/fake_real_model_fin.h5')
def recog(image_path):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    probabilities = model.predict(img_array)
    predicted_class = np.argmax(probabilities)
    one_hot_encoded = np.zeros_like(probabilities)
    one_hot_encoded[0][predicted_class] = 1
    if all(one_hot_encoded[0] == [1., 0., 0., 0.]):
        return 'real'
    else:
        return 'fake'
face_casc=cv2.CascadeClassifier(r'bankchurn/backend/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_casc.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        face_roi=frame[y:y+h+100,x:x+100]
        cv2.imwrite('face.jpg',face_roi)
        cv2.putText(frame,org=(x,y),fontScale=1,color=(0,0,255),thickness=2,text=recog('face.jpg'),lineType=cv2.LINE_AA,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    cv2.imshow('web',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()