from __future__ import division
import dlib
import cv2
import numpy as np

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((81, 2), dtype=dtype)
    
    for i in range(0, 81):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def landmark():
    cap = cv2.VideoCapture(0)

    path = "shape_predictor_81_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path)

    while True:

        ret, frame = cap.read()
        if ret == False:
            print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = resize(grey, width=120)
        detects = detector(frame_resized, 1)
        
        if len(detects) > 0:
            for k, d in enumerate(detects):
                shape = predictor(frame_resized, d)
                shape = shape_to_np(shape)
                new=[]
                new.append(shape[0])
                new.append(shape[7])
                new.append(shape[14])
                new.append(shape[16])
                new.append(shape[20])
                new.append(shape[21])
                new.append(shape[25])
                new.append(shape[26])
                new.append(shape[34])
                new.append(shape[35])
                new.append(shape[38])
                new.append(shape[41])
                new.append(shape[44])
                new.append(shape[50])
                new.append(shape[56])
                new.append(shape[61])
                new.append(shape[65])
                new.append(shape[70])

                for (x, y) in shape:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
    return new

coordinates= landmark()
print(coordinates)