from __future__ import division
import dlib
import cv2
import numpy as np
import calc


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
        if not ret:
            print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            return
        frame = cv2.flip(frame, 1)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = resize(grey, width=120)
        detects = detector(frame_resized, 1)

        for k, d in enumerate(detects):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            reqd_pts = [1, 8, 15, 17, 21, 22, 26, 27, 33, 36, 39, 42, 45, 51, 57, 62, 66, 71]
            new = [shape[i] for i in reqd_pts]
            for i, (x, y) in enumerate(new):
                cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
            cv2.putText(frame, str(calc.errors(new)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


landmark()
