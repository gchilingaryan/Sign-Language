import cv2
import numpy as np
from keras import models

MODEL_NAME = "sign-language.h5"

model = models.load_model(MODEL_NAME)
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    cv2.rectangle(img, (100, 100), (600, 600), (255, 0, 0), 2)
    cropped = img[100:600, 100:600]
    resized = (cv2.cvtColor(cv2.resize(cropped, (28, 28)), cv2.COLOR_RGB2GRAY)) / 255.0
    data = resized.reshape(-1, 28, 28, 1)
    model_out = model.predict([data])[0]
    label = np.argmax(model_out)

    if max(model_out) > 0.9:
        if label == 0:
            letter = "A"
        elif label == 1:
            letter = "B"
        elif label == 2:
            letter = "C"
        elif label == 3:
            letter = "D"
        elif label == 4:
            letter = "E"
        elif label == 5:
            letter = "F"
        elif label == 6:
            letter = "G"
        elif label == 7:
            letter = "H"
        elif label == 8:
            letter = "I"
        elif label == 10:
            letter = "K"
        elif label == 11:
            letter = "L"
        elif label == 12:
            letter = "M"
        elif label == 13:
            letter = "N"
        elif label == 14:
            letter = "O"
        elif label == 15:
            letter = "P"
        elif label == 16:
            letter = "Q"
        elif label == 17:
            letter = "R"
        elif label == 18:
            letter = "S"
        elif label == 19:
            letter = "T"
        elif label == 20:
            letter = "U"
        elif label == 21:
            letter = "V"
        elif label == 22:
            letter = "W"
        elif label == 23:
            letter = "X"
        elif label == 24:
            letter = "Y"

        print letter

    cv2.imshow('img', img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

camera.release()

