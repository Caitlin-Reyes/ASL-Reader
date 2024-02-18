import os
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd
import pyttsx3

## tried initalizing a camera class for log history and setting up tts 
class Camera:
    def __init__(self):
        self.log_history = ""
        self.engine = pyttsx3.init()
        self.cap = cv2.VideoCapture(0)
        self.w = int(self.cap.get(3))  # Width of the frame
        self.h = int(self.cap.get(4))  # Height of the frame

    def process_frame(self, frame):
        message = "Frame processed successfully."
        self.speak_and_log(message)

    def speak_and_log(self, text):
        print("Before saying:", text)
        self.engine.say(text)
        self.engine.runAndWait()
        print("After saying:", text)
        self.log_history += f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {text}\n"

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
        return frame


    def run(self):
        model = load_model('Sign_Language_MNIST.h5')
        mphands = mp.solutions.hands
        hands = mphands.Hands()
        mp_drawing = mp.solutions.drawing_utils

        while True:
            frame = self.capture_frame()

            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Escape hit, closing...") 
                self.engine.say("Escape hit, closing...") ## closing message audio 
                self.engine.runAndWait()
                break
            elif k % 256 == 32:
                analysisframe = frame
                showframe = analysisframe
                cv2.imshow("Frame", showframe)
                framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
                resultanalysis = hands.process(framergbanalysis)
                hand_landmarksanalysis = resultanalysis.multi_hand_landmarks

                if hand_landmarksanalysis:
                    for handLMsanalysis in hand_landmarksanalysis:
                        x_max = 0
                        y_max = 0
                        x_min = self.w
                        y_min = self.h
                        for lmanalysis in handLMsanalysis.landmark:
                            x, y = int(lmanalysis.x * self.w), int(lmanalysis.y * self.h)
                            if x > x_max:
                                x_max = x
                            if x < x_min:
                                x_min = x
                            if y > y_max:
                                y_max = y
                            if y < y_min:
                                y_min = y
                        y_min -= 20
                        y_max += 20
                        x_min -= 20
                        x_max += 20

                analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe, (28, 28))

                nlist = []
                rows, cols = analysisframe.shape
                for i in range(rows):
                    for j in range(cols):
                        k = analysisframe[i, j]
                        nlist.append(k)

                datan = pd.DataFrame(nlist).T
                colname = []
                for val in range(784):
                    colname.append(val)
                datan.columns = colname

                pixeldata = datan.values
                pixeldata = pixeldata / 255
                pixeldata = pixeldata.reshape(-1, 28, 28, 1)
                prediction = model.predict(pixeldata)
                predarray = np.array(prediction[0])
                letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
                letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
                predarrayordered = sorted(predarray, reverse=True)
                high1 = predarrayordered[0]
                high2 = predarrayordered[1]
                high3 = predarrayordered[2]
                for key, value in letter_prediction_dict.items(): ## this is where the tts says each predicted character and it's precentage (rounded)
                    if value == high1:
                        confidence_rounded = round(100 * value)
                        print("Predicted Character 1: ", key)
                        print('Confidence 1: ', 100 * value)
                        self.engine.say(f"Predicted Character 1: {key}. Confidence: {confidence_rounded}")
                        self.engine.runAndWait()
                    elif value == high2:
                        confidence_rounded = round(100 * value)
                        print("Predicted Character 2: ", key)
                        print('Confidence 2: ', 100 * value)
                        self.engine.say(f"Predicted Character 2: {key}. Confidence: {confidence_rounded}")
                        self.engine.runAndWait()
                    elif value == high3:
                        confidence_rounded = round(100 * value)
                        print("Predicted Character 3: ", key)
                        print('Confidence 3: ', 100 * value)
                        self.engine.say(f"Predicted Character 3: {key}. Confidence: {confidence_rounded}")
                        self.engine.runAndWait()


            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(framergb)
            hand_landmarks = result.multi_hand_landmarks
            if hand_landmarks:
                for handLMs in hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = self.w
                    y_min = self.h
                    for lm in handLMs.landmark:
                        x, y = int(lm.x * self.w), int(lm.y * self.h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    y_min -= 20
                    y_max += 20
                    x_min -= 20
                    x_max += 20
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

        self.cap.release()
        cv2.destroyAllWindows()

# camera instance & loop 
camera_instance = Camera()
camera_instance.run()
