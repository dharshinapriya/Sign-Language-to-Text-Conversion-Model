import numpy as np
import cv2
import os, sys
import time
import operator
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
from spellchecker import SpellChecker
import enchant
from keras.models import model_from_json, Sequential

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

class Application:
    def __init__(self):
        self.hs = enchant.Dict("en_US")
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Load main model
        try:
            with open("Models/model_new.json", "r") as json_file:
                self.model_json = json_file.read()
            self.loaded_model = model_from_json(self.model_json, custom_objects={'Sequential': Sequential})
            self.loaded_model.load_weights("Models/model_new.h5")
            print("Main model loaded")
        except Exception as e:
            print(f"Error loading main model: {e}")
            sys.exit()

        # Load DRU model
        try:
            with open("Models/model-bw_dru.json", "r") as json_file_dru:
                self.model_json_dru = json_file_dru.read()
            self.loaded_model_dru = model_from_json(self.model_json_dru, custom_objects={'Sequential': Sequential})
            self.loaded_model_dru.load_weights("Models/model-bw_dru.weights.h5")
            print("DRU model loaded")
        except Exception as e:
            print(f"Error loading DRU model: {e}")
            sys.exit()

        # Load TKDI model
        try:
            with open("Models/model-bw_tkdi.json", "r") as json_file_tkdi:
                self.model_json_tkdi = json_file_tkdi.read()
            self.loaded_model_tkdi = model_from_json(self.model_json_tkdi, custom_objects={'Sequential': Sequential})
            self.loaded_model_tkdi.load_weights("Models/model-bw_tkdi.weights.h5")
            print("TKDI model loaded")
        except Exception as e:
            print(f"Error loading TKDI model: {e}")
            sys.exit()

        # Load SMN model
# Load SMN model
        try:
            with open("Models/model-bw_smn.json", "r") as json_file_smn:
                self.model_json_smn = json_file_smn.read()
            self.loaded_model_smn = model_from_json(self.model_json_smn, custom_objects={'Sequential': Sequential})
            self.loaded_model_smn.load_weights("Models/model-bw_smn.weights.h5")
            print("SMN model loaded")
        except Exception as e:
            print(f"Error loading SMN model: {e}")
            sys.exit()


        print("All models loaded from disk")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)

        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"

        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            predicts = self.hs.suggest(self.word)

            self.bt1.config(text=predicts[0] if len(predicts) > 0 else "", font=("Courier", 20))
            self.bt2.config(text=predicts[1] if len(predicts) > 1 else "", font=("Courier", 20))
            self.bt3.config(text=predicts[2] if len(predicts) > 2 else "", font=("Courier", 20))

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        test_image = test_image.reshape(1, 128, 128, 1)

        result = self.loaded_model.predict(test_image)
        result_dru = self.loaded_model_dru.predict(test_image)
        result_tkdi = self.loaded_model_tkdi.predict(test_image)
        result_smn = self.loaded_model_smn.predict(test_image)

        prediction = {'blank': result[0][0]}
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'R', 'U']:
            prediction = {'D': result_dru[0][0], 'R': result_dru[0][1], 'U': result_dru[0][2]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {'D': result_tkdi[0][0], 'I': result_tkdi[0][1], 'K': result_tkdi[0][2], 'T': result_tkdi[0][3]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['M', 'N', 'S']:
            prediction1 = {'M': result_smn[0][0], 'N': result_smn[0][1], 'S': result_smn[0][2]}
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction1[0][0] if prediction1[0][0] == 'S' else prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
            self.ct['blank'] += 1
            if self.ct['blank'] > 60:
                for i in ascii_uppercase:
                    if abs(self.ct['blank'] - self.ct[i]) <= 20:
                        self.ct['blank'] = 0
                        for j in ascii_uppercase:
                            self.ct[j] = 0
                        return
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
        else:
            if len(self.str) > 16:
                self.str = ""
            self.blank_flag = 0
            self.word += self.current_symbol

    def action1(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 0:
            self.word = ""
            self.str += " " + predicts[0]

    def action2(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 1:
            self.word = ""
            self.str += " " + predicts[1]

    def action3(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 2:
            self.word = ""
            self.str += " " + predicts[2]

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

print("Starting Application...")
(Application()).root.mainloop()