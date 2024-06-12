import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk


model_path = 'C:/Users/George/Desktop/PORTOFOLIU CALCUL/PentruChestionar/PentruTraining/PentruChestionare4.keras'

if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model file not found: {model_path}')

model = load_model(model_path)

labels = [
    "Limită de viteză (20km/h)", "Limită de viteză (30km/h)", "Limită de viteză (50km/h)",
    "Limită de viteză (60km/h)", "Limită de viteză (70km/h)", "Limită de viteză (80km/h)",
    "Sfârșit limită de viteză (80km/h)", "Limită de viteză (100km/h)", "Limită de viteză (120km/h)",
    "Interzis depășirea", "Interzis depășirea pentru vehicule peste 3.5 tone", "Drept de trecere la următoarea intersecție",
    "Drum cu prioritate", "Cedează trecerea", "Oprire", "Acces interzis", "Interzis vehicule peste 3.5 tone",
    "Acces interzis", "Atenție generală", "Curbă periculoasă la stânga", "Curbă periculoasă la dreapta",
    "Curbe duble", "Drum denivelat", "Drum alunecos", "Drum îngustat pe dreapta", "Lucrări pe drum",
    "Semafor", "Trecere de pietoni", "Trecere de copii", "Traversare biciclete", "Atenție gheață/zăpadă",
    "Traversare animale sălbatice", "Sfârșit toate limitele de viteză și depășire", "Obligatoriu la dreapta înainte",
    "Obligatoriu la stânga înainte", "Înainte", "Înainte sau la dreapta", "Înainte sau la stânga",
    "Ține dreapta", "Ține stânga", "Sens giratoriu obligatoriu", "Sfârșit interdicție de depășire",
    "Sfârșit interdicție de depășire pentru vehicule peste 3.5 tone"
]


def preprocess_image(img_path, img_size=(32, 32)):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def load_and_predict():
    img_path = filedialog.askopenfilename()
    if not img_path:
        return
    
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = labels[predicted_class]

    img_display = Image.open(img_path)
    img_display = img_display.resize((150, 150), Image.Resampling.LANCZOS)
    img_display = ImageTk.PhotoImage(img_display)
    
    result_label.config(text=f'Predicted Class: {predicted_label}')
    canvas.create_image(75, 75, image=img_display)
    canvas.image = img_display


root = Tk()
root.title("Traffic Sign Recognition")
root.geometry("400x400")

Label(root, text="Traffic Sign Recognition", font=("Helvetica", 16)).pack(pady=20)

Button(root, text="Load and Predict Image", command=load_and_predict).pack(pady=10)

result_label = Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=20)

canvas = Canvas(root, width=150, height=150)
canvas.pack(pady=20)

root.mainloop()
