import tensorflow as tf
import numpy as np
import cv2
import argparse

def load_and_prepare_image(image_path, img_size=(224, 224)):
    """
    Carica e prepara un'immagine per la predizione
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine da {image_path}")
    
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0  
    img = np.expand_dims(img, axis=0)  #aggiungo una dimensione perchè mi da errroe sulle shape, si aspetta anche la dimensione del batch
    return img

def predict_image(model, image_path):
    img = load_and_prepare_image(image_path)
    prediction = model.predict(img)

    print(prediction)
    probability = prediction[0][0]
    label = 1 if probability < 0.5 else 0
    class_label = "Reale" if label == 1 else "Generata da IA"
    return probability, class_label

def main():


    model = tf.keras.models.load_model("assignment3/modello 8 92%/modello_cnn.keras")
    print("Modello caricato correttamente")


    path= input()
    path="assignment3/"+path

    probability, label = predict_image(model, path)
    print(f"\nRisultato della classificazione:")
    print(f"- Probabilità: {probability:.4f}")
    print(f"- Classe predetta: {label}")


if __name__ == "__main__":
    main()