import cv2
import numpy as np
import argparse
from skimage.feature import local_binary_pattern
import os
import pickle


RAGGIO = 1
PUNTI_P8 = 8
PUNTI_P256 = 256
METODO_P8 = 'uniform'
METODO_P256 = 'uniform'  

def inizializzaCascadeClassifier():
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
    args = parser.parse_args()
    
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(args.face_cascade)):
        print('Errore nel caricamento del classificatore')
    return face_cascade

def estrai_caratteristiche_lbp(img_grigia, tipo_lbp='P8'):
    if tipo_lbp == 'P8':
        lbp = local_binary_pattern(img_grigia, PUNTI_P8, RAGGIO, METODO_P8)
        hist, _ = np.histogram(lbp, bins=np.arange(0, PUNTI_P8 + 3), range=(0, PUNTI_P8 + 2), density=True)
    else: 
        lbp = local_binary_pattern(img_grigia, PUNTI_P256, RAGGIO, METODO_P256)
        hist, _ = np.histogram(lbp, bins=np.arange(0, PUNTI_P256 + 3), range=(0, PUNTI_P256 + 2), density=True)
    
    return hist

def rileva_e_ritaglia_volto(img, fare_resize=True):
    img_grigia = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facce = face_cascade.detectMultiScale(img_grigia, scaleFactor=1.1, minNeighbors=5)

    if len(facce) != 1:
        print("Volti rilevati:", len(facce))
        return None

    x, y, w, h = facce[0]
    volto = img[y:y + h, x:x + w]
    
    if fare_resize:
        volto = cv2.resize(volto, (150, 150))

    return volto

def classifica_immagine(percorso_img, tipo_lbp, fare_resize=True):
    img = cv2.imread(percorso_img)
    
    if img is None:
        print("Errore nel caricamento dell'immagine.")
        return

    img = rileva_e_ritaglia_volto(img, fare_resize)
    if img is None:
        return

    img_grigia = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = estrai_caratteristiche_lbp(img_grigia, tipo_lbp)


    if tipo_lbp == 'P8':
        with open("normale/miglior_modello.pkl", "rb") as f:
            modello = pickle.load(f)
        scaler_path = "normale/scaler.pkl"
    else:  
        with open("lbpinvertiti/miglior_modello.pkl", "rb") as f:
            modello = pickle.load(f)
        scaler_path = "lbpinvertiti/scaler.pkl"




    usa_scaler = os.path.exists(scaler_path)

    if usa_scaler:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None

    #faccio il reshape perche il modello si aspetta input 2D, come nell addestramento
    #mi è spuntato un errore in cui mi diceva di inserirla e mi spiegava il perchè
    features = hist.reshape(1, -1)
    if scaler:
        features = scaler.transform(features)
    pred = modello.predict(features)

    print(f"Tipo LBP utilizzato: {tipo_lbp}")
    print("Resize applicato:", "Sì" if fare_resize else "No")
    print("Classe predetta:", "FAKE" if pred[0] == 1 else "REALE")

face_cascade = inizializzaCascadeClassifier()

print("inserisci il percorso dell immagine")
percorso_img=input()


#IMPORTANTE: SE UTILIZZO UN IMMAGINE FAKE DEL DATASET NON DEVO FARE IL RESIZE O SENNò NON RIESCE A CLASSIFICARLO BENE
#SONO GIA' 150X150

classifica_immagine(percorso_img, tipo_lbp='P8', fare_resize=True)
classifica_immagine(percorso_img, tipo_lbp='P256', fare_resize=True)

