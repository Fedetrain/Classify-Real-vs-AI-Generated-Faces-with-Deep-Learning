import cv2
import argparse
import os
from tqdm import tqdm
import numpy as np
from skimage.feature import local_binary_pattern
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

def inizializzaCascadeClassifier():
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
    args = parser.parse_args()
    
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(args.face_cascade)):
        print('Errore nel caricamento del classificatore')
    return face_cascade


def ritaglia_volto(img, classificatore,percorso_file):
    img_grigia = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facce = classificatore.detectMultiScale(img_grigia, scaleFactor=1.1, minNeighbors=5)
    
    if len(facce) != 1:
        return None
    
    x, y, w, h = facce[0]
    volto = img[y:y+h, x:x+w]
    volto_ritagliato = cv2.resize(volto, (150, 150))
    
    return volto_ritagliato

def estrai_caratteristiche_lbp(img_grigia):

    lbp_p8 = local_binary_pattern(img_grigia, PUNTI_P8, RAGGIO, METODO_P8)
    print(lbp_p8)
    lbp_p256 = local_binary_pattern(img_grigia, PUNTI_P256, RAGGIO, METODO_P256)
    
    if(METODO_P256 == "default"):

        #density Fa in modo che l'istogramma venga normalizzato: Non ti dà il numero di pixel per ogni bin,Ma ti restituisce una distribuzione di probabilità 
        hist_p8, _ = np.histogram(lbp_p8, bins=np.arange(0, PUNTI_P8 + 3), range=(0, PUNTI_P8 + 2),density=True)

        hist_p256, _ = np.histogram(lbp_p256, bins=256, range=(0, 256),density=True)
    else:
        hist_p8, _ = np.histogram(lbp_p8, bins=256, range=(0, 256),density=True)

        hist_p256, _ = np.histogram(lbp_p256, bins=np.arange(0, PUNTI_P256 + 3), range=(0, PUNTI_P256 + 2),density=True)


    return hist_p8, hist_p256  

def elabora_dataset(cartella_input, file_output_p8, file_output_p256, is_fake=False, classificatore=None):
    caratteristiche_p8 = []
    caratteristiche_p256 = []
    nomi_file = []
    etichette = []
    soggetti = []

    """os.walk(cartella_input):genera una sequenza di tuple (root, dirs, files) per ogni directory nell'albero delle cartelle:

    root: percorso della directory corrente
    _: lista delle sottodirectory (usiamo _ perché non ci serve)
    files: lista dei file nella directory corrente

    tqdm(): volevo avere una specie di progress bar che mi dicesse il progresso del codice e tqmd mi 
    restitusice il numero di cartelle, cioè di soggetti che sono stati processati e anche il tempo trascorso.
    """
    
    for root, _, files in tqdm(os.walk(cartella_input), desc=f"Processando {'fake' if is_fake else 'real'}"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                percorso = os.path.join(root, file)
                id_soggetto = os.path.basename(root)
                
                img = cv2.imread(percorso)

                if img is None:
                    continue
                
                if not is_fake:
                    img_ritagliata = ritaglia_volto(img, classificatore,percorso)
                    if img_ritagliata is None:
                        continue
                    img = img_ritagliata  # aggiorna img con la versione ritagliata
                
                grigio = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist_p8, hist_p256 = estrai_caratteristiche_lbp(grigio)
                
                caratteristiche_p8.append(hist_p8)
                caratteristiche_p256.append(hist_p256)
                nomi_file.append(percorso)
                etichette.append(1 if is_fake else 0)
                soggetti.append(id_soggetto)

    
    with open(file_output_p8, 'wb') as f:
        pickle.dump({
            'nomi_file': nomi_file,
            'caratteristiche': caratteristiche_p8,
            'etichette': etichette,
            'soggetti': soggetti
        }, f)
    
    with open(file_output_p256, 'wb') as f:
        pickle.dump({
            'nomi_file': nomi_file,
            'caratteristiche': caratteristiche_p256,
            'etichette': etichette,
            'soggetti': soggetti
        }, f)

        print("Features real e fake salvate nei file pickle")
    return caratteristiche_p8, caratteristiche_p256, etichette, soggetti



def dividi_per_soggetti(caratteristiche, etichette, soggetti):

    soggetti = np.array(soggetti)
    caratteristiche = np.array(caratteristiche)
    etichette = np.array(etichette)

    # soggetti contiene piu valori duplicati poiche per un soggetto sono associate piu immagini
    soggetti_unici = np.unique(soggetti)

    #infatti il train test split lo faccio sui soggetti unici
    train_soggetti, temp_soggetti = train_test_split(soggetti_unici, test_size=0.4)
    val_soggetti, test_testsoggetti = train_test_split(temp_soggetti, test_size=0.5)
    
    #makera lunga quanto soggetti con a true le posizioni dei soggetti che si trovano nel train
    mask_train = np.isin(soggetti, train_soggetti)
    mask_val = np.isin(soggetti, val_soggetti)
    mask_test = np.isin(soggetti, test_testsoggetti)
    

    return {
        'X_train': caratteristiche[mask_train],
        'X_val': caratteristiche[mask_val],
        'X_test': caratteristiche[mask_test],
        'y_train': etichette[mask_train],
        'y_val': etichette[mask_val],
        'y_test': etichette[mask_test]
    }
def addestra_modello(modello, nome_modello, standardizzazione, X_train, y_train, X_val, y_val):
    print(f"\nModello: {nome_modello}")

    # Shuffle 
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

    if standardizzazione:
        print("standardizzazione attiva")
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_shuffled)
        X_val_std = scaler.transform(X_val)
    else:
        X_train_std = X_train_shuffled
        X_val_std = X_val

    modello.fit(X_train_std, y_train_shuffled)
    acc_train = modello.score(X_train_std, y_train_shuffled)
    acc_val = modello.score(X_val_std, y_val)

    return acc_train, acc_val


# Modifica nella funzione valuta_e_registra
def addestra_modelli(modelli, dati, tipo_lbp):
    for modello, nome in modelli:
        # prima addestro i modelli apllicando la standardizzazione
        acc_train, acc_val = addestra_modello(
            modello, f"{nome} ({tipo_lbp})",
            True, 
            dati['X_train'], dati['y_train'],
            dati['X_val'], dati['y_val'],
        )
        
        risultati['Modello'].append(nome)
        risultati['Tipo LBP'].append(tipo_lbp)
        risultati['Standardizzazione'].append(True)  
        risultati['Accuracy Train'].append(acc_train)
        risultati['Accuracy Val'].append(acc_val)
        
        # successivamento li addestro senza standardizzazione
        acc_train, acc_val = addestra_modello(
            modello, f"{nome} ({tipo_lbp})",
            False,  
            dati['X_train'], dati['y_train'],
            dati['X_val'], dati['y_val'],
        )
        
        risultati['Modello'].append(nome)
        risultati['Tipo LBP'].append(tipo_lbp)
        risultati['Standardizzazione'].append(False)  
        risultati['Accuracy Train'].append(acc_train)
        risultati['Accuracy Val'].append(acc_val)


def salva_modello_migliore():
    # Caricamento dei dataset
    with open("normale/dataset_p8.pkl", "rb") as f:
        dati_p8 = pickle.load(f)
    with open("normale/dataset_p256.pkl", "rb") as f:
        dati_p256 = pickle.load(f)
    
    # argmax restituisce l'indice con il valore maggiore
    indice_migliore = np.argmax(risultati['Accuracy Val'])
    tipo_lbp = risultati['Tipo LBP'][indice_migliore]

    # CAMBIARE VALORI QUA QUANDO FACCIO INVERTITO
    dati = dati_p8 if tipo_lbp == "uniform" else dati_p256
    
    nome_modello = risultati['Modello'][indice_migliore]
    usa_standardizzazione = risultati['Standardizzazione'][indice_migliore]
    print("standardizzato:", usa_standardizzazione)
    
    modello_dict = {
        "Random Forest": RandomForestClassifier(),
        "Regressione Logistica": LogisticRegression(max_iter=1000),
        "SVC Lineare": LinearSVC()
    }
    modello = modello_dict[nome_modello]

    X_train = dati['X_train']
    y_train = dati['y_train']
    X_val = dati['X_val']
    y_val = dati['y_val']
    X_test = dati['X_test']
    y_test = dati['y_test']

    if usa_standardizzazione:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        with open("normale/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    modello.fit(X_train, y_train)
    
    val_accuracy = modello.score(X_val, y_val)
    test_accuracy = modello.score(X_test, y_test)
    
    print(f"\nPerformance del modello migliore ({nome_modello} - {tipo_lbp}):")
    print(f"Accuracy Validation: {val_accuracy:.4f}")
    print(f"Accuracy Test: {test_accuracy:.4f}\n")
    
    def plot_confusion_matrix(y_true, y_pred, title):
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
        disp.ax_.set_title(title)
        plt.show()
        
    print("Matrice di confusione Validation Set:")
    plot_confusion_matrix(y_val, modello.predict(X_val), "Validation Set")
    
    print("Matrice di confusione Test Set:")
    plot_confusion_matrix(y_test, modello.predict(X_test), "Test Set")

    with open("normale/miglior_modello.pkl", "wb") as f:
        pickle.dump(modello, f)
    print("Modello salvato correttamente come 'miglior_modello.pkl'")





RAGGIO = 1
PUNTI_P8 = 8
PUNTI_P256 = 256
METODO_P8 = 'uniform'
METODO_P256 = 'default'



#inizializzo le cartelle  dove si trovano i dataset e i file pickle
cartella_real = "real/lfw-deepfunneled/lfw-deepfunneled"
cartella_fake = "fake/cropped_images"
file_real_p8 = "normale/features_real_p8.pkl"
file_real_p256 = "normale/features_real_p256.pkl"
file_fake_p8 = "normale/features_fake_p8.pkl"
file_fake_p256 = "normale/features_fake_p256.pkl"



#se non ho gia i file per l addestramento fai il processo di estrazione delle features
if not (os.path.exists("normal/dataset_p8.pkl") and os.path.exists("normal/dataset_p256.pkl")):
    classificatore = inizializzaCascadeClassifier()
    
    feat_real_p8, feat_real_p256, lab_real, sog_real = elabora_dataset(
        cartella_real, file_real_p8, file_real_p256, False, classificatore)
    
    feat_fake_p8, feat_fake_p256, lab_fake, sog_fake = elabora_dataset(
        cartella_fake, file_fake_p8, file_fake_p256, True, None)
    
    dati_p8 = dividi_per_soggetti(
        feat_real_p8 + feat_fake_p8,
        lab_real + lab_fake,
        sog_real + sog_fake
    )
    dati_p256 = dividi_per_soggetti(
        feat_real_p256 + feat_fake_p256,
        lab_real + lab_fake,
        sog_real + sog_fake
    )
    
    with open("normale/dataset_p8.pkl", "wb") as f:
        pickle.dump(dati_p8, f)
    
    with open("normale/dataset_p256.pkl", "wb") as f:
        pickle.dump(dati_p256, f)
else:
    with open("normale/dataset_p8.pkl", "rb") as f:
        dati_p8 = pickle.load(f)
            
    with open("normale/dataset_p256.pkl", "rb") as f:
        dati_p256 = pickle.load(f)


# Modelli da valutare
modelli = [
    (RandomForestClassifier(), "Random Forest"), 
    #max iter 1000 perche mi dava errore e mi diceva di scegliere un valore max da inserire
    (LogisticRegression(max_iter=1000), "Regressione Logistica"),
    
    (LinearSVC(), "SVC Lineare")
]

# dzionario per salvare i risultati
risultati = {
    'Modello': [],
    'Tipo LBP': [],
    'Standardizzazione': [],  
    'Accuracy Train': [],
    'Accuracy Val': [],
}


addestra_modelli(modelli, dati_p8, "uniform")
addestra_modelli(modelli, dati_p256, "default")

salva_modello_migliore()

df_risultati = pd.DataFrame(risultati)
print(df_risultati.to_string(index=False))



