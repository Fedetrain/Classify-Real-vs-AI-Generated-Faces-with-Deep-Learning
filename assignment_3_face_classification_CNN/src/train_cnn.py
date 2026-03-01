
import tensorflow as tf
from tensorflow.keras import layers,regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import os
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse
import pickle

import numpy as np
from sklearn.model_selection import train_test_split




def elabora_dataset(img_size=(224, 224), 
                   max_img_ia1=6000, max_img_ia2=6000, 
                   max_img_reali_lfw=3000, max_img_reali_principale=9000):
    
    
    immagini = []  
    etichette = []  
    contatori = {
        'ia1': 0,
        'ia2': 0,
        'reali_lfw': 0,
        'reali_principale': 0,
    }

    # percorsi delle immagini
    percorso_ia2 = "assignment3/iaimage2"
    percorso_ia1 = "assignment3/iaimage1/thispersondoesnotexist"
    percorso_lfw = "assignment3/lfw-deepfunneled"
    percorso_real = "assignment3/realimage"


    # funzione per elaborare singola immagine
    def processa_immagine(img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        return img
    
    for root, _, files in tqdm(os.walk(percorso_ia2)):
        for file in files:
            if contatori['ia2'] >= max_img_ia2:
                break
                
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = processa_immagine(os.path.join(root, file))
                if img is not None:
                    immagini.append(img)
                    etichette.append(0)
                    contatori['ia2'] += 1

    for root, _, files in tqdm(os.walk(percorso_ia1)):
        for file in files:
            if contatori['ia1'] >= max_img_ia1:
                break
                
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = processa_immagine(os.path.join(root, file))
                if img is not None:
                    immagini.append(img)
                    etichette.append(0)
                    contatori['ia1'] += 1

    for root, _, files in tqdm(os.walk(percorso_lfw)):
        for file in files:
            if contatori['reali_lfw'] >= max_img_reali_lfw:
                break
                
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = processa_immagine(os.path.join(root, file))
                if img is not None:
                    immagini.append(img)
                    etichette.append(1)
                    contatori['reali_lfw'] += 1

    for root, _, files in tqdm(os.walk(percorso_real)):
        for file in files:
            if contatori['reali_principale'] >= max_img_reali_principale:
                break
                
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(file)
                img = processa_immagine(os.path.join(root, file))
                if img is not None:
                    immagini.append(img)
                    etichette.append(1)
                    contatori['reali_principale'] += 1

    # converto in array numpy per fare lo split
    immagini = np.array(immagini)
    etichette = np.array(etichette)

    # Divisione in train, validation e test
    #Con stratify=etichette, train_test_split garantisce che la stessa proporzione di classi venga mantenuta in entrambi i sottoinsiemi:
    X_train, X_temp, y_train, y_temp = train_test_split(immagini, etichette, test_size=0.3, stratify=etichette)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)


    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }



class SelfAttention(tf.keras.Layer):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # Dense layers per generare le matrici Query, Key e Value a partire dall'input
        self.query_dense = layers.Dense(embed_dim)  
        self.key_dense = layers.Dense(embed_dim)    
        self.value_dense = layers.Dense(embed_dim)  

    def call(self, inputs):
        # inputs  tensore di forma (batch_size, seq_len, embed_dim)

        # calcolo dei tensori Query, Key e Value
        query = self.query_dense(inputs)  
        key = self.key_dense(inputs)     
        value = self.value_dense(inputs)  

        # calcolo dell attenzione: prodotto scalare tra Query e Key trasposta
        scores = tf.matmul(query, key, transpose_b=True)  

        # scalatura  per evitare valori troppo grandi e migliorare la stabilità numerica
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_scores = scores / tf.math.sqrt(dim_key)

        # applicazione della softmax 
        weights = tf.nn.softmax(scaled_scores, axis=-1) 

        # calcolo dell'output 
        output = tf.matmul(weights, value)  

        return output  

    

def modello_convoluzionale(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = layers.Conv2D(16, (3,3), padding='valid')(inputs)  
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, (3,3), padding='valid')(x) 
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    

    x = layers.Conv2D(64, (3,3), padding='valid')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, (3,3), padding='valid',kernel_regularizer=regularizers.l2(2e-4))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, (3,3), padding='valid',kernel_regularizer=regularizers.l2(2e-4))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    
    #   LAYER DI ATTENZIONE
    x = layers.Reshape((-1, 128))(x)  
    x = SelfAttention(embed_dim=128)(x)
    
    x = layers.Flatten()(x)

    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)  

    
    x = layers.Dense(16, activation='relu')(x)  
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def addestra_modello(modello, X_train, y_train, X_val, y_val, epochs, batch_size):

    #La funzione modello.fit() in TensorFlow/Keras restituisce un oggetto di tipo History, che contiene le informazioni sull'addestramento del modello.

    #patience tiene conto di quante epoche senza miglioramenti aspettare prima di interrompere l addestramento,
    #restore_best_weight alla fine dell addestramento mi va a prendere i pesi dell epoca con la val_loss piu bassa
    early_stopping = EarlyStopping(
    monitor='val_loss',       
    patience=4,              
    verbose=1,
    restore_best_weights=True  
    )   

    hystory_training = modello.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],  
        verbose=1
    )
    return hystory_training

def plot_performance(hystory_training):
    plt.figure(figsize=(12, 4))
    
    # grafico che mostra accuracy in train e val 
    plt.subplot(1, 2, 1)
    plt.plot(hystory_training.history['accuracy'], label='Train Accuracy')
    plt.plot(hystory_training.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # grafico loss in train e val
    plt.subplot(1, 2, 2)
    plt.plot(hystory_training.history['loss'], label='Train Loss')
    plt.plot(hystory_training.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def test_modello(model, X_test, y_test):

    # uso  model.evaluate() per calcolare la loss e l'accuratezza sui dati di test.
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # model.predict() mi  genera le probabilità predette per ciascun esempio nel test set siccome sono tra 0 e 1, imposto una soglia a 0,5
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype("int32")
        
    # usso classification_report di scikit-learn per stampare:
    #precision: quanti dei positivi predetti sono corretti
    #recall: quanti dei veri positivi sono stati trovati
    #f1-score: media armonica tra precision e recall
    #support: numero di istanze per classe
    #macro avg:
    #Media delle metriche (precision, recall, f1) calcolata dando uguale peso a ogni classe.
    #weighted avg:
    #Media delle metriche pesata sul numero di campioni per classe

    print(classification_report(y_test, y_pred_classes))
    
    # matrice di confusione
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def save_pickle(data, filename="assignment3/24k_normalizzati.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)



if __name__ == "__main__":


    #parte commentata poichè riguarda l elaborazione dei dataset

    # dati = elabora_dataset(
    #     img_size=(224, 224)  
    # )

    # save_pickle(dati)
    
    print("prendo file pickle")
    
    with open("assignment3/24k_normalizzati224.pkl", 'rb') as f:
        dati = pickle.load(f)

    print("file pickle preso")
    modello = modello_convoluzionale(input_shape=(224, 224, 3))

    #bynary cross entropy perchè classificazione binaria
    modello.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    modello.summary()
    
    # Addestramento
    hystory_training = addestra_modello(
        modello,
        dati['X_train'], dati['y_train'],
        dati['X_val'], dati['y_val'],
        epochs=20,
        batch_size=32
    )
    
    plot_performance(hystory_training)
    
    # valutazione dekl modello sul test_set
    test_modello(modello, dati['X_test'], dati['y_test'])
    modello.save("modello_cnn.keras") 
