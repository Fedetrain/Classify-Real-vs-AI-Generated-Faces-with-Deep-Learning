import cv2 as cv
import argparse
import numpy as np
import dlib
import imageio.v2 as imageio  
import os




def inizializzaCascadeClassifier():

    #codice preso dalla documentazione di open cv
    #https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

    #Un parser è un componente di un programma che analizza e interpreta dati strutturati, convertendoli in un formato utilizzabile dal software
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')

    #aggiungo un argomento alla riga di comando chiamato --face_cascade e specifico il path dove è presente il file xml,
    #si trova installato nel mio enviroment venv
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
    args = parser.parse_args()

    # Inizializzazione dei classificatori
    face_cascade = cv.CascadeClassifier()

    # Caricamento dei classificatori
    if not face_cascade.load(cv.samples.findFile(args.face_cascade)):
        print('Errore nel caricamento del classificatore')
    return face_cascade



def get_image_from_cartella(cartella):
    #prende l'unica immagine dentro la cartella specificata
    files = os.listdir(cartella)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    image_path = os.path.join(cartella, images[0])
    return cv.imread(image_path)


def align_faces(image1, image2, face_cascade):

    #conversione in scla di grigi poiche lo vuole cosi
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))


    # in alcune immagini mi rilevava 2 volti, per comodità scelgo solo il rettangolo piu grande che dovrebbe essere quello giusto
    if len(faces1) > 1:
        faces1 = [max(faces1, key=lambda rect: rect[2] * rect[3])]
    if len(faces2) > 1:
        faces2 = [max(faces2, key=lambda rect: rect[2] * rect[3])]

    if len(faces1) == 0 or len(faces2) == 0:
        print("Errore: Non sono stati trovati volti in una o entrambe le immagini.")
        return None, None


    x1, y1, w1, h1 = faces1[0]
    x2, y2, w2, h2 = faces2[0]


    # prendo i punti per la trasformazione affine
    pts1 = np.float32([[x1, y1], [x1 + w1, y1], [x1, y1 + h1]])
    pts2 = np.float32([[x2, y2], [x2 + w2, y2], [x2, y2 + h2]])

    # calcolo la matrice di trasformazione affine
    M = cv.getAffineTransform(pts2, pts1)

    # applico la trasformazione all intera immagine
    immagine_allineata = cv.warpAffine(image2, M, (image2.shape[1], image2.shape[0]), borderMode=cv.BORDER_REPLICATE)

    return immagine_allineata




def get_landmark_triangoli(img, shape_predictor, face_cascade):

    height, width = img.shape[:2]

    #il classificatore vuole l immagine in scala di grigi
    img_scala_grigi = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #m trovo il rettangolo attorno al volto , i parametri li ho messi cosi perchè o sennò mi trovava piu rettangoli in un immagine
    #scaleFactor, controlla come viene scalata l'immagine durante il processo di rilevamento.
    rettangoli = face_cascade.detectMultiScale(img_scala_grigi, scaleFactor=1.1 , minNeighbors=5, minSize=(100, 100))

    # questa riga definisce un rettangolo che delimita l'area in cui verranno inseriti i punti per la triangolazione.
    # ​La riga di codice subdiv = cv.Subdiv2D((0, 0, width, height)) crea un oggetto Subdiv2D di OpenCV, utilizzato per eseguire una triangolazione di Delaunay
    # ho messo come vertici del triangolo questi perchè devo inserire anche i punti ai vertici dell immagine per avere i triangoli anche al di fuori dal volto


    #specifica l’area rettangolare in cui si costruirà la triangolazione.
    subdiv = cv.Subdiv2D((0, 0, width, height))  

    landmarks_list=list() 
    
    if len(rettangoli) == 0:
        print("Nessun volto rilevato.")
        return None, None, None
    elif len(rettangoli) >= 1:
        rettangoli = [max(rettangoli, key=lambda r: r[2] * r[3])]  

    x, y, w, h = rettangoli[0]

    #funzione per disegnare il rettangolo 
    #cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #per calcolare la triangolazione ho bisogno di utiliizare un oggetto rettangolo della libreria dlib, quindi ricalcolo 
    #il triangolo usando come vertici quelli trovati da OpenCV
    rettangolo_predictor_dlib = dlib.rectangle(x, y, x + w, y + h)
    landmarks = shape_predictor(img_scala_grigi, rettangolo_predictor_dlib)

    #landmark è un oggetto dlib.full_object_detection e non mi permette di aggiungere altri landmark manualmente, quindi lo converto il lista
    landmarks_list = list(landmarks.parts())
    landmarks_list.append(dlib.point(0, 0))
    landmarks_list.append(dlib.point(0, height - 1))
    landmarks_list.append(dlib.point(width - 1, 0))
    landmarks_list.append(dlib.point(width - 1, height - 1))
    

    #estraggo ogni coordinata x e y di ogni landmark utilizzando numpy e li aggiungo a subdiv per la triangolazione
    #li vuole per forza in float
    points = np.array([(pt.x, pt.y) for pt in landmarks_list],dtype=np.float32)
    subdiv.insert(points)
    
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    triangles_indici = []

    for t in triangles:
        #da array 1d ad array 2d
        pts = t.reshape(-1, 2)
        #cv.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1) 

    #PROBLEMA: subdiv.getTriangles() non garantisce l ordine dei triangoli, infatti non mi funzioanava
    #quindi per ciascun triangolo mi salvo la posizione (indice) dei landmark che contribuiscono ad ogni triangolo

    dizionario_punti_indici = {(pt.x, pt.y): i for i, pt in enumerate(landmarks_list)}

    #per ogni triangolo, utilizzo il dizionario per ottenere gli indici
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        triangle_indices = [dizionario_punti_indici[pt] for pt in pts]
        triangles_indici.append(triangle_indices)

    return triangles_indici, landmarks_list, rettangoli


def calcolaLandmarkIntermedi(landmark_img_src, landmark_img_dst):
    #mi calcolo i landmark intermedi usando la formula delle slide
    landmarks_intermedi=[]
    
    for i in range(numero_step):                               
        t = i / numero_step  # Fattore di interpolazione

        src_points = np.array([(pt.x, pt.y) for pt in landmark_img_src])
        dst_points = np.array([(pt.x, pt.y) for pt in landmark_img_dst])
        
        landmark_intermedio = (1 - t) * src_points + t * dst_points
        landmarks_intermedi.append(landmark_intermedio)

    return landmarks_intermedi


def transformazione(immagine,landamark_img_src, landmark_intermedi_img, triangoli_indici):
    rows, cols = immagine.shape[:2]

    #creo 2 matrici di coordinate usando meshgrid per mappare le posizione dei pixel nell immagine
    x_map, y_map = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    print(x_map)
    print(x_map.shape)

    frame_intermedi = []

    for landamark_step_corrispondente in landmark_intermedi_img:
        frame=immagine.copy()
        for tri_ind in triangoli_indici:
            #prendo i punti del trisngolo sorgente
            punti_triangolo_src = np.float32([[landamark_img_src[i].x, landamark_img_src[i].y] for i in tri_ind])

            #punti del triangolo in cui lo devo trnasformsre
            punti_triangolo_dst = np.float32([landamark_step_corrispondente[i] for i in tri_ind])

            # qua utilizzo una maschera. una matrice di tutti zeri
            mask = np.zeros((rows, cols), dtype=np.uint8)
            #disegno in bianco il triangolo originale all interno della maschera, 
            #questo mi servirà per prendere gli indici solo del triangolo per apllicare la tranformazione
            #piecewise
            cv.fillConvexPoly(mask,punti_triangolo_src.astype(np.int32),255)

            #metodo di opencv per calcolare una matrice di trandformazione dando in input 2 triangolki
            A = cv.getAffineTransform(punti_triangolo_src, punti_triangolo_dst)

            #la mia matrice deve essre 3x3 , quindi gli aggiungo un altra riga che è sempre 0,0,1
            A_homogeneous = np.vstack([A, [0, 0, 1]])
            #stiamo applicando un backward mapping, quindi si utilizza la matrice inversa
            A_inv = np.linalg.inv(A_homogeneous)
            #mi prendp gli indici di riga e colonna che individuano il triangolo nell immagine
            indici_riga_triangolo,indici_colonna_triangolo = np.where(mask == 255)

            #prodotto scalare della mia matrice inversa, SOLO con gli indici del triangolo, per calcolare le coordinate
            #dei pixel del triangolo transformato
            coords = np.dot(A_inv, np.vstack([indici_colonna_triangolo,indici_riga_triangolo, np.ones(indici_riga_triangolo.shape[0])]))
            
            # aggiorno nelle matrici delle coordinate, SOLO le nuove posizioni del triangolo tranformato
            x_map[indici_riga_triangolo, indici_colonna_triangolo] = coords[0, :].astype(np.float32)
            y_map[indici_riga_triangolo, indici_colonna_triangolo] = coords[1, :].astype(np.float32)

        frame = cv.remap(immagine, x_map, y_map, interpolation=cv.INTER_LINEAR)
        frame_intermedi.append(frame)

    return frame_intermedi


def blending_e_creazioneGif():
    blended_frames = []
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  
    video_out = cv.VideoWriter('blending_video.mp4', fourcc, 10, (frame11_12[0].shape[1], frame11_12[0].shape[0]))

    for i in range(numero_step):
        t = i / numero_step

        #cosi sto creadno il frame blended tra le due immagini
        blended_frame = cv.addWeighted(frame11_12[i], 1 - t, frame12_11[numero_step - 1 - i], t, 0)

        #converto da BGR a RGB per imageio
        blended_frame_rgb = cv.cvtColor(blended_frame, cv.COLOR_BGR2RGB)
        
        #aggiungo il frame alla lista per la GIF
        blended_frames.append(blended_frame_rgb)
        
        #scrivo il frame nel video
        video_out.write(blended_frame)

    # Salva la GIF
    imageio.mimsave('blending.gif', blended_frames, duration=0.2)
    
    # Rilascia il video writer per terminare il video
    video_out.release()



#il file shape_predictor_68_face_landmarks.dat è un file che contiene le coordinate dei 68 landmark del volto
#non è incluso direttamente ma l ho scaricato dal sito dlib http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
shape_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks/shape_predictor_68_face_landmarks.dat")
face_cascade=inizializzaCascadeClassifier()


CARTELLA_SRC = "img_src"
CARTELLA_DST = "img_dst"

image1_1=get_image_from_cartella(CARTELLA_SRC)
image1_2=get_image_from_cartella(CARTELLA_DST)


image1_1 = cv.resize(image1_1,dsize=(0,0),fx=0.6, fy=0.6)
image1_2 = cv.resize(image1_2,dsize=(0,0),fx=0.6, fy=0.6)

image1_2= align_faces(image1_1,image1_2,face_cascade)


numero_step=50


triangoli_indici_img1_1, landmark_img1_1, rettangolo_image1_1=get_landmark_triangoli(image1_1,shape_predictor, face_cascade)
triangoli_indici_img1_2, landmark_img1_2, rettangolo_image1_2=get_landmark_triangoli(image1_2,shape_predictor, face_cascade)



landmark_intermedi_11_12= calcolaLandmarkIntermedi(landmark_img1_1,landmark_img1_2)
landmark_intermedi_12_11= calcolaLandmarkIntermedi(landmark_img1_2,landmark_img1_1)

frame11_12=transformazione(image1_1,landmark_img1_1,landmark_intermedi_11_12,triangoli_indici_img1_1)
frame12_11=transformazione(image1_2,landmark_img1_2,landmark_intermedi_12_11,triangoli_indici_img1_2)


blending_e_creazioneGif()

