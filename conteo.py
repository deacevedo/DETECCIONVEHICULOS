#Importanción de Librerías
import cv2
import numpy as np
import imutils
#Cargamos el video que vamos a usar.
cap = cv2.VideoCapture('autos.mp4')
#Creamos el background ademas del kernel
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#Contador de autos 
car_counter = 0

while True:
    #Creacion del frame
    ret, frame = cap.read()
    if ret == False: break
    #Se usa imutils.resize para redimencionar nuestro frame
    frame = imutils.resize(frame, width=640)

    #Se especifica el area que se va a analizar.
    area_pts = np.array([[330, 216], [frame.shape[1]-80, 216], [frame.shape[1]-80, 271], [330, 271]])

    # Se hace uso de una imagen auxiliar que sera el actuador 
    # para la deteccion del movimineto 
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(frame, frame, mask=imAux)    
    #obtenemos una imagen binaria del movimiento hay que tomar 
    #en cuenta que la extraccion de la mascara es a blanco y negro
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=5)

    #Ahora dibujamos lo contornos de las imagenes. 
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in cnts:
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 1)

        #Ahora determinamos si el auto cruza esta seccion que seria entre 440 y 460 que es
        #es el espacio que hay dentro de nuestro recuadro. 
        if 440 < (x + w) < 460:
            car_counter = car_counter + 1
            cv2.line(frame, (450, 216), (450, 271), (0, 255, 0), 3)


    #Visualizacion
    cv2.drawContours(frame, [area_pts], -1, (255, 0, 255), 2)
    cv2.line(frame, (450, 216), (450, 271), (0, 255, 255), 1)
    cv2.rectangle(frame, (frame.shape[1]-70, 215), (frame.shape[1]-5, 270), (0, 255, 0), 2)
    cv2.putText(frame, str(car_counter), (frame.shape[1]-55, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('image_area', image_area)
    cv2.imshow('fgmask', fgmask)
    
    k = cv2.waitKey(70) & 0xFF
    if k ==27:
        break
cap.release()
cv2.destroyAllWindows()
