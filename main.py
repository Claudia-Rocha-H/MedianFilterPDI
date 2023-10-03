import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    imagen = cv2.imread("Noisy-Lena-image-The-noise-is-AWGN-with-standard-deviation-35.png")  # Imagen con ruido AWGN (Ruido Blanco Gaussiano)
    #Ventana 3x3
    #Si aumento el tamaño de la ventana se reducirá mas el ruido pero se perderan detalles de la figura
    #Si reduzco el tamaño de la ventana los detalles de la imagen se conservaran mejor pero se eliminara menos ruido
    ventana = 3
    alto, ancho, canales = imagen.shape
    print(f'Canales de color:  {canales}')
    print(f'Alto:  {alto}')
    print(f'Ancho:  {ancho}')
    # Creo una matriz vacía para la imagen filtrada
    imgFiltro = np.zeros((alto, ancho, canales), dtype=np.uint8)
    for i in range(alto):
        for j in range(ancho):
            for k in range(canales):
                # Obtengo region dentro de la ventana
                ventana_x1 = max(0, j - ventana // 2) #Vertice superior izquierdo coordenada x
                ventana_x2 = min(ancho, j + ventana // 2 + 1) #Vertice inferior derecho coordenada x
                ventana_y1 = max(0, i - ventana // 2) # Vertice superior izquierdo coordenada y
                ventana_y2 = min(alto, i + ventana // 2 + 1) #Vertice inferior derecho coordenada y

                # Obtengo la region de la ventana con el canal k
                region = imagen[ventana_y1:ventana_y2, ventana_x1:ventana_x2, k]

                # Saco promedio de los pixeles en la ventana para el canal k
                promedio = int(np.mean(region))

                # Asigno el valor promedio a la matriz de la imagen filtrada
                imgFiltro[i, j, k] = promedio
    #Filtro de media con cv2.filter2D
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    imgFiltroCV2 = cv2.filter2D(imagen, -1, kernel)

    # Figura con Matplotlib para mostrar las 3 imagenes en una misma ventana
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Mostrar la imagen original en la primera columna
    ax1.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    print(f'Tamaño imagen original:  {imagen.size}')
    ax1.set_title('Imagen Original')

    # Mostrar la imagen filtrada en la segunda columna
    ax2.imshow(cv2.cvtColor(imgFiltro, cv2.COLOR_BGR2RGB))
    print(f'Tamaño imagen con filtro:  {imgFiltro.size}')
    ax2.set_title('Imagen Filtrada')

    # Mostrar la imagen filtrada con CV2  en la tercera columna
    ax3.imshow(cv2.cvtColor(imgFiltroCV2, cv2.COLOR_BGR2RGB))
    ax3.set_title('Imagen Filtrada con cv2.Filter2D')
    plt.show()
except cv2.error as e:
    print(f'Error de OpenCV: {str(e)}')
except Exception as e:
    print(f'Error no relacionado con OpenCV: {str(e)}')