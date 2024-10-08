import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# Lista de categorías para la interfaz (sin 'cafe verde')
categories = ['brocado', 'verde', 'vinagre', 'marron vinagre']

# Lista completa de categorías para el modelo (incluyendo 'cafe verde')
categories_model = ['brocado', 'verde', 'vinagre', 'cafe verde', 'marron vinagre']

# Diccionario para contar las imágenes procesadas por categoría (sin 'cafe verde')
imagenes_procesadas = {categoria: 0 for categoria in categories}

# Cargar el modelo CNN preentrenado y modificar el optimizador
modelo_path = "modelo_cafe_caracteristicas_cnn.keras"
if os.path.exists(modelo_path):
    modelo = load_model(modelo_path)
    optimizer = Adam(learning_rate=0.001)
    modelo.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print("Modelo CNN cargado y optimizador ajustado con éxito")
else:
    raise FileNotFoundError(f"No se encontró el archivo del modelo CNN en la ruta {modelo_path}")

# Obtener el tamaño de entrada esperado por el modelo CNN
input_shape = modelo.input_shape
IMG_SIZE = (input_shape[1], input_shape[2])

# Inicializar la lista de granos y el índice del grano actual
granules = []
idx_grano = 0
datos_corregidos = {'imagenes': [], 'etiquetas': []}
predicciones = []
etiquetas_reales = []

# Función para mostrar la imagen del grano y la predicción
def mostrar_grano(grano_idx):
    global current_image, img_label, granules

    if grano_idx < len(granules):
        grano_img, categoria_predicha = granules[grano_idx]

        # Convertir imagen de OpenCV (BGR) a PIL para usarla en Tkinter
        img_rgb = cv2.cvtColor(grano_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((400, 400))
        current_image = ImageTk.PhotoImage(img_pil)

        if img_label is None:
            img_label = tk.Label(frame_imagen, image=current_image, bg="#000000")
            img_label.grid(row=0, column=0, padx=10, pady=10)
        else:
            img_label.config(image=current_image)

        # Mostrar la categoría predicha (omitimos 'cafe verde' en la interfaz)
        categoria_interfaz = categories_model[categoria_predicha]
        if categoria_interfaz == 'cafe verde':
            categoria_interfaz = 'verde'  # Cambiar visualización de 'cafe verde' a 'verde' en la interfaz
        
        # Incrementar el contador de imágenes procesadas para la categoría correspondiente
        if categoria_interfaz in imagenes_procesadas:
            imagenes_procesadas[categoria_interfaz] += 1

        correccion_var.set(categoria_interfaz)
        prediccion_label.set(f"Predicción: {categoria_interfaz}")

# Función para detectar y clasificar granos de café en una imagen
def detectar_y_clasificar_granos(image_path):
    global granules, idx_grano
    imagen = cv2.imread(image_path)
    if imagen is None:
        print(f"No se pudo cargar la imagen en la ruta {image_path}")
        return

    # Procesamiento de la imagen (redimensionar, aplicar máscara, etc.)
    altura, ancho = 700, 1100
    imagen = cv2.resize(imagen, (ancho, altura))

    # Convertir a HSV y aplicar máscaras (ajustado)
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    color_min = np.array([0, 30, 50], np.uint8)
    color_max = np.array([45, 255, 255], np.uint8)
    
    # Aplicar máscara
    mascara_total = cv2.inRange(imagen_hsv, color_min, color_max)
    
    # Limpieza de la máscara usando operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)
    mascara_total = cv2.erode(mascara_total, kernel, iterations=2)
    mascara_total = cv2.dilate(mascara_total, kernel, iterations=3)
    
    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara_total)

    # Convertir a escala de grises y detectar contornos
    gris = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (7, 7), 0)
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 3)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    granules.clear()
    for idx, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if area > 500 and perimetro > 0:
            circularidad = 4 * np.pi * area / (perimetro * perimetro)
            if circularidad > 0.5:  # Filtrar por forma más circular
                x, y, w, h = cv2.boundingRect(contorno)
                grano_img = imagen[y:y+h, x:x+w]
                grano_resized = cv2.resize(grano_img, IMG_SIZE)

                # Normalizar la imagen (escala de 0 a 1)
                grano_resized = grano_resized.astype('float32') / 255.0
                
                # Expandir dimensiones para hacerla compatible con la entrada de la CNN
                grano_resized = np.expand_dims(grano_resized, axis=0)

                # Realizar predicción
                prediccion = modelo.predict(grano_resized)
                
                # Mostrar las probabilidades predichas para cada clase
                print(f"Probabilidades predichas para el grano {idx + 1}: {prediccion}")

                # Obtener la clase con mayor probabilidad
                categoria_predicha = np.argmax(prediccion, axis=1)[0]

                granules.append((grano_img, categoria_predicha))

                # Almacenar predicciones y etiquetas reales (para la matriz de confusión)
                predicciones.append(categoria_predicha)
                etiquetas_reales.append(categoria_predicha)

    idx_grano = 0
    mostrar_grano(idx_grano)

# Función para seleccionar una imagen desde la interfaz
def seleccionar_imagen():
    file_path = filedialog.askopenfilename(
        title="Seleccionar Imagen",
        filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        detectar_y_clasificar_granos(file_path)

# Función para pasar al siguiente grano
def siguiente_grano():
    global granules, idx_grano
    if granules:
        idx_grano = (idx_grano + 1) % len(granules)
        mostrar_grano(idx_grano)

# Función para guardar la corrección
def guardar_correccion():
    global granules, idx_grano
    if granules:
        grano_img, categoria_predicha = granules[idx_grano]

        # Obtener la categoría corregida del menú desplegable (sin 'cafe verde')
        categoria_correcta = correccion_var.get()
        categoria_idx = categories_model.index(categoria_correcta)

        grano_resized = cv2.resize(grano_img, IMG_SIZE).astype('float32') / 255.0
        datos_corregidos['imagenes'].append(grano_resized)
        datos_corregidos['etiquetas'].append(categoria_idx)

        print(f"Corrección guardada: {categoria_correcta}")
        siguiente_grano()

# Función para reentrenar el modelo con los datos corregidos y luego calcular métricas
def reentrenar_modelo():
    if datos_corregidos['imagenes']:
        # Convertir los datos a arrays
        x_train = np.array(datos_corregidos['imagenes'])
        y_train = np.array(datos_corregidos['etiquetas'])

        # One-hot encoding de las etiquetas con 5 clases (incluyendo "cafe verde")
        y_train_one_hot = to_categorical(y_train, num_classes=5)

        # Reentrenar el modelo
        print("Reentrenando el modelo con los datos corregidos...")
        history = modelo.fit(x_train, y_train_one_hot, epochs=5, verbose=1)

        # Guardar el modelo reentrenado
        modelo.save(modelo_path)
        print(f"Modelo reentrenado y guardado con éxito en {modelo_path}")

        # Predicciones después del reentrenamiento
        predicciones_despues = np.argmax(modelo.predict(x_train), axis=1)
        print(f"Predicciones después del reentrenamiento: {predicciones_despues}")

        # Mostrar métricas después del reentrenamiento
        mostrar_metricas(y_train, predicciones_despues)
    else:
        print("No hay correcciones guardadas para reentrenar.")

# Función para generar y mostrar la matriz de confusión
def mostrar_matriz_confusion():
    if predicciones and etiquetas_reales:
        # Generar la matriz de confusión
        matriz_confusion = confusion_matrix(etiquetas_reales, predicciones)

        # Visualizar la matriz de confusión usando matplotlib
        plt.figure(figsize=(10, 7))
        plt.imshow(matriz_confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusión')
        plt.colorbar()
        tick_marks = np.arange(len(categories_model))
        plt.xticks(tick_marks, categories_model, rotation=45)
        plt.yticks(tick_marks, categories_model)

        # Etiquetas
        fmt = 'd'
        thresh = matriz_confusion.max() / 2.
        for i, j in np.ndindex(matriz_confusion.shape):
            plt.text(j, i, format(matriz_confusion[i, j], fmt),
                     ha="center", va="center",
                     color="white" if matriz_confusion[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel('Predicciones')
        plt.ylabel('Etiquetas reales')
        plt.show()

    else:
        print("No hay suficientes datos para mostrar la matriz de confusión.")

# Función para calcular y mostrar las métricas de evaluación por categoría
def mostrar_metricas(y_true, y_pred):
    # Calcular métricas por categoría 
    precision = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 4])  # Excluyendo clase 3 ('cafe verde')
    recall = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 4])       # Excluyendo clase 3 ('cafe verde')
    f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 4])              # Excluyendo clase 3 ('cafe verde')

    # Mostrar las métricas en una tabla
    resultados_frame = tk.Toplevel()
    resultados_frame.title("Métricas de Evaluación por Categoría")
    tk.Label(resultados_frame, text="Métricas de Evaluación por Categoría", font=("Helvetica", 14)).pack(pady=10)

    # Crear la tabla de métricas
    table = ttk.Treeview(resultados_frame, columns=('Categoría', 'Precision', 'Recall', 'F1-Score'), show='headings')
    table.heading('Categoría', text='Categoría')
    table.heading('Precision', text='Precisión')
    table.heading('Recall', text='Recall')
    table.heading('F1-Score', text='F1-Score')

    # Insertar los datos por categoría (sin 'cafe verde')
    for idx, categoria in enumerate(categories):
        table.insert('', 'end', values=(categoria, precision[idx], recall[idx], f1[idx]))

    # Mostrar la tabla
    table.pack(pady=20)

    # Mostrar el número de imágenes procesadas por categoría
    tk.Label(resultados_frame, text="Imágenes procesadas por categoría", font=("Helvetica", 14)).pack(pady=10)
    for categoria, count in imagenes_procesadas.items():
        tk.Label(resultados_frame, text=f"{categoria}: {count} imágenes").pack()

# Interfaz gráfica de Tkinter
root = tk.Tk()
root.title("Detección y Clasificación de Granos de Café")
root.configure(bg="#c1d631")  # Color de fondo de la ventana

# Crear un estilo personalizado para los widgets de ttk
estilo = ttk.Style()
estilo.theme_use('clam')

# Cambiar los estilos personalizados de los widgets
estilo.configure('TButton', font=('Helvetica', 12), padding=6, background='#a2bf39', foreground='black')
estilo.map('TButton', background=[('active', '#c1d631')])
estilo.configure('TLabel', font=('Helvetica', 12), background="#a2bf39", foreground='black')
estilo.configure('TMenubutton', font=('Helvetica', 12), background='#a2bf39', foreground='black')

# Crear un marco para los controles (botones, predicción, etc.)
frame_controles = tk.Frame(root, bg="#a2bf39")
frame_controles.grid(row=0, column=0, padx=10, pady=10, sticky="n")

# Crear otro marco para la imagen del grano
frame_imagen = tk.Frame(root, bg="#dce775")
frame_imagen.grid(row=0, column=1, padx=10, pady=10)

# Variables de interfaz
img_label = None
current_image = None
correccion_var = tk.StringVar(value='verde')  # Configurar valor inicial a 'verde'
prediccion_label = tk.StringVar(value="Predicción: ")

# Cargar el logo de la UTS
logo_path = r"C:\Users\Usuario\Desktop\Proyecto de grado\base de datos granos de cafe\uts.jpg"
if os.path.exists(logo_path):
    logo_img = Image.open(logo_path)
    logo_img = logo_img.resize((200, 200), Image.LANCZOS)
    logo_tk = ImageTk.PhotoImage(logo_img)
    tk.Label(frame_controles, image=logo_tk, bg="#a2bf39").grid(row=0, column=0, padx=10, pady=10, sticky="nw")

# Etiqueta para mostrar la predicción de la CNN
ttk.Label(frame_controles, textvariable=prediccion_label).grid(row=1, column=0, padx=10, pady=10)

# Menú desplegable para seleccionar la categoría correcta (sin 'cafe verde')
ttk.Label(frame_controles, text="Categoría predicha:").grid(row=2, column=0, padx=10, pady=10)
ttk.OptionMenu(frame_controles, correccion_var, categories[0], *categories).grid(row=3, column=0)

# Botones de la interfaz
ttk.Button(frame_controles, text="Cargar Imagen", command=seleccionar_imagen).grid(row=4, column=0, pady=10)
ttk.Button(frame_controles, text="Guardar Corrección", command=guardar_correccion).grid(row=5, column=0, pady=10)
ttk.Button(frame_controles, text="Reentrenar Modelo", command=reentrenar_modelo).grid(row=6, column=0, pady=10)
ttk.Button(frame_controles, text="Siguiente Grano", command=siguiente_grano).grid(row=7, column=0, pady=10)
ttk.Button(frame_controles, text="Mostrar Matriz de Confusión", command=mostrar_matriz_confusion).grid(row=8, column=0, pady=10)

# Ejecutar la ventana
root.mainloop()
