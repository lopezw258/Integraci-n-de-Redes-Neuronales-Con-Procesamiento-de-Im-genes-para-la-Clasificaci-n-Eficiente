import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Directorio de datos que contiene subcarpetas para cada categoría
data_dir = r'C:\Users\Usuario\Desktop\Proyecto de grado\base de datos granos de cafe'
categories = ['brocado', 'verde', 'vinagre', 'cafe verde', 'marron vinagre']
img_size = 128  # Tamaño de las imágenes para la CNN

# Función para cargar y procesar las imágenes
def cargar_imagenes_y_etiquetas():
    data = []
    labels = []
    total_imagenes = 0

    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        # Comprobar si la carpeta existe
        if not os.path.exists(path):
            print(f"La carpeta {category} no existe en el directorio {data_dir}.")
            continue

        print(f"Cargando imágenes para la categoría: {category}")
        count = 0  # Contador de imágenes cargadas por categoría

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Error al cargar la imagen: {img_name} en la categoría {category}")
                continue

            try:
                # Redimensionar la imagen para que todas tengan el mismo tamaño
                img = cv2.resize(img, (img_size, img_size))
                img = img.astype('float32') / 255.0  # Normalizar los valores de píxel entre 0 y 1

                data.append(img)
                labels.append(label)
                count += 1
                total_imagenes += 1
            except Exception as e:
                print(f"Error al procesar la imagen {img_name} en {category}: {e}")

        print(f"Total imágenes cargadas para {category}: {count}")

    print(f"Total de imágenes cargadas: {total_imagenes}")
    return np.array(data), np.array(labels)

# Cargar imágenes y etiquetas
imagenes, etiquetas = cargar_imagenes_y_etiquetas()

# Comprobar si se han cargado imágenes
if len(imagenes) == 0 or len(etiquetas) == 0:
    print("No se han podido cargar imágenes. Verifique la estructura de las carpetas y las imágenes.")
else:
    # Convertir las etiquetas a categóricas (One-hot encoding)
    etiquetas = to_categorical(etiquetas, num_classes=len(categories))

    # Separar en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    x_train, x_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

    # Definir el modelo CNN
    model = Sequential([
        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Aplanar para pasar a la capa densa
        Flatten(),

        # Capa densa completamente conectada
        Dense(128, activation='relu'),

        # Capa de salida con activación softmax para clasificación multiclase
        Dense(len(categories), activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Precisión en el conjunto de prueba: {test_acc}")

    # Guardar el modelo entrenado
    model.save("modelo_cafe_caracteristicas_cnn.keras")

    # Gráficas de precisión y pérdida durante el entrenamiento
    history_dict = history.history

    # Crear las gráficas
    plt.figure(figsize=(12, 4))

    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history_dict['val_accuracy'], label='Precisión de validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Pérdida de entrenamiento')
    plt.plot(history_dict['val_loss'], label='Pérdida de validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()
    plt.show()
