import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Iterate over each label file in the labels folder
def create_csv_annotations(images_folder, labels_folder, name):
    annotations = []
    image_width = 300
    image_height = 300

    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_folder, label_file), 'r') as f:
                lines = f.readlines()
            
            image_name = os.path.splitext(label_file)[0] + '.jpg'
            image_path = os.path.join(images_folder, image_name)
            
            for line in lines:
                class_label, x_center, y_center, width, height = map(float, line.split())
                x_min = (x_center - width / 2)
                y_min = (y_center - height / 2)
                x_max = (x_center + width / 2)
                y_max = (y_center + height / 2)
                
                annotations.append([image_path, x_min, y_min, x_max, y_max, image_width, image_height, class_label])

        # Here we create a DataFrame from annotations list and then we convert the df into a csv file
        df = pd.DataFrame(annotations, columns=['img_path', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'label'])
        df.to_csv(name, index=False)

# Create csv annotations for train and val datasets
"""""
create_csv_annotations('images/train', 'labels/train', 'annotations_train.csv')
create_csv_annotations('images/val', 'labels/val', 'annotations_val.csv')
"""

# Load annotations from CSV
annotations = pd.read_csv('annotations_train.csv')

train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

input_shape = (300, 300, 3)  # Example input shape for images

# Function to preprocess image and annotations -> this is because 
# the annotations are still not in the format required for TF
def preprocess_data(annotation):
    image = load_img(annotation['img_path'], target_size=(input_shape[0], input_shape[1]))
    image_array = img_to_array(image)
    image_array /= 255.0
    bbox = [annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']]
    label = annotation['label']
    return image_array, bbox, label

train_data = train_annotations.apply(preprocess_data, axis=1)
val_data = val_annotations.apply(preprocess_data, axis=1)

# Convert preprocessed data into arrays -> this is the format needed for TF
X_train, y_train_bbox, y_train_label = zip(*train_data)
X_val, y_val_bbox, y_val_label = zip(*val_data)

# Convert lists to numpy arrays
X_train = tf.convert_to_tensor(X_train)
y_train_bbox = tf.convert_to_tensor(y_train_bbox)
y_train_label = tf.convert_to_tensor(y_train_label)
X_val = tf.convert_to_tensor(X_val)
y_val_bbox = tf.convert_to_tensor(y_val_bbox)
y_val_label = tf.convert_to_tensor(y_val_label)

from tensorflow.keras import Input, Model

# This is the RCNN model, this is just base model for testing
def create_rcnn_model(input_shape, num_classes):
    #layer.Input para definir explícitamente el tensor de entrada
    input_tensor = layers.Input(shape=input_shape)
    # Apila las capas convolucionales y de pooling
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)

    # Aplana la salida para la capa densa
    x = layers.Flatten()(x)

    # Añade capas densas
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax', name='classifier_output')(x)

    # Crea el modelo
    model = Model(inputs=input_tensor, outputs=output)

    return model

# Number of classes
#Aquí se obtiene el número de clases a partir de las anotaciones, +1 para indicar que el último también entra 
num_classes = int(annotations['label'].max()+1)

# Create an instance of the R-CNN model
rcnn_model = create_rcnn_model(input_shape, num_classes)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


s
#Aquí se configuran los callbacks para el entrenamiento del modelo
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1), # Detiene el entrenamiento si no hay mejora
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1) # Reduce la tasa de aprendizaje si no hay mejora
]

# Compile the model with appropriate losses and metrics
rcnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # sparse_categorical_crossentropy para etiquetas enteras

# Train the model
rcnn_model.fit(X_train, y_train_label, validation_data=(X_val, y_val_label), epochs=10,batch_size=32,  
    callbacks=callbacks) 

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.argmax(rcnn_model.predict(X_val), axis=1)

y_val_label = np.array(y_val_label, dtype=int)
y_pred = np.array(y_pred, dtype=int)

#Aquí se obtienen los nombres de las clases a partir de las anotaciones, hay que tener en cuenta que se añade una clase más para indicar que el último también entra
if num_classes == 4:
    target_names = ['Vehiculos', 'Construcciones', 'Vias']
elif num_classes == 5:
    target_names = ['Vehiculos', 'Construcciones', 'Vias', 'Rios']
elif num_classes == 6:
    target_names = ['Vehiculos', 'Construcciones', 'Vias', 'Rios', 'Mineria']

print(classification_report(y_val_label, y_pred, target_names=target_names))


conf_matrix = confusion_matrix(y_val_label, y_pred)
print("Confusion Matrix:")
print(conf_matrix)



import matplotlib.pyplot as plt
import cv2
import seaborn as sns

# Calcula la matriz de confusión
conf_matrix = confusion_matrix(y_val_label, y_pred)

# Heatmap de la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Ejemplo para visualizar una imagen con anotaciones
sample_data = annotations.iloc[0]
image = cv2.imread(sample_data['img_path'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Coordenadas desnormalizadas para visualización
x_min = int(sample_data['xmin'] * sample_data['width'])
x_max = int(sample_data['xmax'] * sample_data['width'])
y_min = int(sample_data['ymin'] * sample_data['height'])
y_max = int(sample_data['ymax'] * sample_data['height'])

cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

plt.imshow(image)
plt.show()