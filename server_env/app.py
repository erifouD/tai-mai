import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'aircraft_multi_model.keras'
MAPPING_PATH = 'classes_mapping.csv'

# Глобальные переменные для хранения модели и словаря классов
model = None
MODEL_CLASSES = {0: "Pending Mapping..."}
TYPE_CLASSES = {0: "Гражданский пассажирский", 1: "Военный истребитель", 2: "Военно-транспортный"}

def load_resources():
    global model, MODEL_CLASSES
    # Загрузка модели
    if model is None and os.path.exists(MODEL_PATH):
        try:
            # Патч: переопределяем Dense и InputLayer чтобы они игнорировали
            # неизвестные аргументы из новых версий Keras (quantization_config, batch_shape)
            OriginalDense = tf.keras.layers.Dense
            OriginalInputLayer = tf.keras.layers.InputLayer

            class PatchedDense(OriginalDense):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('quantization_config', None)
                    super().__init__(*args, **kwargs)

            class PatchedInputLayer(OriginalInputLayer):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('optional', None)
                    if 'batch_shape' in kwargs:
                        kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
                    super().__init__(*args, **kwargs)

            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'Dense': PatchedDense, 'InputLayer': PatchedInputLayer}
            )
            print("Модель успешно загружена.")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")

            
    # Загрузка словаря моделей
    if list(MODEL_CLASSES.values())[0] == "Pending Mapping..." and os.path.exists(MAPPING_PATH):
        try:
            mapping_df = pd.read_csv(MAPPING_PATH)
            MODEL_CLASSES = dict(zip(mapping_df['ID'], mapping_df['Model']))
            print("Маппинг классов загружен.")
        except Exception as e:
            print(f"Ошибка при загрузке маппинга: {e}")

# Попытка загрузить при старте
load_resources()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Попробуем загрузить ресурсы
    load_resources()

    if model is None:
        return jsonify({'error': 'Модель еще не загружена на сервер. Пожалуйста, поместите aircraft_multi_model.h5 в корневую директорию.'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    try:
        # Открываем изображение и подготавливаем для сети
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = np.array(img).astype('float32')
        # MobileNetV2 ожидает специальную нормализацию пикселей [-1, 1]
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0) # Формат (1, 224, 224, 3)
        
        # Инференс
        predictions = model.predict(img_array)
        
        # predictions - это список из двух выходов [type_output, model_output]
        type_preds = predictions[0][0]
        model_preds = predictions[1][0]
        
        top_type_idx = int(np.argmax(type_preds))
        top_type_conf = float(type_preds[top_type_idx])
        
        top_model_idx = int(np.argmax(model_preds))
        top_model_conf = float(model_preds[top_model_idx])
        
        return jsonify({
            'type': {
                'class': TYPE_CLASSES.get(top_type_idx, f"Type {top_type_idx}"),
                'confidence': f'{top_type_conf*100:.1f}%'
            },
            'model': {
                'class': MODEL_CLASSES.get(top_model_idx, f"Model {top_model_idx}"),
                'confidence': f'{top_model_conf*100:.1f}%'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Локальный запуск. В production (в Docker) Flask будет запущен через Gunicorn
    app.run(host='0.0.0.0', port=5000)
