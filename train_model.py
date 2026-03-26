import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import MobileNetV2

def build_multi_output_model(input_shape=(224, 224, 3), num_types=3, num_models=8):
    inputs = Input(shape=input_shape, name='image_input')
    
    # Базовая предобученная модель (Transfer Learning)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # Замораживаем базовые слои, чтобы не сломать их "умные" веса на начальном этапе
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Преобразуем многомерные признаки в плоский вектор
    x = Dense(256, activation='relu')(x)
    shared_features = Dropout(0.5)(x)
    
    # Выход 1: Тип самолета
    type_dense = Dense(64, activation='relu')(shared_features)
    type_output = Dense(num_types, activation='softmax', name='type_output')(type_dense)
    
    # Выход 2: Конкретная Модель
    model_dense = Dense(128, activation='relu')(shared_features)
    model_output = Dense(num_models, activation='softmax', name='model_output')(model_dense)
    
    model = Model(inputs=inputs, outputs=[type_output, model_output], name="Aircraft_MultiOutput_MobileNet")
    
    model.compile(
        optimizer='adam',
        loss={
            'type_output': 'sparse_categorical_crossentropy',
            'model_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={'type_output': 1.0, 'model_output': 1.0},
        metrics={'type_output': ['accuracy'], 'model_output': ['accuracy']}
    )
    return model

# Функция для загрузки и аугментации изображений
def load_image_and_labels(image_path, type_label, model_label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # MobileNetV2 ожидает специальную нормализацию пикселей [-1, 1], а не просто [0, 1]
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, {'type_output': type_label, 'model_output': model_label}

def main():
    print("Инициализация пайплайна обучения нейросети...\n")
    if not os.path.exists('dataset_metadata.csv'):
        print("Ошибка: dataset_metadata.csv не найден.")
        return

    df = pd.read_csv('dataset_metadata.csv')
    num_types = len(df['type_label'].unique())
    print(f"Всего изображений: {len(df)}")
    
    # Автоматическая кодировка текстовыех названий моделей в числа
    model_encoder = LabelEncoder()
    df['model_encoded'] = model_encoder.fit_transform(df['model_label'])
    num_models = len(model_encoder.classes_)
    
    # Сохраняем маппинг в файл, чтобы веб-интерфейс знал, под каким ID какая модель
    classes_df = pd.DataFrame({'Model': model_encoder.classes_, 'ID': range(num_models)})
    classes_df.to_csv('classes_mapping.csv', index=False)
    
    # Разбиваем данные на обучающую и проверочную выборки (80% / 20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Обучающая выборка: {len(train_df)}, Валидационная выборка: {len(val_df)}")
    
    # Создаем быстрый tf.data.Dataset пайплайн
    def df_to_dataset(dataframe, batch_size=32):
        paths = dataframe['image_path'].values
        type_labels = dataframe['type_label'].values
        model_labels = dataframe['model_encoded'].values
        
        ds = tf.data.Dataset.from_tensor_slices((paths, type_labels, model_labels))
        ds = ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(dataframe)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
        
    train_ds = df_to_dataset(train_df, batch_size=32)
    val_ds = df_to_dataset(val_df, batch_size=32)
    
    # Мы изменили имя файла сохранения, чтобы скрипт начал строить НОВУЮ сеть MobileNet, 
    # а не подцепил старую простую CNN сеть, которая уже есть в папке.
    model_path = 'aircraft_multi_model_mobilenet.h5'
    if os.path.exists(model_path):
        print(f"\n[INFO] Найден файл {model_path}! Загружаем его для ПРОДОЛЖЕНИЯ обучения (дообучение)...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("\n[INFO] Сохраненной модели не найдено. Создаем новую с нуля...")
        model = build_multi_output_model(num_types=num_types, num_models=num_models)
    
    # Коллбэки: Сохранять модель только если она улучшилась на валидации
    checkpoint = ModelCheckpoint('aircraft_multi_model_mobilenet.h5', monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("\nСтарт обучения.")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,  # 30 эпох с ранней остановкой
        callbacks=[checkpoint, early_stop]
    )
    
    print("\nОбучение завершено.")

if __name__ == "__main__":
    main()
