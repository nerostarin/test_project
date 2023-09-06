import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image

# 데이터 경로 설정
train_data_dir = 'C:/Users/NERO/Desktop/test_project/path_to_train_data_directory'
validation_data_dir = 'C:/Users/NERO/Desktop/test_project/path_to_validation_data_directory'
num_classes = len(os.listdir(train_data_dir))
img_width, img_height = 150, 150
batch_size = 32

# 데이터 생성기 설정
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 이진 분류 문제에 맞게 조정
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 콜백
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# 모델 훈련
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# 테스트 이미지가 있는 폴더 경로 설정
test_image_folder = 'C:/Users/NERO/Desktop/test_project/getimage'

# 폴더 내의 이미지 파일 목록을 가져옵니다.
image_files = [f for f in os.listdir(test_image_folder) if os.path.isfile(os.path.join(test_image_folder, f))]

# 피드백 처리를 위한 반복문
for random_image_file in image_files:
    random_image_path = os.path.join(test_image_folder, random_image_file)

    # 테스트 이미지 불러오기
    img = image.load_img(random_image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3) 형태로 변환

 # 이미지 예측
predictions = model.predict(img_array)

# 예측 결과 클래스
predicted_class = 1 if predictions[0] > 0.5 else 0

# 클래스에 대응하는 폴더 이름 리스트
class_folder_names = ['box', 'plastic']  # 클래스에 해당하는 폴더 이름 리스트

# 예측 결과 출력
predicted_folder_name = class_folder_names[predicted_class]
print(f'image file: {random_image_file}')
print(f'this is a: {predicted_folder_name}')

# 사용자에게 정답 입력 받기
print("Is this prediction correct? (yes/no):")
correct_answer = 'no'  # input().strip().lower()

# 정답이 틀렸을 경우 모델 재학습
if correct_answer == 'no':
    print("Please enter the correct class (box/plastic):")
    correct_class = 'plastic'  # input().strip().lower()

    if correct_class in class_folder_names:
        # 잘못 예측된 이미지와 정답 라벨을 수집하고 이를 새로운 학습 데이터로 추가
        misclassified_images = img_array / 255.0  # 이미지를 0~1 사이로 정규화
        correct_labels = np.array([1 if correct_class == 'plastic' else 0])

        # 새로운 데이터로 모델을 재학습
        model.fit(
            misclassified_images,
            correct_labels,
            epochs=5,  # 원하는 횟수로 설정
            batch_size=batch_size
        )
    else:
        print("Invalid class. Please enter 'box' or 'plastic'.")