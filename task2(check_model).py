import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Завантаження збереженої моделі
model = load_model('my_model.keras')

# Функція для завантаження та підготовки зображення
def prepare_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Перетворення BGR в RGB
    img = cv2.resize(img, (32, 32))  # Зміна розміру зображення до 32x32 пікселів
    img = img.astype('float32') / 255.0  # Нормалізація зображення
    img = np.expand_dims(img, axis=0)  # Додавання осі для пакетної обробки
    return img

# Завантаження та підготовка власного зображення
file_path = 'images_to_check_model/bear_1.jpg'
own_image = prepare_image(file_path)

# Розпізнавання за допомогою натренованої моделі
predictions = model.predict(own_image)
predicted_class = np.argmax(predictions[0])

# Виведення результату
class_names = ['rocket', 'bear', 'seal']
print(f'Розпізнаний клас: {class_names[predicted_class]}')

# Відображення зображення
plt.imshow(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB))
plt.title(f'Розпізнаний клас: {class_names[predicted_class]}')
plt.show()