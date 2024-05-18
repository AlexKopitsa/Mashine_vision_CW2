import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Завантаження збереженої моделі
model = load_model('my_model.keras')

# Перевірка точності на тестових даних
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Назви класів CIFAR-100
cifar100_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Вибірка класів rocket, bear та seal
selected_classes = ['rocket', 'bear', 'seal']
selected_class_indices = [cifar100_labels.index(cls) for cls in selected_classes]

# Фільтрування даних для вибраних класів
def filter_classes(x, y, class_indices):
    filtered_x = []
    filtered_y = []
    for i in range(len(y)):
        if y[i][0] in class_indices:
            filtered_x.append(x[i])
            filtered_y.append(class_indices.index(y[i][0]))
    return np.array(filtered_x), np.array(filtered_y)

x_test_filtered, y_test_filtered = filter_classes(x_test, y_test, selected_class_indices)

# Перетворення міток на категоріальний формат
y_test_filtered = to_categorical(y_test_filtered, num_classes=len(selected_classes))

# Оцінка моделі
test_loss, test_acc = model.evaluate(x_test_filtered, y_test_filtered, verbose=2)
print(f'\nТочність на тестових даних: {test_acc}')