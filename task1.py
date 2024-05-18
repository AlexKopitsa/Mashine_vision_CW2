import cv2
import numpy as np
import os


PATH_METADATA = r"content/metadata.txt"

gt_values = {}
i = 0
for line in open(PATH_METADATA, "r"):
    if i < 5:
        name, value = line.split(" - ")
        gt_values[name] = float(value)
        i += 1

# Solution

def calculate_distance_to_object(image_files, known_distance, known_image_index=0):
    # Ініціалізація детектора ORB
    orb = cv2.ORB_create()

    # Ініціалізація словника для зберігання результату
    estimated_distances = {}

    # Ініціалізація списків для зберігання ключових точок, дескрипторів та назв фото
    keypoints_list = []
    descriptors_list = []
    images_names = []

    # Виявлення ключових точок та дескрипторів для кожного зображення, додавання назв фото у список
    for file in image_files:
        file_name = os.path.basename(file)
        images_names.append(file_name)
        img = cv2.imread(file, 0)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # Ініціалізація BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Зіставлення дескрипторів між відомим зображенням та іншими зображеннями
    distances = []
    for i, descriptors in enumerate(descriptors_list):
        if i == known_image_index:
            distances.append(known_distance)
            continue

        matches = bf.match(descriptors_list[known_image_index], descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Обчислення коефіцієнта пропорційності на основі зіставлених ключових точок
        ratio_sum = 0
        for match in matches:
            pt1 = keypoints_list[known_image_index][match.queryIdx].pt
            pt2 = keypoints_list[i][match.trainIdx].pt
            distance1 = np.linalg.norm(np.array(pt1))
            distance2 = np.linalg.norm(np.array(pt2))
            ratio_sum += distance2 / distance1

        # Середній коефіцієнт пропорційності
        scale_ratio = ratio_sum / len(matches)
        distance = known_distance / scale_ratio
        distances.append(distance)

        for i in range(len(distances)):
            estimated_distances[images_names[i]] = distances[i]

    return estimated_distances

# Приклад використання
image_files = [
    'content/book_1.jpg',
    'content/book_2.jpg',
    'content/book_3.jpg',
    'content/book_4.jpg',
    'content/book_5.jpg'
]

known_distance = 100.0  # відстань для першого зображення

estimated_distances = calculate_distance_to_object(image_files, known_distance)
print("Відстані до об'єкта для кожного зображення:")
for name, value in estimated_distances.items():
    print(name, ": ", value)


# Metric calculation

general_dist_error = 0.0

i = 0
for name, estimated_dist in estimated_distances.items():
    if i > 0:
        general_dist_error += abs(estimated_dist - gt_values[name])
    i += 1

avarage_dist_error = general_dist_error / 4.0

print("Avarage distance error for images 2-5: ", avarage_dist_error)



