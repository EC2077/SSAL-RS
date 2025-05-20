import os
import cv2
import numpy as np

# 定义类别
CLASSES = ['airport_runway', 'artificial_grassland', 'avenue', 'bare_land', 'bridge', 'city_avenue', 'city_building', 'city_green_tree', 'city_road', 'coastline', 'container', 'crossroads', 'dam', 'desert', 'dry_farm', 'forest', 'fork_road', 'grave', 'green_farmland', 'highway', 'hirst', 'lakeshore', 'mangrove', 'marina', 'mountain', 'mountain_road', 'natural_grassland', 'overpass', 'parkinglot', 'pipeline', 'rail', 'residents', 'river', 'river_protection_forest', 'sandbeach', 'sapling', 'sea', 'shrubwood', 'snow_mountain', 'sparse_forest', 'storage_room', 'stream', 'tower', 'town', 'turning_circle']

def load_RSI_CB128_data(data_dir):
    images = []
    labels = []

    for idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128,128))  # 尺寸为 256x256 像素 RESNET:128
            images.append(img)
            labels.append(idx)  # 使用类别的索引作为标签

    # 转换为 NumPy 数组
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

