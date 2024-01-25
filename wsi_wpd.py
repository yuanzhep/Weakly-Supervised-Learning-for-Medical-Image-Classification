import os
import pywt
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import logging

wavelet_filter = 'db3'
# wavelet_filter = 'haar'
# wavelet_filter = ['db2', 'db4', 'sym4', 'coif4']
# first tree-structure, then 0.1 std
wavelet_level = 3
# wavelet_level = 4
wavelet_mode = 'symmetric'

def process_package(package_path):
    package_features = []
    for file in os.listdir(package_path):
        if file.endswith(".jpg"):
            image_path = os.path.join(package_path, file)
            features = process_image(image_path)
            package_features.append((file, features))
    return package_features

def process_folder(folder_path):
    all_features = {}
    for package in os.listdir(folder_path):
        package_path = os.path.join(folder_path, package)
        if os.path.isdir(package_path):
            package_features = process_package(package_path)
            all_features[package] = package_features
    return all_features

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print(f"Image '{image_path}' is represented as a {img.shape[0]}x{img.shape[1]} matrix.")
    # print(f"First 5x5 pixels:\n{img[:5, :5]}\n")
    p, q = img.shape
    wp = pywt.WaveletPacket2D(data=img, wavelet=wavelet_filter, mode=wavelet_mode, maxlevel=wavelet_level)
    energy_features = []
    for node in wp.get_level(wavelet_level, "natural"):
        coeff = node.data
        #energy = np.sum((coeff ** 2) / ((p * q) ** 2))  
        energy = np.sum(np.abs(coeff) / (p * q))
        energy_features.append(energy)
    return energy_features

def save_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            image_name, features = row
            writer.writerow(features)

def main():
    # Configure logging
    logging.basicConfig(filename='0305_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    folder_path = "/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0301_wpd/"
    all_features = process_folder(folder_path)

    for package, package_features in all_features.items():
        output_csv = f"0305_6w_{package}_output.csv"
        save_to_csv(package_features, output_csv)
        logging.info(f"CSV file saved: {output_csv}")
        output_dir = "/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0301_wpd/output/"
        logging.info(f"Visualization saved for package: {package}")

if __name__ == "__main__":
    main()
