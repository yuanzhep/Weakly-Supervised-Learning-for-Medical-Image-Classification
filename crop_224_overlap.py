# 0619_2023, yz
# crop 1 * 2048*2048 -> 10*10 224*224 (with stride)

import os
import logging
from PIL import Image

logging.basicConfig(filename='img_crop_processing.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def split_image(img, patch_size, stride):
    width, height = img.size
    patches = []

    for i in range(0, width - patch_size + 1, stride):
        for j in range(0, height - patch_size + 1, stride):
            patch = img.crop((i, j, i + patch_size, j + patch_size))
            patches.append(patch)

    return patches

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output folder {output_folder}")

    total_patches = 0
    overlapping_values = []

    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                logging.info(f"Processing file {file_path}")

                with Image.open(file_path) as img:
                    patch_size = 224
                    num_tiles = 10
                    stride = (img.width - patch_size) // (num_tiles - 1)
                    
                    patches = split_image(img, patch_size, stride)
                    overlapping_values.append(stride)
                    total_patches += len(patches)

                    # Print the stride value for each patch
                    for i, patch in enumerate(patches, 1):
                        print(f"Patch {i} - Stride: {stride}")

                    # Create new output subfolder for each file
                    file_output_folder = os.path.join(output_folder, f"{subfolder}_224")
                    if not os.path.exists(file_output_folder):
                        os.makedirs(file_output_folder)
                        logging.info(f"Created output subfolder {file_output_folder}")

                    for i, patch in enumerate(patches, 1):
                        patch_file_name = f"{file_output_folder}/{file.split('.')[0]}_{patch.size[0]}_{i}.jpg"
                        patch.save(patch_file_name)
                        logging.info(f"Saved patch {patch_file_name}")

    with open("overlap_stats.txt", "w") as file:
        file.write("Overlapping Values:\n")
        for i, val in enumerate(overlapping_values, 1):
            file.write(f"Patch {i}: {val}\n")
        file.write(f"\nTotal Number of Patches: {total_patches}")


if __name__ == "__main__":
    input_folder = "/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0308_code/yz_img/2048/"
    output_folder = "/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0308_code/yz_img/224_img_stride/"
    main(input_folder, output_folder)
