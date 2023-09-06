# -*- coding: utf-8 -*-
"""
Sky Pixel (Region) Detection using Computer Vision Algorithm
by Lee Yuen Yao 
"""
########## Import Libraries ##########
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
# Import the functions from sky_detection_function.py
import sky_detection_function as f 
##################################### 

########## Sky Region Detection Algorithm ##########
def detect_sky_region(dataset_number, file_name):
    image_path = f"{dataset_number}/{file_name}"
    image = cv2.imread(image_path,1)
       
    # Scene Classification
    scene_result = f.classify_scene(image, dataset_number)
    
    # Pre-Process the image if it is nighttime
    if scene_result == "nighttime":
        processed_image = f.preprocess_image(image,dataset_number)
        edges = cv2.Canny(processed_image, 65, 160)
    # Daytime
    else: 
        processed_image = image[:, :, 0]
        edges = cv2.Canny(processed_image, 16, 160)
    
    # Obtain sky region        
    result_mask = f.floodfill(edges, image, dataset_number)
    
    # Obtain skyline
    skyline = f.find_skyline(result_mask, image)
    
    # Create directory to save the output
    current_directory = os.getcwd()
    new_directory = os.path.join(current_directory, f"outputs/{dataset_number}")
    detected_sky_region_dir = os.path.join(new_directory, "detected sky region")
    comparison_figure_dir = os.path.join(new_directory, "comparison figures")
    if not os.path.exists(detected_sky_region_dir):
        os.makedirs(detected_sky_region_dir)
    if not os.path.exists(comparison_figure_dir):
        os.makedirs(comparison_figure_dir)

    cv2.imwrite(f"{detected_sky_region_dir}/{file_name}_skyRegion.jpg", result_mask)
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(result_mask, cmap="gray")
    axes[1].set_title("Sky Region")
    axes[1].axis("off")
    axes[2].imshow(skyline, cmap="gray")
    axes[2].set_title("Skyline")
    axes[2].axis("off")
    fig.text(0.5, 0.1, f"The image is taken during {scene_result}", ha="center", fontsize=12, color="blue")
    plt.tight_layout() 
    fig.suptitle(f"Sky Region and Skyline Detection Result for {file_name}", fontsize=16)
    plt.savefig(f"{comparison_figure_dir}/{image_name}_comp.jpg") 
    plt.close(fig)
    return result_mask
##################################################

if __name__ == "__main__":
    current_directory = os.getcwd()
    dataset_number = ["623","684","9730"]
    for number in dataset_number:
        dataset_directory = os.path.join(current_directory, f'{number}')   
        ground_truth_mask_filename = os.path.join(current_directory, "masks", f"{number}_mask.png")
        ground_truth_mask = cv2.imread(ground_truth_mask_filename, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        
        total_images = 0
        total_accuracy = 0
        total_rmse = 0
        
        print(f"Processing for image set {number}...")
        for file_name in os.listdir(dataset_directory):  
            result_mask = detect_sky_region(number, file_name)
            accuracy_rate, rmse = f.calculate_accuracy_rmse(result_mask, ground_truth_mask)
            total_accuracy += accuracy_rate
            total_rmse += rmse
            total_images += 1
            # Uncomment the code below to see the accuracy and RMSE of each image
            # print("Current for image: ", file_name)
            # print(f"Accuracy Rate: {accuracy_rate:.2f}%")
            # print(f"RMSE: {rmse:.2f}\n")
       
        average_accuracy = total_accuracy / total_images
        average_rmse = total_rmse  / total_images
             
        print()
        print(f"==========Summary for image set {number}===========")
        print(f"Total Images: {total_images}")
        print(f"Average Accuracy: {average_accuracy:.2f}%")
        print(f"Average RMSE: {average_rmse:.2f}\n")
        
    print("Process Ended.")
