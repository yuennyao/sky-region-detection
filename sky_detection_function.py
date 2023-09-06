# -*- coding: utf-8 -*-
"""
Sky Pixel (Region) Detection using Computer Vision Algorithm
by Lee Yuen Yao 
"""
#This files contains all functions that will be used in the algorithm.
########## Import Libraries ##########
import numpy as np
import cv2
#####################################

########## Function: Scene Classification ############  
# To determine whether the input image was taken during daytime or nighttime based on the average intensity and color temperature.
def classify_scene(image, dataset_number):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    average_intensity = cv2.mean(gray_image)[0]
      
    r, g, b = cv2.split(image)
    r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
    color_temp = 255 * np.array([r_mean, g_mean, b_mean]) / np.sum([r_mean, g_mean, b_mean])

    avg_intensity_threshold = 115
    color_temp_threshold = 200
    
    if average_intensity > avg_intensity_threshold or color_temp[0] > color_temp_threshold:
        return "daytime"
    else:
        return "nighttime"

######## Function: Noise Filtering & Adaptive Histogram Equalisation ###########
def preprocess_image(image, dataset_number):   
    if dataset_number == "9730":
        filtered_image = cv2.GaussianBlur(image, (7, 7), 0)
    else:
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)

    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_image = clahe.apply(gray_image)
    return preprocessed_image

########### Function: Region Floodfillng ##############
# Reference: https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/#:~:text=Steps%20for%20implementing%20imfill%20in%20OpenCV&text=Threshold%20the%20input%20image%20to,white%20and%20white%20becomes%20black%20).
# To fill enclosed areas in the binary image obtained from edge detection.
def floodfill(edges, image, dataset_number):
     closed_edges_se = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45)) 
     closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closed_edges_se, 2)
     
     floodfill_image = closed_edges.copy().astype("uint8")
     h, w = image.shape[:2]
     floodfill_mask = np.zeros((h+2, w+2), np.uint8)
     seed_point = (50, 50) 
     cv2.floodFill(floodfill_image, floodfill_mask, seed_point, 255)
 
     inverted_image = cv2.bitwise_not(floodfill_image)
     or_mask = cv2.bitwise_or(closed_edges, inverted_image, mask=None)
     result_mask = cv2.bitwise_not(or_mask)
     
     contours, _ = cv2.findContours(255 - result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     for contour in contours:
         area = cv2.contourArea(contour)
         if area < 500:
             has_small_black_region = True
         else:
             has_small_black_region = False
             
     closing_se_size = (45, 45) if has_small_black_region else (11, 11)
     closing_se = cv2.getStructuringElement(cv2.MORPH_RECT, closing_se_size)  
     result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, closing_se, iterations=1)
     return result_mask

############## Function: Skyline Detection #################
# To find the contour of the skyline from the merged binary mask obtained after flood-filling. 
def find_skyline(merged_mask, image):
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
    return contour_image

############ Function: Accuracy Calculations ###############
# To calculate the accuracy and Root Mean Square Error (RMSE) of the result by comparing it with the provided ground truth mask
def calculate_accuracy_rmse(detected_mask, ground_truth_mask):
    ground_truth_mask_binary = (ground_truth_mask > 0).astype(np.uint8)
    detected_mask_binary = (detected_mask > 0).astype(np.uint8)
    
    correct_px = np.sum(ground_truth_mask_binary == detected_mask_binary)
    total_px = ground_truth_mask_binary.size 
    accuracy_rate = (correct_px / total_px) * 100.0
    
    rmse = np.sqrt(np.mean((ground_truth_mask_binary - detected_mask_binary) ** 2))
    return accuracy_rate, rmse


