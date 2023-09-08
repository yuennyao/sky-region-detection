# Sky Pixel (Region) Detection using Computer Vision Algorithm
The aim of this project is to develop a computer vision algorithm that can be used to identify pixels that belong to the sky region automatically. Identifying the sky region in outdoor images is the very first step in many other applications, such as weather forecast, solar exposure detection and ground robot navigation.

The main tasks are to: 
(i) develop a computer vision-based system to identify pixels belong to the sky region using Python and OpenCV, and 
(ii) evaluate its performance by applying the proposed system on images taken from the SkyFinder dataset 623, 684, 9730.


<h3>Steps to run the algorithm:</h3>
1. Folder "623", "684", and "9730" contain the images for evaluation, while the "mask" folder includes the ground truth mask obtained from SkyFinder datasets for evaluation purposes <br>
2. There are 2 .py files for the algorithm: sky_detection_function.py and sky_detection_operation.py <br>
3. It is preferred to run the code in Spyder <br>
4. Before running the code in Spyder, go to Tools, Preferences, Ipython Console, Graphics and under Graphics Backend select "inline" instead of "automatic". This is to prevent the 200 pltplot figures from displaying in a new window separately <br>
5. Run sky_detection_operation.py and wait until the process ends <br>
6. Upon completion of the process, the summary of results will be displayed in the console. There will also be an "output" folder added inside the root folder which contains the detected sky regions and comparison figures for each dataset <br>

<h3>Results:</h3>
![20121024_214145_comp](https://github.com/yuennyao/sky-region-detection/assets/87840513/cbc9945a-52c3-4591-ace0-674687113527)
![20130622_065704_comp](https://github.com/yuennyao/sky-region-detection/assets/87840513/f5885d42-de6a-422d-be6c-6174fc467c5b)


<hr>
<h6>Project completed in 2023</h6>
