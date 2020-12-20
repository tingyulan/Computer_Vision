# Computer Vision
**Computer Vision at National Chiao Tung University**  
Lecturer: [Prof. Wei-Chen Chiu](https://walonchiu.github.io)
<br><br>

- HW1: Camera Calibration 
    - Capture chessboard images by our own smart phone (Remember to close the auto focus function)
    - Find our intrinsic matrix and extrinsic matrixs. Then, plot the position of the camera while taking each images. 
    - [note] The visualization code was provided by TAs

- HW2:
    - Hybrid image
        - Sum a low-passed filtered version of a image with another high-passed filtered version of second image.
        - Try both ideal and Gaussian filters
    - Image pyramid
        - A collection of images at different resolution
        - Show both Gaussian and Laplacian pyramids
    - Colorizing the Russian Empire
        - Automatically produce a color image from the digitized Prokudin- Gorskii glass plate images with as few visual artifacts as possible
    - [note] All functions are implemented from scartch, except for Fourier Transform related functions and image I/O functions

- HW3: Automatic Panoramic Image Stitching
    - Find interest point by SIFT
    - Feature matching by SIFT features
    - RANSAC to find homography matrix
    - Warp image to create panoramic image
    - [note] All functions are implemented from scartch, except for descriptors = cv2.xfeatures2d.SIFT_create(), descriptors.detectAndCompute, cv2.resize, and image I/O

- HW4: Structure from Motion
    - Take our own photos, do calibration on our own photos, and reconsturct 3D model
    - [note] All functions are implemented from scartch, except for calibration related functions, visualization functions, and functions allowed to use in HW3

- [Note] All assignments are done in a group of three.
