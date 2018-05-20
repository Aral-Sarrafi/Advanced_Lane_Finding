# Advanced Lane Finding
With in this project some of the more advanced topics from the computer vision is used to detect the lane line in a video stream. The outline of the project is as follows:

### 1. Camera Calibration
### 2. Image Correction
### 3. Color Threshold
### 4. Gradient Threshold
### 5. Combining the Color and Gradient Threshold
### 6. Bird's-eye View
### 7. Lane Detection in a single image
### 8. Lane Detection in a Video Stream

In the next pragraphs I will be expailning each of the topics in detail.

## 1. Camrera Calibration
The pin hole camera model is an ideal camera model, but is reality the camera's are equiped with lenses that may induce unwanted errors and distorsions to the image. Therefore, the distorsion should be corrected to have a better lane detection algorithm. Moreover, the curveture of the lanes will be also effected by the lens distorsions. In order to correct the lens distorsions the camera should be calibrated.

**Calibration Method:** The calibration starts with detecting corners in images from chessborade taken from different camra pose and location. Because the chessborad is not moving the corresponding corners can be used to aquire the calibration parameters including the correction matrix.
