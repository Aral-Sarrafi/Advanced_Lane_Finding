
# coding: utf-8

# In[9]:


import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt


# In[2]:


def warp(img):
    
    h = img.shape[0]
    w = img.shape[1]
    
    src = np.float32([(575,464),
                      (707,464), 
                      (258,682), 
                      (1049,682)])
    dst = np.float32([(450,0),
                      (w-450,0),
                      (450,h),
                      (w-450,h)])
    
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, (w,h), flags = cv2.INTER_LINEAR)
    
    return warped, Minv


# In[10]:


def HLS_threshold (img, thresh = (0,255)):
    HLS_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    S_channel = HLS_img[:,:,2]
    Normalized_S_channel = (255.0*S_channel)/(np.max(S_channel))
    
    S_binary = np.zeros_like(S_channel)
    S_binary[(Normalized_S_channel >= thresh[0]) & (Normalized_S_channel <= thresh[1])] = 1
    return S_binary


# In[4]:


def abs_sobel_thresh(img, orient = 'x', sobel_kernel = 3, thresh = (0, 255)):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if orient == 'x':
        grad = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    else:
        grad = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        
    grad_binary = np.zeros_like(gray_img)
    
    grad_binary[(grad > thresh[0]) & (grad <= thresh[1])] = 1
    
    return grad_binary


# In[5]:


def mag_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = (255*grad_mag/np.max(grad_mag))
    
    mag_binary = np.zeros_like(gray_img)
    mag_binary[(grad_mag > thresh[0]) & (grad_mag <= thresh[1])] = 1
    
    return mag_binary


# In[6]:


def dir_thresh(img, sobel_kernel = 3, thresh = (-np.pi/2, np.pi/2)):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    
    
    grad_dir = np.arctan2(np.abs(grad_y), np.abs(grad_x))
    
    dir_binary = np.zeros_like(gray_img)
    dir_binary[(grad_dir > thresh[0]) & (grad_dir <= thresh[1])] = 1
    
    return dir_binary


# In[7]:


def grad_thresh(img, thresh_x, thresh_y, thresh_m, thresh_d):
    
    gradx = abs_sobel_thresh(img, orient = 'x', sobel_kernel = 3, thresh = thresh_x)
    grady = abs_sobel_thresh(img, orient = 'y', sobel_kernel = 3, thresh = thresh_y)
    mag_binary = mag_thresh(img, sobel_kernel = 3, thresh = thresh_m)
    dir_binary = dir_thresh(img, sobel_kernel = 3, thresh = thresh_d)
    
    combined_binary = np.zeros_like(gradx)
    
    combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined_binary


# In[8]:


def combine_thresholds(binary_1, binary_2):
    final_binary = np.zeros_like(binary_1)
    
    final_binary[(binary_1 == 1) | (binary_2 == 1)] = 1
    
    return final_binary


# In[ ]:


def region_mask(binary_warped):
    grid = np.indices(binary_warped.shape)
    cols = grid[1]
    mask = np.zeros_like(cols)
    mask[(cols<950) & (cols>400)] = 1
    
    binary_warped[mask!=1] = 0
    
    return binary_warped


# In[ ]:


def Generate_Binary_Warped(img, mtx, dist):
    
    undist_image = cv2.undistort(img, mtx, dist, None, mtx)
    
    Color_binary = HLS_threshold(undist_image, (160, 255))
    Grad_binary = grad_thresh(undist_image, (25, 255), (200, 255), (60, 255), (0.35, 1.57))
    
    combined_binary = combine_thresholds(Color_binary, Grad_binary)
    
    binary_warped, Minv = warp(combined_binary)
    
    binary_warped = region_mask(binary_warped)
  
    
    return binary_warped, Minv, undist_image


# In[ ]:


def Image_Lane_detection(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, leftx, lefty, rightx, righty, left_lane_inds,right_lane_inds


# In[ ]:


def annotate_image(undist_image, binary_warped, Minv, left_fit, right_fit):

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    


    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_image.shape[1], undist_image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist_image, 1, newwarp, 0.4, 0)
    
    return result


# In[ ]:


def IMAGE_PROCESS(img, mtx, dist):
    
    binary_warped, Minv, undist_image = Generate_Binary_Warped(img, mtx, dist)
    
    left_fit, right_fit, leftx, lefty, rightx, righty, left_lane_inds,right_lane_inds = Image_Lane_detection(binary_warped)
    
    output = annotate_image(undist_image, binary_warped, Minv, left_fit, right_fit)
    
    return output, left_fit, right_fit, left_lane_inds, right_lane_inds


# In[11]:


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #number of detected pixels
        self.px_count = None
        
        # Method to keep track of the lane
    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.0005 or                self.diffs[1] > 0.5 or                self.diffs[2] > 50.) and                len(self.current_fit) > 0:
                # bad fit!
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 7:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-7:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)


# In[ ]:


def VIDEO_PROCESS(cap, mtx, dist, Video_name):

    
    # Video infomration    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # output video spec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(Video_name, fourcc, fps,(frameWidth,frameHight))
    
    # Left and Right Lane classes
    left_fit = None
    right_fit = None
    Right_Lane = Line()
    Left_Lane = Line()

    for i in range(frameCount):
        
        ret, img = cap.read()
        
        if ret == True:
            if i == 0:

                output, left_fit, right_fit, left_lane_inds, right_lane_inds = IMAGE_PROCESS(img, mtx, dist)
                
                Left_Lane.add_fit(left_fit, left_lane_inds)
                Right_Lane.add_fit(right_fit, right_lane_inds)
                
                out.write(output)


            else:

                binary_warped, Minv, undist_image = Generate_Binary_Warped(img, mtx, dist)

                left_fit, right_fit, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds = Frame_Lane_detection(binary_warped, left_fit , right_fit)
                
                Left_Lane.add_fit(left_fit, left_lane_inds)
                Right_Lane.add_fit(right_fit, right_lane_inds)
                
                left_fit = Left_Lane.best_fit
                right_fit = Right_Lane.best_fit
                

                output = annotate_image(undist_image, binary_warped, Minv, left_fit, right_fit)
                out.write(output)
                
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# In[ ]:


def Frame_Lane_detection(binary_warped, left_fit , right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
    
    return left_fit, right_fit, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds


# In[ ]:


def Lane_Curveture(leftx, lefty, rightx, righty):
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    y_eval = np.max(ploty)


    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

