##Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[CH_Explantation]: ./writeup_images/checker_undistort.png "Distorted Chessboard vs Undistorted"
[PT_Explantation]: ./writeup_images/perspective_transform.png "Alt"
[CO_Explantation]: ./writeup_images/coefficients.png "Alt"
[AX_Explantation]: ./writeup_images/array_of_xs.png "Alt"
[PL_Explantation]: ./writeup_images/points_on_lane.png "Alt"
[SP_Explantation]: ./writeup_images/smooth_points.png "Alt"
[RC_Explantation]: ./writeup_images/radius_of_curve.png "Alt"

[imageRoadOrig]: ./output_images/Step1a_preDistorted.jpg "Road Distorted"
[imageRoadUndist]: ./output_images/Step1b_postDistorted.jpg "Road Undistorted"
[imageBinaryThresh]: ./output_images/Step2_binaryThreshold.jpg "Binary Image Example"
[imageWarp]: ./output_images/Step3_perspectiveTransform.jpg "Warp Example"
[imageBinLanes1]: ./output_images/Step4c_extractLanes.jpg "Warp Example"
[imageBinLanes2]: ./output_images/Step4a_binaryLanes.jpg "Warp Example"
[imageFitLanes]: ./output_images/Step5a_fitLanes.jpg "Warp Example"
[imageFitLanesWarped]: ./output_images/Step5b_fittedLanes.jpg "Warp Example"
[imageFinalColorLanes]: ./output_images/Step5c_mergedLaneImages.jpg "Warp Example"
[imageDataWrite]: ./output_images/Step6_final.jpg "Warp Example"

[video1]: ./output_images/project_video_out.mp4 "Result Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is it, you're reading this point:).

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the "source/calibration.py" file. We are provided a set of chessboard images to help with calibration. I start by preparing an array that holds "object points". These will be the (x, y, z) coordinates of the chessboard corners in the real world. For every chessboard image, I use the `cv2.findChessboardCorners` to detect the corners.
I maintain two arrays, `objpoints` and `imgpoints`. `objpoints` will contain the same `objpoint` object for every time all the edges are detected in an image. `imgpoints` will contain the actual locations of the edges. I then used the `objpoints` and `imgpoints` arrays to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. These coefficients are saved in a pickle file. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][CH_Explantation]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
For each image that gets fed into the pipeline, I use the pickled coefficient data to undistort the image. This is seen in the `source/process_image.py` file, in the labeled section (approximately lines 22-26).
This is an example of this step:
![alt text][imageRoadOrig]
![alt text][imageRoadUndist]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
To generate my binary image, I used gradient thresholds and color thresholds (`source/process_image.py` file, in the labeled section, approximately lines 37-43).
I combined the outputs of the image's x-axis gradient, y-axis gradient, and then lastly, the results of thresholding on the images HSV and HLS color scale. I had a low threshold on the x-axis gradient when compared to the y-axis gradient, since it is better able to capture vertical lines that encompass lanes. For the HLS and HSV thresholding, I used the saturation channel and value channel respectively to threshold. These channels appear to be most resistent to noise like shadows and lane discoloration. An example of this can be seen here:
![alt text][imageBinaryThresh]


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
Each image input into the pipeline goes through a perspective transform. The purpose of this transformation is to take the image from its current view, and convert it to a top-down birds-eye view of the road. 
![alt text][PT_Explantation]

To complete a perspective transform on each input, I specified a set of points that would be source points, and another set for destination points. The source points would connect to define a trapezoid, and when transformed, the destination points define a rectangle. The code can be seen in: (`source/process_image.py` file, in the labeled section, approximately lines 59-70). My points are not hardcoded. Instead I hardcode some ratios.
I define the bottom-width of the trapezoid to be about 80% of the original image, the height of the trapezoid is about 60% of the input image, and the top-width of the trapezoid is about 10% of the input image. I verified that my perspective transform was correct by testing that the lanes were parrallel after the transform. The results are seen below:
![alt text][imageWarp]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Now the lanes need to be detected. To do this, I used horizontal slices of the image and a histogram of the values along these slices. This along with the sliding-window technique was enough to detect pixels that defined the lanes with good results. The code can be seen in: (`source/process_image.py` file, in the labeled section, approximately lines 91-147). A breakdown of these steps is seen next. 
I am using the `find_window_centroids()` function in the `centroid_tracker` file to detect pairs of pixels that lie on the left and right lanes. To do this, the image is broken down into horizontal slices. The first points on the bottom most slice are found using a sliding window defined to a width and a height. For every subsequent layer, the left and right borders of each lane are defined, and with that, the center is calculated (for each lane). These pairs are tracked in a list and returned after being smoothed.
![alt text][imageBinLanes1]
![alt text][imageBinLanes2]


Now that we have a set of points that define the left and right lanes, we augment the input image by coloring the lanes in. Further, we calculate curvature of the lane. The code can be seen in: (`source/process_image.py` file, in the labeled section, approximately lines 151-171). I used the 'numpy.polyfit()' function to help fit the points to a second degree polynomial. This function returns the coefficients for this polynomial. 
An example of polyfit from one of the lanes (an array of 3 floats, each representing coefficients of a second degree polynomial): 
![alt text][CO_Explantation]


I generated an array that spanned the height of the image, and combined with the polynomials, generated points that now mapped all along the height of the image and also fit the lane curvature. I did this for both lanes.

Step1: generate an array of y-values spanning the top of the image to the bottom:
![alt text][AX_Explantation]

Step2: generate an array of xy-values spanning the lane:
![alt text][PL_Explantation]

Step3: visualizing these generate points:
![alt text][SP_Explantation]


Using this data, it is now possible to augment the original input with a nicely colored markup of the lanes.
![alt text][imageFitLanes]
![alt text][imageFitLanesWarped]
![alt text][imageFinalColorLanes]






####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
To calculate the vehicles offset from the center of the lane: I relied on the fact that we have already calculated the position of the lanes. With this, I can calculate a point that lies on the center of both lanes. After this, we make another assumption. We assume that the camera is located on the center dasboard of the car, and thus the input images we receive are the exact view of where the car is actually positioned. Given this, we can calculate the offset by taking the difference between the middle of the lane, and the middle of the input image. If the value is positive, the car is to the right of the center. Otherwise the car is to the left.

To calculate the radius of the curve, I used used the left lane only to fit another polynomial. The code can be seen in: (`source/process_image.py` file, in the labeled section, approximately lines 218-236). With this polynomial, we proceed as follows:
This is the math used to calculate the radius of the curvature:
![alt text][RC_Explantation]







####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
The result of augmenting the calculated data onto the original image is seen here:
![alt text][imageDataWrite]
Wow, so perrty.

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
Here's a [link to my result][video1]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline is still likely to fail with complex streams or environments. This can be seen in the challenge videos. This is because the pipeline needs to be tweaked in order to provide the best results. Tweaking that occurs: the distance for how far out to detect lanes is hard-coded. This means that in scenarios where it would not be optimal to look far out into the distance (tight curves that span out of view), the pipeline will still do so. One approach to try and eliminate this issue would be to make this distance variable dynamic. In fact, being able to dynamically detect how wide the lane is, and far out ahead the lane is visisble would greatly improve this pipeline. Another issue is still seen with color-noise. In the challenge video, shadows still provide false-negative detections for the lanes. More tweaking with the color thresholding parameters is needed. Lastly, I think another option could be to eliminate lane tracking altogether if the system cannot stabalize it's lane detections. I would rather have the pipeline not augment the stream if it is extremely uncertain.

 

