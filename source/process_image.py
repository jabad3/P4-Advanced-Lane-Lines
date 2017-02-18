import numpy as np
import cv2
from utils import import_calibration_data
from utils import abs_sobel_thresh
from utils import color_threshold
from centroid_tracker import centroid_tracker
np.set_printoptions(threshold=np.inf)


# pickle file with data to correct for distortion
mtx, dist = import_calibration_data()


def process_image(img, write_files=False):
    ########
    ########
    # Undistort image using calibration pickle.
    # 
    # prior to the pipeline, the camera has gone through calibration. the results
    # were saved in a pickle file. the pickle file can now be used to undistort each image
    # as it comes in through the pipeline.
    distort_correction = cv2.undistort(img, mtx, dist, None, mtx)
    if(write_files):
        cv2.imwrite("../output_images/Step1a_preDistorted.jpg", img);
        cv2.imwrite("../output_images/Step1b_postDistorted.jpg", distort_correction)
    img = distort_correction


    ########
    ########
    # Color-space processing
    #
    # the first step of the pipeline is to threshold the image to help identify
    # lanes. first calculate gradients, then convert image to HLS and HSV color space
    # to reduce the effects of noise like shadows, weather, unevern lane colors, etc.
    # the output of this step is a binary image.
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255)) # x-gradient captures lanes better
    grady = abs_sobel_thresh(img, orient='y', thresh=(25,255))
    color_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255)) # pull HSV/HSL color space data
    preprocessImage[((gradx==1)&(grady==1)|(color_binary==1))] = 255 # union all the filters
    if(write_files):
        cv2.imwrite("../output_images/Step2_binaryThreshold.jpg", preprocessImage)


    ########
    ########
    # Perspective Transform
    #
    # a critical step in determining the curvature of the lane is to convert to a
    # Birds Eye View (looking straight down on the road). With a view like this, we will
    # have parallel lines that we can trace from the bottom of the image upwards (or vice versa).
    # calculating curvature is more straightforward from this view, no need to worry about distance
    # and perspective

    # all of these are percents, the end goal being that they help define a trapezoid.
    # the trapezoid will be widest at the bottom. the bottom and top lines are parallel.
    # the sides taper towards each other.
    img_size = (img.shape[1], img.shape[0])
    bot_width = 0.76 
    mid_width = 0.08
    height_pct = 0.62
    bottom_trim = 0.935

    # define 4 vertices on a trapezoid
    src = np.float32([[img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct],
                      [img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct],

                      [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim],
                      [img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]])
    # distance between the two lanes after perspective transform is applied
    # increasing the offset shrinks the space between lanes
    offset = img_size[0]*.25
    # define a rectangle to project the source points onto
    dst = np.float32([  [offset,0],[img_size[0]-offset,0],
                        [img_size[0]-offset,img_size[1]],[offset,img_size[1]]])

    # calculte matrix for how to map from src points to dst points 
    M = cv2.getPerspectiveTransform(src, dst)
    
    # calculate the opposite of this transformation so we can map back to the untransformed image once we finish
    # computing and drawing on this image
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # finally, warp the image to achieve perspective tranform
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image
    if(write_files):
        cv2.imwrite("../output_images/Step3_perspectiveTransform.jpg", warped)


    ########
    ########
    # Detecting the lanes
    #
    # now that we have a top-down view of the lanes, we need to concretely identify
    # some points that can define the lane. in this step, these points are helpful for
    # marking the lanes in the image. in the next step, these points will be used to
    # calculate curvature
    # approach: the tracker class is used to generate this data. see class for documentation.

    # define a search window. the search window is used to split the image into slices while searching.
    window_height = 30
    window_width = 30
    
    # instantiate a class to help generate lane points
    curve_centers = centroid_tracker(window_width=window_width, window_height=window_height, padding=25, smoothing=15)
    
    # generate points along the left and right lanes using the sliding window technique.
    window_centroids = curve_centers.find_window_centroids(warped)

    # store the points on this map
    leftx = []  # store raw points
    rightx = [] # store raw points
    l_points = np.zeros_like(warped) # store aggregated points mapped onto a canvas
    r_points = np.zeros_like(warped) # store aggregated points mapped onto a canvas

    # the image was broken into horizontal slices in the 'find_window_centroids' method above.
    # here, for each slice, we grab the points for corresponding left and right lanes and 
    # create a mask that will be used to mark up the lanes.
    # the mask is literally a set of blocks along the lane signals
    def window_mask(width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[ int(img_ref.shape[0]-(level+1)*height) : int(img_ref.shape[0]-level*height),
                max(0, int(center-width)) : min(int(center+width),img_ref.shape[1])
              ] = 1
        return output

    for level in range(0,len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        l_points[(l_points==255)|((l_mask==1))] = 255 # aggregate detected points onto a canvas
        r_points[(r_points==255)|((r_mask==1))] = 255 # aggregate detected points onto a canvas

    template = np.array(r_points+l_points,np.uint8)
    if(write_files):
        cv2.imwrite("../output_images/Step4a_binaryLanes.jpg", template)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
    if(write_files):
        cv2.imwrite("../output_images/Step4b_binaryExpandedLanes.jpg", template)
    # take the perspective transformed image and increase the dimensions
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) 
    result = cv2.addWeighted(warpage,1,template,0.5,0.0)
    if(write_files):
        cv2.imwrite("../output_images/Step4c_extractLanes.jpg", result)


    ########
    ########
    # Fitting a curve to the lanes
    #
    # fit the center of the boxes drawn in the previous step to a curve.
    
    # a range from 0 to the height of the input image
    # this will be combined with the coefficients of a second degree polynomial to create a smooth curve
    yvals = range(0, warped.shape[0]) 
    
    # the y values for the boxes in the previous step
    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height) 

    # find the coefficients of the second degree polynomial that fits the curve
    # along the lines that were generated in the previous step
    left_fit = np.polyfit(res_yvals, leftx, 2) # derive some coefficients
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2] # array of values x values that catch the coefficients
    left_fitx = np.array(left_fitx, np.int32) # round and turn to numpy array

    right_fit = np.polyfit(res_yvals, rightx, 2) # coefficients
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2] # x values
    right_fitx = np.array(right_fitx, np.int32) # round

    # some complex array math that is used to create (x,y) coordinates along the left and right lanes.
    # see the figures in the documentation to see a breakdown of the generated points
    left_lane = np.array(
        list(
            zip( # concatenate a list of x values (derived from the coefficients), with that same list in reverse
                 np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),       
                 # concat a static list of y vals from (0-height) with a list of yvals from height to 0
                 # 0, 1, 2, .. 719, 720, 720, 719, 718, ... 0
                 np.concatenate((yvals,yvals[::-1]), axis=0) 
                 )                                           
            ), np.int32)

    right_lane = np.array(
        list(
            zip( np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0), # x coordinates
                 np.concatenate((yvals,yvals[::-1]), axis=0) # y coordinates
                 )
            ), np.int32)

    inner_lane = np.array(
        list(
            zip( np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0), # x coordinates
                 np.concatenate((yvals,yvals[::-1]), axis=0) # y coordinates
                 )                                          
            ), np.int32)

    # create a canvas to draw the lanes on (left:blue, middle:green, right:red )
    road = np.zeros_like(img)
    cv2.fillPoly(road,[left_lane],color=[255,0,0])
    cv2.fillPoly(road,[inner_lane],color=[0,80,0])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])
    if(write_files):
        cv2.imwrite("../output_images/Step5a_fitLanes.jpg", road)

    # warp the current image from birds eye view to regular camera view.
    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    if(write_files):
        cv2.imwrite("../output_images/Step5b_fittedLanes.jpg", road_warped)

    # blend the original image with the colored lanes
    result = cv2.addWeighted(img, 1.0, road_warped, 1.0, 0.0)
    if(write_files):
        cv2.imwrite("../output_images/Step5c_mergedLaneImages.jpg", result)


    ########
    ########
    # Calculating the radius of the curvature and offset from center lane
    # 
    # the first step is to create a mapping from real world space to pixel space.
    # this mapping is used to calculate the radius of the curve the car is following,
    # and also the offset from the center of the lane.

    # create a mapping from real 3d space to pixel 2d space
    ym_per_pix = (10/720) # conversion from real space to pixel space (10 meters aproximetly 720 pixels)
    xm_per_pix = (4/384) # conversion from real space to pixel space (4 meters aproximetly 384 pixels)

    # use another polyfit to calculate the curvature of the left lane only
    curve_fit_cr = np.polyfit( np.array(res_yvals, np.float32) * ym_per_pix, # mapping yvalues to
                               np.array(leftx, np.float32) * xm_per_pix,     # xvalues, fitting a 2degree polynomial
                               2 )

    # this formula is derived and explained in the lessons.
    curverad = ( (1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix+curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

    # find last pixels on either lane, then find the midpoint 
    lane_center = ( left_fitx[-1]+right_fitx[-1] )/2

    # camera is mounted on the center of the car. so find the difference between the center of the image's
    # x axis and the center of the lanes is the offset of the car
    center_diff = (lane_center-warped.shape[1]/2) * xm_per_pix # also convert from pixel space to real space

    # calculate on which side of the center the car is at
    offset_side = 'right' if(center_diff<=0) else 'left'



    ########
    ########
    # Final write to the canvas
    # The final step is augment the calculation from the previous step onto
    # the frame.
    # 
    cv2.putText(result, "Curve Radius: "+str(round(curverad,2))+"m", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(result, "Drift from center: "+str(abs(round(center_diff,2)))+"m "+offset_side, (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    if(write_files):
        cv2.imwrite("../output_images/Step6_final.jpg", result)

    ####
    # end.
    return result

if __name__ == '__main__':
    img = cv2.imread("../test_images/test5.jpg")
    res = process_image(img, write_files=True)
