import numpy as np
import cv2

class centroid_tracker():
    # for every single frame in a video or stream, create this object to track lane lines
    # window width and window height are search dimensions
    # padding is how far past the dimensions to search
    # smoothing is used to average out the points
    # the sliding window approach is used to find the lane lines
    def __init__(self, window_width, window_height, padding, smoothing=15):
        self.window_width = window_width
        self.window_height = window_height
        self.padding = padding
        self.smoothing = smoothing
        self.recent_centers = []

    # sliding window technique using convolutions
    def find_window_centroids(self, warped):
        window_centroids = []
        window_width = self.window_width
        window_height = self.window_height
        padding = self.padding
        
        # a template for the sliding window
        window = np.ones(window_width) # a flat array of 1's that is window_width long. ex: [1,1,1,1] if width=4 
        
        # find the fist points along the bottom of the image that start off the lanes
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0) # create a squash signal of the bottom
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)    
        # append the first pair here
        window_centroids.append((l_center,r_center))
    
        # search each layer for pixel peaks
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # squash signal
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            
            offset = window_width/2 # center of box area
            l_min_index = int(max(l_center+offset-padding,0))
            l_max_index = int(min(l_center+offset+padding,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            r_min_index = int(max(r_center+offset-padding,0))
            r_max_index = int(min(r_center+offset+padding,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            window_centroids.append((l_center, r_center)) # append output
           
        # merge calculated centers to recent centers
        self.recent_centers.append(window_centroids)
        return np.average(self.recent_centers[-self.smoothing:], axis=0) # index into the last collected n indexes, average, return
