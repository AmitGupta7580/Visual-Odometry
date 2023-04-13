import pyrealsense2.pyrealsense2 as rs
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def extract_features(image, detector='sift', mask=None):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
    elif detector == 'surf':
        det = cv2.xfeatures2d.SURF_create()
        
    kp, des = det.detectAndCompute(image, mask)
    
    return kp, des
    
    
def match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image
    matching -- (str) can be 'BF' for Brute Force or 'FLANN'
    detector -- (str) can be 'sift or 'orb'. Default is 'sift'
    sort -- (bool) whether to sort matches by distance. Default is True
    k -- (int) number of neighbors to match to each feature.

    Returns:
    matches -- list of matched features from two images. Each match[i] is k or less matches for 
               the same query descriptor
    """
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=k)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=k)
    
    if sort:
        matches = sorted(matches, key = lambda x:x[0].distance)

    return matches
    
    
def filter_matches_distance(matches, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    for m, n in matches:
        if m.distance <= dist_threshold*n.distance:
            filtered_match.append(m)

    return filtered_match
    
    
def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    return image_matches
    
    
def estimate_motion(match, kp1, kp2, intr, depth1, max_depth=2000):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    intr -- camera intrinsic calibration matrix 
    
    Optional arguments:
    depth1 -- Depth map of the first frame. Set to None to use Essential Matrix decomposition
    max_depth -- Threshold of depth to ignore matched features. 3000 is default

    Returns:
    rmat -- estimated 3x3 rotation matrix
    tvec -- estimated 3x1 translation vector
    image1_points -- matched feature pixel coordinates in the first image. 
                     image1_points[i] = [u, v] -> pixel coordinates of i-th match
    image2_points -- matched feature pixel coordinates in the second image. 
                     image2_points[i] = [u, v] -> pixel coordinates of i-th match
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in match])
    
    k = np.eye(3)
    cx = k[0, 2] = intr.ppx
    cy = k[1, 2] = intr.ppy
    fx = k[0, 0] = intr.fx
    fy = k[1, 1] = intr.fy
    object_points = np.zeros((0, 3))
    delete = []
    
    # Extract depth information of query image at match points and build 3D positions
    for i, (u, v) in enumerate(image1_points):
        z = depth1[int(v), int(u)]
        if z > max_depth:
            delete.append(i)
            continue
                
        # Use arithmetic to extract x and y (faster than using inverse of k)
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy
        object_points = np.vstack([object_points, np.array([x, y, z])])

    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)
    
    print(len(object_points), len(delete))
        
    # Use PnP algorithm with RANSAC for robustness to outliers
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
    #print('Number of inliers: {}/{} matched features'.format(len(inliers), len(match)))
        
    # Above function returns axis angle rotation representation rvec, use Rodrigues formula
    # to convert this to our desired format of a 3x3 rotation matrix
    rmat = cv2.Rodrigues(rvec)[0]
    
    return rmat, tvec, image1_points, image2_points
    
    
def visual_odometry(camera, detector='sift', matching='BF', filter_match_distance=None, 
                    plot=False):
    '''
    Function to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Takes as input a Data_Handler object and optional parameters.
    
    Arguments:
    manager -- Camera Manager object
    
    Optional Arguments:
    detector -- (str) can be 'sift' or 'orb'. Default is 'sift'.
    matching -- (str) can be 'BF' for Brute Force or 'FLANN'. Default is 'BF'.
    filter_match_distance -- (float) value for ratio test on matched features. Default is None.
    plot -- (bool) whether to plot the estimated vs ground truth trajectory. Only works if
                   matplotlib is set to tk mode. Default is False.
    
    Returns:
    trajectory -- Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    
    '''
    if plot:
        fig = plt.figure(figsize=(14, 14))
        # ax = fig.add_subplot(projection='3d')
        # ax.view_init(elev=-20, azim=270)
        
    # Establish homogeneous transformation matrix. First pose is identity    
    T_tot = np.eye(4)
    trajectory = T_tot[:3, :].reshape(1, 3, 4)
    
    profile = camera.pipeline.start(camera.config)
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    
    time.sleep(3)
    image_plus1, depth_plus1 = camera.get_feed()
    
    frame_cnt = 1
    
    # Iterate through all frames of the sequence
    while True:
        start = datetime.datetime.now()
        
        image, depth = image_plus1, depth_plus1
        image_plus1, depth_plus1 = camera.get_feed()
            
        # Get keypoints and descriptors for left camera image of two sequential frames
        kp1, des1 = extract_features(image, detector, None)
        kp2, des2 = extract_features(image_plus1, detector, None)
        
        # Get matches between features detected in the two images
        matches_unfilt = match_features(des1, des2, 
                                        matching=matching, 
                                        detector=detector, 
                                        sort=True)
        
        # Filter matches if a distance threshold is provided by user
        if filter_match_distance is not None:
            matches = filter_matches_distance(matches_unfilt, filter_match_distance)
        else:
            matches = matches_unfilt
            
        # Estimate motion between sequential images of the left camera
        rmat, tvec, img1_points, img2_points = estimate_motion(matches, kp1, kp2, intr, depth)
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        T_tot = T_tot.dot(np.linalg.inv(Tmat))
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory = np.vstack((trajectory, T_tot[:3, :].reshape(1, 3, 4)))
        
        # End the timer for the frame and report frame rate to user
        end = datetime.datetime.now()
        print('Time to compute frame {}:'.format(frame_cnt), end-start)
        
        if plot:
            xs = trajectory[:frame_cnt+1, 2, 3]
            ys = trajectory[:frame_cnt+1, 1, 3]
            # zs = trajectory[:frame_cnt+1, 2, 3]
            plt.plot(xs, ys, c='chartreuse')
            plt.pause(1e-32)
            
        frame_cnt += 1
            
    if plot:
        plt.close()
        
    return trajectory


class Camera_Manager:
    def __init__(self, width, height):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)

    def get_feed(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())
        
        return color, depth
