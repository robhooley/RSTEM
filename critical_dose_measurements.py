
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
from operator import itemgetter
import skimage.exposure
from scipy.integrate import simps
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from scipy.signal import find_peaks, savgol_filter
from skimage import restoration, feature, transform
import math
from math import sqrt
import pandas as pd
from collections import Counter

from bisect import bisect_left
from expert_pi.RSTEM.easy_4D_processing import scan_4D_basic
from expert_pi.RSTEM.utilities import get_microscope_parameters
from expert_pi import grpc_client
from expert_pi.stream_clients import cache_client
from expert_pi.controllers import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT
#import json

from utilities import get_microscope_parameters,get_number_of_nav_pixels,calculate_dose,create_circular_mask

def closest_coord(coord,coord_list):
    """Finds the coordinate in a list closest to the specified coordinate"""
    distances = []
    for item in range(len(coord_list)):
        distance_between_coords = math.dist(coord,coord_list[item])
        distances.append(distance_between_coords)

    smallest_distance=min(distances)
    #print(smallest_distance)
    coord_index = distances.index(smallest_distance)
    closest_coordinate = coord_list[coord_index]
    return closest_coordinate

def get_template_spot_positions(processed_template,spot_size_in_pixels):
    """Cross correlation for image filtering"""
    spot_template = np.ones([16,16],dtype=np.float32)#define spot template background
    #spot_template = spot_template.astype(np.float32)
    mask = create_circular_mask(spot_template.shape[0],spot_template.shape[1],mask_radius=spot_size_in_pixels,mask_center_coordinates=(spot_template.shape[0]/2,spot_template.shape[1]/2))
    mask = mask*255
    spot_template = spot_template*(mask == 255)
    img_for_template_matching = processed_template.astype(np.float32)
    spot_locations = cv2.matchTemplate(img_for_template_matching,spot_template,cv2.TM_CCOEFF_NORMED)
    thresholded = spot_locations>0.2

    #plt.imshow(thresholded)
    #plt.show()

    thresholded_eroded = cv2.erode((thresholded.astype("uint8")),kernel = np.ones((3, 3), np.uint8),iterations=1)
    inverted = np.invert(thresholded_eroded) #invert image contrast
    inverted = cv2.equalizeHist(inverted) #equalise histogram to fix binarisation issue

    #plt.imshow(inverted)
    #plt.show()

    """# create the params and deactivate the 3 filters"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.filterByColor = False
    params.maxArea = 100
    params.minArea = 6
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByCircularity = True
    params.minCircularity = 0.5

    """detect the blobs"""
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)
    """display them"""

    template_spots = []
    spot_sizes = []
    for i in range(len(keypoints)): #change coordinates to camera centered TODO this should really be centered around the refined center
        template_spots_x = keypoints[i].pt[0] + ((spot_template.shape[0]/2))
        template_spots_y = keypoints[i].pt[1] + ((spot_template.shape[0]/2))
        template_spots.append((template_spots_x,template_spots_y))
        spot_sizes.append(keypoints[i].size)

    distances = []


    plt.rcParams["figure.figsize"] = [15, 15]
    template_intensity = np.max(processed_template)/10

    plt.imshow(processed_template,vmax=template_intensity)

    plt.title("Click on crosswise pairs of diffraction spots to enhance the center positioning")
    crosswise_coords = plt.ginput(n=-1,timeout=0)
    plt.close()

    """convert coordinates to be centered around camera center""" # TODO this should really be centered around the refined center
    centered_polygon_coords = []
    for coord in range(len(crosswise_coords)):
        coord_x = crosswise_coords[coord][0] + ((spot_template.shape[0] / 2)) #-1
        coord_y =  crosswise_coords[coord][1] + ((spot_template.shape[0] / 2)) #-1
        centered_polygon_coords.append((coord_x,coord_y))


    refinement_coords = []
    for coord in range(len(centered_polygon_coords)):

        closest_spot = closest_coord(centered_polygon_coords[coord],template_spots)
        refinement_coords.append(closest_spot)
        #circle = Circle(closest_spot, 5, fill=False, color="blue")
        #ax.add_patch(circle)

    polygon = np.asarray(refinement_coords,dtype=np.float64)
    polygon2 = np.roll(polygon, -1, axis=0)
    signed_areas = 0.5 * np.cross(polygon, polygon2)
    centroids = (polygon + polygon2) / 3.0
    refined_center_spot_position = np.average(centroids, axis=0, weights=signed_areas)

    """threshold out any distances below 10 pixels from the center"""

    distances = []
    list_mask = []
    for item in range(len(template_spots)):
        spot = template_spots[item]
        distance = math.dist(spot,refined_center_spot_position) #
        #print("distance of spot",item, "is",distance,"pixels")
        distances.append(distance)
        if distance < 10 or distance > 256:
            list_mask.append(False)
        else:
            list_mask.append(True)

    """janky list based masking of spots"""
    thresholded_spots = []
    for spot in range(len(template_spots)):
        if list_mask[spot] == True:
            thresholded_spots.append(template_spots[spot])

    fig, ax = plt.subplots(1)
    ax.imshow(processed_template, vmax=template_intensity)
    for spot in range(len(thresholded_spots)):
        circle = Circle(thresholded_spots[spot], 5, fill=False, color="blue")
        ax.add_patch(circle)
        ax.arrow(thresholded_spots[spot][0],thresholded_spots[spot][1], 2, 0, color="blue")
        ax.arrow(thresholded_spots[spot][0],thresholded_spots[spot][1], 0, 2, color="blue")
        ax.arrow(thresholded_spots[spot][0],thresholded_spots[spot][1], -2, 0, color="blue")
        ax.arrow(thresholded_spots[spot][0],thresholded_spots[spot][1], 0, -2, color="blue")

    circle = Circle(refined_center_spot_position, 10, fill=False, color="red")
    ax.add_patch(circle)
    ax.arrow(refined_center_spot_position[0],refined_center_spot_position[1],5,0,color="red")
    ax.arrow(refined_center_spot_position[0], refined_center_spot_position[1], 0, 5, color="red")
    ax.arrow(refined_center_spot_position[0], refined_center_spot_position[1], -5, 0, color="red")
    ax.arrow(refined_center_spot_position[0], refined_center_spot_position[1], 0, -5, color="red")

    plt.show()

    return template_spots,refined_center_spot_position

def create_spot_masks_list(template_spots,image,integration_mask_radius):
    """Generates the spot masks so the function is only called once"""
    h, w = image.shape[:2] #TODO refactor
    spot_mask_list = []
    for i in range(len(template_spots)):  # for every spot in the template
        center = int(template_spots[i][1]),int(template_spots[i][0])
        integration_mask = create_circular_mask(h, w, center, radius=integration_mask_radius)
        spot_mask_list.append(integration_mask)
    return spot_mask_list

#TODO move mask generation to a separate function so they are only created once
def get_spot_intensities(template_spots,image,spot_mask_list):
    spot_intensities = []
    """    h, w = image.shape[:2]
    spot_masks = []
    for i in range(len(template_spots)):  # for every spot in the template
        center = int(template_spots[i][1]),int(template_spots[i][0])
        integration_mask = create_circular_mask(h, w, center, radius=integration_mask_radius)
        spot_masks.append(integration_mask)"""

    for i in range(len(template_spots)):  # for every spot in the template
        integration_mask = spot_mask_list[i]
        integrated_intensity = np.sum(image[integration_mask]) # measures the intensity in the mask
        spot_intensities.append(integrated_intensity)  # adds the intensity to the list of spot intensities
    return spot_intensities

def acquire_datapoint(num_pixels,camera_frequency_hz,output="sum",use_precession=True): #checked ok
    point = scan_4D_basic(num_pixels,camera_frequency_hz,use_precession)
    camera_data = point[0]
    if output == "sum":
        output_data = np.sum(camera_data,(0,1),dtype=np.float32) #sums 4D acquisition to single diffraction pattern
    else:
        output_data = camera_data #gives an array where shape is shape4D (scanX,scanY,cameraX,cameraY)
    return output_data

def beam_size_matched_acquisition(camera_FPS=4500,pixels=32): #checked ok
    optical_mode = grpc_client.microscope.get_optical_mode()

    beam_size = grpc_client.illumination.get_beam_diameter() #in meters

    matched_sampling_fov = beam_size*pixels #calculates FOV for 1:1 beam:pixel sampling
    grpc_client.scanning.set_field_width(matched_sampling_fov) #set fov in meters
    diffraction_pattern = acquire_datapoint(pixels,camera_FPS,"sum",use_precession=True) #acquires a 4D-dataset and sums to 1 diffraction pattern
    return diffraction_pattern,matched_sampling_fov
