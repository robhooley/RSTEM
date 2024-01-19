import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import patches, gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle as p
import easygui as g
import cv2 as cv2
from datetime import datetime
import os
from time import sleep
from tqdm import tqdm
from matplotlib.path import Path as matpath
import fnmatch
import matplotlib.colors as mcolors
from matplotlib import transforms as tf
#from expert_pi import grpc_client
#from expert_pi.controllers import scan_helper
#from expert_pi.stream_clients import cache_client
#from expert_pi.grpc_client.modules._common import DetectorType as DT


#from expert_pi.RSTEM.utilities import create_circular_mask,get_microscope_parameters,spot_radius_in_px,create_scalebar #utilities file in RSTEM directory
from utilities import create_circular_mask,get_microscope_parameters,spot_radius_in_px,create_scalebar






#TODO use metadata for scalebar, maybe common function for all measurements?
def calibrated_selected_area_diffraction(data_array):
    """Takes a 4D data array as produced by scan_4D_basic and allows the user to select virtual apertures in the image
    to integrate diffraction from"""

    if type(data_array) is tuple: #checks for metadata dictionary #TODO Checked ok with and without metadata
        image_array = data_array[0]
        metadata = data_array[1]
        print("Metadata exists")

    else:
        image_array = data_array
        metadata=None
        print("Metadata not present")

    camera_data_shape = image_array[0][0].shape #shape of first image to get image dimensions
    dataset_shape = image_array.shape[0],image_array.shape[1] #scanned region shape
    radius = 30  # pixels for rough VBF image construction
    VBF_intensity_list = [] #empty list to take virtual bright field image sigals
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1], mask_radius=radius)
    for row in image_array: #iterates through array
        for pixel in row:
            VBF_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
            VBF_intensity_list.append(VBF_intensity) #adds to the list

    VBF_intensity_array = np.asarray(VBF_intensity_list) #converts list to array
    VBF_intensity_array = np.reshape(VBF_intensity_array, (dataset_shape[0], dataset_shape[1])) #reshapes array to match image dimensions
    plt.figure(figsize=(10,10)) #sets figure size large enough for clear display
    plt.imshow(VBF_intensity_array) #plots VBF

    plt.title("Click to add points, right click to remove previous point, middle click to finsh and complete polygon")
    plt.gray() #sets grayscale
    coords = plt.ginput(n=-1,show_clicks=True,timeout=0) #use user mouse input to define the integration region
    plt.close()

    polygon = patches.Polygon(coords,fill=False,edgecolor="red") #creates a polygon from the user defined intergration region
    poly = matpath(coords) #draws a polygon around the coordinates extracted from the image

    all_pixel_coordinates = [] #list for all possible coordinates
    inside_pixels = [] #list for coordinates within the polygon

    """this is hacky but works"""
    points = image_array.shape[0],image_array.shape[1]
    for i in range(0,points[0]):
        for j in range(0,points[1]):
            all_pixel_coordinates.append([i,j])  # this can probably be done more elegantly
    for pixel in all_pixel_coordinates: #for all possible pixels
        is_inside = poly.contains_point(pixel) #checks if the pixel is inside the polygon
        if is_inside == True: #if so, then
            inside_pixels.append(pixel) #adds internal pixel to list

    number_of_summed_patterns = len(inside_pixels) #number of pixels within the polygon



    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10)) #defines a 1x2 plot of reasonable size
    ax1.title.set_text("Integration region")

    """pixel_to_axis_scale = (metadata["FOV in microns"]*1e3)/image_array.shape[0]


    trans = tf.Affine2D().scale(pixel_to_axis_scale) + ax1.transData #coordinate system transform from image to spatial scale

    poly = patches.Polygon(np.c_[coords], facecolor='red', edgecolor='red', fill=False,
                   transform=trans) #creates a polygon using the real spatial scale

    ax1.add_patch(poly) #shows the polygon of the integration window in the calibrated image"""

    ax1.add_patch(polygon)  # shows the polygon of the integration window in the calibrated image

    #if metadata is not None:
        #flipped = np.flipud(VBF_intensity_array)
        #create_scalebar(ax1,10,metadata)
        #ax1.imshow(flipped,extent=[0,metadata["FOV in microns"]*1e3,0,metadata["FOV in microns"]*1e3]) #shows the VBF image under it
        #ax1.invert_yaxis()
    #else:
    #    ax1.imshow(VBF_intensity_array)  # shows the VBF image under it

    ax1.imshow(VBF_intensity_array)

    subset_DP_list = [] #empty list to take the diffraction patterns inside the polygon
    for pixel in inside_pixels: #for each pixel inside the polygon
        pattern = image_array[pixel[1]][pixel[0]] #take the diffraction pattern
        pattern.astype(np.float64) #convert it to 64 bit (better for summation)
        subset_DP_list.append(pattern) #add it to the empty list

    subset_summed_DP = sum(subset_DP_list) #sum the list together


    zero_excluded_max = max(subset_summed_DP[~integration_mask]) #highest intensity outside the zero order disk
    ax2.imshow(subset_summed_DP,vmin=0,vmax=zero_excluded_max) #show the image and scale to that intensity
    ax2.title.set_text(("Summed diffraction pattern from",number_of_summed_patterns,"patterns")) #add title

    plt.show() #show plot
    #TODO figure out how to use the buffer and canvas part of the multi VDF script to make this export an annotated image
    return subset_summed_DP, polygon, VBF_intensity_array #return the summed image, the polygon and the VBF image