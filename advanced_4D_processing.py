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
from expert_pi import grpc_client
from expert_pi.controllers import scan_helper
from expert_pi.stream_clients import cache_client
from expert_pi.grpc_client.modules._common import DetectorType as DT, RoiMode as RM


from expert_pi.RSTEM.utilities import create_circular_mask,get_microscope_parameters,spot_radius_in_px,create_scalebar,check_memory #utilities file in RSTEM directory
#from utilities import create_circular_mask,get_microscope_parameters,spot_radius_in_px,create_scalebar,check_memory

#TODO scan 4D basic but with ROI mode
#TODO and direct saving to disk in temp folder

#TODO needs STEM server 0.10
def scan_4D_fast(scan_width_px=128,camera_frequency_hz=18000,use_precession=False,roi_mode=128):
    """Parameters
    scan width: pixels
    camera_frequency: camera speed in frames per second up to 72000
    use_precession: True or False
    roi_mode: optional variable to enable ROI mode, either 128,256 or False
    returns a tuple of (image_array, metadata)
    """

    sufficient_RAM = check_memory(camera_frequency_hz,scan_width_px,roi_mode)
    if sufficient_RAM == False:
        print("This dataset will probably not fit into RAM, trying anyway but expect a crash")
     #gets the microscope and acquisition metadata
    if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(DT.HAADF) == True: #if either STEM detector is inserted
        grpc_client.stem_detector.set_is_inserted(DT.BF,False) #retract BF detector
        grpc_client.stem_detector.set_is_inserted(DT.HAADF, False) #retract ADF detector
        for i in tqdm(range(5),desc="stabilising after detector retraction",unit=""):
            sleep(1) #wait for 5 seconds
    grpc_client.projection.set_is_off_axis_stem_enabled(False) #puts the beam back on the camera if in off-axis mode
    sleep(0.2)  # stabilisation after deflector change
    metadata = get_microscope_parameters(scan_width_px,use_precession,camera_frequency_hz)

    if roi_mode==128: #512x128 px
        grpc_client.scanning.set_camera_roi(roi_mode=RM.Lines_128, use16bit=False)
        camera_shape=(128,512)
    elif roi_mode==256: #512x256 px
        grpc_client.scanning.set_camera_roi(roi_mode=RM.Lines_256,use16bit=False)
        camera_shape=(256,512)
    else:
        grpc_client.scanning.set_camera_roi(roi_mode=RM.Disabled,use16bit=True)
     #sets to ROI mode
    print(grpc_client.scanning.get_camera_roi())
    scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/camera_frequency_hz, 8), total_size=scan_width_px, frames=1, detectors=[DT.Camera], is_precession_enabled=use_precession)
    print("Acquiring",scan_width_px,"x",scan_width_px,"px dataset at",camera_frequency_hz,"frames per second")
    image_list = [] #empty list to take diffraction data
    for i in tqdm(range(scan_width_px),desc="Retrieving data from cache",total=scan_width_px,unit="rows"): #retrives data one scan row at a time to avoid crashes
        header, data = cache_client.get_item(scan_id, scan_width_px)  # cache retrieval in rows
        camera_size = camera_shape#data["cameraData"].shape[1],data["cameraData"].shape[2] #gets shape of diffraction patterns
        image_data = data["cameraData"] #take the data for that row
        #image_data = np.asarray(image_data) #convers to numpy array
        image_row = np.reshape(image_data,(scan_width_px,camera_size[0],camera_size[1])) #reshapes data to an individual image #TODO necessary?
        image_list.append(image_row) #adds it to the list of images
    """This should give a scan width length list with scan width length lists in it"""
    image_array = np.asarray(image_list) #converts the image list to an array
    del image_list #flush image list to clear out RAM
    return (image_array,metadata) #tuple with image data and metadata


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