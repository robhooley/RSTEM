#%%threaded
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.pyplot import Circle
from matplotlib import patches, gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import pickle as p
import easygui as g
import cv2 as cv2
import glob
from datetime import datetime
import os
import time
from time import sleep
from statistics import mean
from tqdm import tqdm
from skimage.transform import radon
from skimage import exposure
from pathlib import Path
from math import ceil

from expert_pi import grpc_client
from expert_pi.controllers import scan_helper
from expert_pi.stream_clients import cache_client
from expert_pi.grpc_client.modules._common import DetectorType as DT
from matplotlib.path import Path as matpath
import matplotlib.colors as mcolors

#TODO - add comments throughout
#TODO clean up code and imports
#TODO check imports for non-standard dependancies - easygui etc
#TODO test refactored scan_4D_basic on microscope


def get_microscope_parameters(scan_width_px,use_precession,camera_frequency_hz):
    fov = grpc_client.scanning.get_field_width() #get the current scanning FOV
    pixel_size_nm = fov*1e3/scan_width_px #work out the pixel size in nanometers
    time_now = datetime.now()
    acquisition_time = time_now.strftime("%d_%m_%Y %H_%M")
    microscope_info = { #creates a dictionary of microscope parameters
    "FOV in microns" : fov*1e6,
    "pixel size nm" : pixel_size_nm,
    "probe current in picoamps" : grpc_client.illumination.get_current()*1e12,
    "convergence semiangle mrad" : grpc_client.illumination.get_convergence_half_angle()*1e3,
    "beam diameter (d50) in nanometers" : grpc_client.illumination.get_beam_diameter()*1e9,
    "diffraction size in mrad" : grpc_client.projection.get_max_camera_angle()*1e3,
    "Dwell time in ms": (1/camera_frequency_hz)*1e3,
    "Acquisition date and time":acquisition_time}
    if use_precession is True:
        microscope_info["Using precssion"]="True"
        microscope_info["precession angle in mrad"] = grpc_client.scanning.get_precession_angle()*1e3
        microscope_info["Precession Frequency in HZ"] = grpc_client.scanning.get_precession_frequency()
    return microscope_info


def create_circular_mask(image_height, image_width, mask_center_coordinates=None, mask_radius=None):

    if mask_center_coordinates is None:  # use the middle of the image
        mask_center_coordinates = (int(image_width/2), int(image_height/2))
    if mask_radius is None:  # use the smallest distance between the center and image walls
        mask_radius = min(mask_center_coordinates[0], mask_center_coordinates[1], image_width - mask_center_coordinates[0], image_height - mask_center_coordinates[1])

    Y, X = np.ogrid[:image_height, :image_width]
    dist_from_center = np.sqrt((X - mask_center_coordinates[0])**2 + (Y - mask_center_coordinates[1])**2)

    mask = dist_from_center <= mask_radius
    return mask

"""def coordinates_to_index(coordinates_list,num_pixels): #TODO can this be replaced with np.ravel_multi_index? #TODO Yes
    index_baseline = num_pixels
    coordinates_list = coordinates_list
    index_list = []
    for item in coordinates_list:
        index = item[0]+(index_baseline[0]*item[1])
        index_list.append(index)
    return index_list"""


def scan_4D_basic(scan_width_px=100,camera_frequency_hz=1000,use_precession=True,save_as=None): #refactored but partially functional on F4
    #TODO add in metadata and add to tuple in image array variable
    """Parameters
    scan width: pixels
    camera_frequency: camera speed in frames per second up to 72000
    use_precession: True or False
    save_as : "None" does not save and keeps the data in RAM,"TIFF" saves a folder full of TIFFs,
    "NPY","Pickle" """

    if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(DT.HAADF) == True:
        grpc_client.stem_detector.set_is_inserted(DT.BF,False)
        grpc_client.stem_detector.set_is_inserted(DT.HAADF, False)
        print("Stabilising after detector retraction")
        sleep(5)
    grpc_client.scanning.set_precession_frequency(72000)
    grpc_client.projection.set_is_off_axis_stem_enabled(False) #puts the beam back on the camera
    sleep(0.2)  # stabilization
    scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/camera_frequency_hz, 8), total_size=scan_width_px, frames=1, detectors=[DT.Camera], is_precession_enabled=use_precession)
    print("Acquiring",scan_width_px,"x",scan_width_px,"px dataset at",camera_frequency_hz,"frames per second")
    image_list = []
    for i in range(scan_width_px): #retrives data one scan row at a time to avoid crashes
        print("getting data", i, "/", scan_width_px)
        header, data = cache_client.get_item(scan_id, scan_width_px)  # cache retrieval in rows
        camera_size = data["cameraData"].shape[1],data["cameraData"].shape[2]
        for j in range(scan_width_px):
            image_data = data["cameraData"][j]
            image_data = np.asarray(image_data)
            image_data = np.reshape(image_data,camera_size) #reshapes data to an individual image
            image_list.append(image_data)

    metadata = get_microscope_parameters(scan_width_px,use_precession,camera_frequency_hz)

    print("reshaping array")
    image_array = np.asarray(image_list) #converts the image list to an array
    image_array = np.reshape(image_array, (scan_width_px, scan_width_px, camera_size[0], camera_size[1])) #reshapes the array to match the acquisition
    print("Array reshaped")
    """    if save_as == None:
        print("Image array stored in RAM only")
    else:
        print("Preparing for data saving")
        directory = g.diropenbox("select directory to save to", "select save directory")
        time_now = datetime.now()
        filename = directory + "\\4D-STEM_" + time_now.strftime("%d_%m_%Y %H_%M")
    if save_as == "TIFF":
        print("Saving",len(image_list),"files as .TIFF")
        i = 0
        if camera_frequency_hz>=2250: #faster than 2250 FPS, dectris camera only produces 8 bit images
            data_type = np.uint8
        else:
            data_type = np.uint16 #if slower than 2250 FPS, camera can produce 16 bit images
        for row in image_array:
            print("Saving row",row,"of",scan_width_px)
            for image in row:
                image.astype(data_type) # sets data type based on camera speed
                filename = f"{directory}\\4D_stem_{i:06}.tiff" #increments the number of the frame
                cv2.imwrite(filename, image) #writes the frame
                i += 1 #increases the number for the next frame
        print("Saving complete") #status update
    elif save_as == "NPY" or "npy":
        print("Saving to numpy array")
        np.save(filename, image_array)
        print("Saving complete")
    elif save_as == "Pickle" or "pickle":
        print("saving as Pickle")
        with open (f"{filename}.pdat","wb")as f:
            p.dump(image_array,f)
        print("Pickling complete")"""
    return (image_array,metadata)



def selected_area_diffraction(data_array):

    if type(data_array) is tuple: #checks for metadata dictionary
        image_array = data_array[0]
        metadata = data_array[0]
        print("Metadata exists")
    else:
        image_array = data_array
        metadata=None
        print("Metadata not present")

    camera_data_shape = image_array[0][0].shape #shape of first image to get image dimensions almost always 512x512
    print("Cam data shape",camera_data_shape)
    dataset_shape = image_array.shape[0],image_array.shape[1] #scanned region shape
    print("dataset shape",dataset_shape)
    radius = 30  # pixels
    VBF_intensity_list = []
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1], mask_radius=radius)
    for j in image_array:
        for k in j:
            VBF_intensity = np.sum(k[integration_mask])  # measures the intensity in the masked image - for zero order determination
            VBF_intensity_list.append(VBF_intensity)

    VBF_intensity_array = np.asarray(VBF_intensity_list)
    VBF_intensity_array = np.reshape(VBF_intensity_array, (dataset_shape[0], dataset_shape[1]))
    plt.imshow(VBF_intensity_array)

    plt.title("Click to add points, right click to remove previous point, middle click to finsh and complete polygon")
    plt.gray()
    coords = plt.ginput(n=-1,show_clicks=True,timeout=0) #use 4 clicks to define the integration region
    plt.close()

    polygon = patches.Polygon(coords,fill=False,edgecolor="red")
    poly = matpath(coords) #draws a polygon around the coordinates extracted from the image

    all_pixel_coordinates = []
    inside_pixels = []
    #num_pixels = image_array.shape[0],image_array.shape[1]
    #print("num pixels",num_pixels)

    points = image_array.shape[0],image_array.shape[1]  #change this to root of number of pixels #TODO do this better
    for i in range(0,points[0]):
        for j in range(0,points[1]):
            all_pixel_coordinates.append([i,j])  # this can probably be done more elegantly
    for pixel in all_pixel_coordinates:
        is_inside = poly.contains_point(pixel)
        if is_inside == True:
            #print("True")
            inside_pixels.append(pixel)

    number_of_summed_patterns = len(inside_pixels)
    print(number_of_summed_patterns,"patterns summed together")

    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.title.set_text("Integration region")
    ax1.add_patch(polygon)
    ax1.imshow(VBF_intensity_list)


    subset_DP_list = []
    for pixel in inside_pixels:
        pattern = image_array[pixel[1]][pixel[0]]
        pattern.astype(np.float64)
        subset_DP_list.append(pattern)

    subset_summed_DP = sum(subset_DP_list)

    zero_excluded_average = int(np.average(subset_summed_DP[~integration_mask]))
    zero_excluded_max = max(subset_summed_DP[~integration_mask])
    print("max intensity outside of zero order disk",zero_excluded_max)
    print("average outside of zero order disk",zero_excluded_average)


    ax2.imshow(subset_summed_DP,vmin=0,vmax=zero_excluded_max)
    ax2.title.set_text(("Summed diffraction pattern from",number_of_summed_patterns,"patterns"))

    plt.show()

    return subset_summed_DP, polygon, VBF_intensity_list

"""def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask"""

def multi_VDF(data_array,radius=10):
    """
     Produces virtual dark field images from user selected points
     Arguments:
         image_list : numpy array of all diffraction patterns
         radius : virtual aperture radius in pixels, default is 10 pixels

    Returns:
        Summed diffraction pattern taken from all pixels
        List of Virtual Dark Field images
     """

    if type(data_array) is tuple: #checks for metadata dictionary #TODO test this
        image_array = data_array[0]
        metadata = data_array[0]
        print("Metadata exists")
    else:
        image_array = data_array
        metadata=None
        print("Metadata not present")


    dataset_shape = image_array.shape[0], image_array.shape[1]
    dp_shape = image_array[0][0].shape

    num_pixels = dataset_shape[0]*dataset_shape[1]

    subset_images = []
    for i in range(0,3*dataset_shape[0]): #take a number of random images from the dataset
        random_image = image_array[np.random.randint(0,dataset_shape[0])][np.random.randint(0,dataset_shape[1])]
        subset_images.append(random_image.astype(np.float64))

    sum_diffraction = sum(subset_images)
    av_int = np.average(sum_diffraction)

    plt.title("Click to place virtual apertures, right click to remove and middle click to finish, max masks 8")
    plt.imshow(sum_diffraction,vmax=av_int*10)
    mask_list = plt.ginput(n=8,show_clicks=True,timeout=0)
    plt.close()
    all_mask_intensities = []
    for row in tqdm(image_array):
        for pixel in row:
            mask_intensities = []
            for mask_coords in mask_list:
                integration_mask = create_circular_mask(dp_shape[0], dp_shape[1], mask_center_coordinates=mask_coords, mask_radius=radius)
                mask_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
                mask_intensities.append(mask_intensity)

            all_mask_intensities.append(mask_intensities)

    DF_images = []
    for mask in range(len(mask_list)):
        DF_output = [i[mask] for i in all_mask_intensities] #TODO make this more intuitive

        DF_output = np.reshape(DF_output,(dataset_shape)) #reshapes the DF intensities to the scan dimensions
        DF_images.append(DF_output)

    if len(mask_list) ==1:
        grid_rows,grid_cols=1,2 #1x2 plot for 1 mask
    elif len(mask_list) ==2:
        grid_rows, grid_cols = 1, 3 #1x3 plot for 2 masks +1DP
    elif len(mask_list) == 3:
        grid_rows, grid_cols = 2, 2 #2x2 plot for 3 masks +1DP
    elif len(mask_list) <= 5:
        grid_rows, grid_cols = 2, 3
    elif len(mask_list) <= 7:
        grid_rows, grid_cols = 2,4  #2x4 plot for 7 masks +1DP
    elif len(mask_list) ==8:
        grid_rows, grid_cols = 3, 3 #3x3 plot for 8 +1DP


    plot_grid = gridspec.GridSpec(grid_rows, grid_cols)


    fig = plt.figure(figsize=(grid_cols*5,grid_rows*5))

    ax=fig.add_subplot(plot_grid[0])
    ax.title.set_text("Summed diffraction with integration windows")
    av_int = np.average(sum_diffraction)
    max_int = np.max(sum_diffraction)
    print("average intensity",av_int)
    print("max intensity",max_int)
    plt.imshow(sum_diffraction, vmin=0, vmax=av_int*10)
    plt.setp(ax, xticks=[], yticks=[])
    colors = list(mcolors.TABLEAU_COLORS)
    for coords in range(len(mask_list)):
        circle = plt.Circle(mask_list[coords], radius=radius, color=colors[coords], fill=False)
        ax.add_patch(circle)
        # ax.annotate(str(coords+1),(mask_list[coords][0]-10,mask_list[coords][1]-15),color=colors[coords])
    canvas = FigureCanvasAgg(fig)
    fig.canvas.draw()
    buf = canvas.buffer_rgba()
    buffer = np.asarray(buf)
    for i in range(len(mask_list)):
        ax=fig.add_subplot(plot_grid[i+1])
        ax.imshow(DF_images[i],cmap="gray")
        ax.title.set_text(i+1)
        ax.title.set_color(colors[i])
        ax.spines['bottom'].set_color(colors[i])
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['top'].set_color(colors[i])
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_color(colors[i])
        ax.spines['right'].set_linewidth(3)
        ax.spines['left'].set_color(colors[i])
        ax.spines['left'].set_linewidth(3)
        #bounding_box = plt.Rectangle(facecolor="none",color=colors[i])
        #ax.add_patch()
        plt.setp(ax, xticks=[], yticks=[])
    plt.gray()
    plt.show()
    return buffer,sum_diffraction,DF_images


def save_as_tiffs(data_array,output_resolution=None): #TODO remove

    """Save an array to 16 bit TIFFs
    Use output_resolution variable to rescale the images"""
    image_array = data_array[0]  # TODO confirm this works
    metadata = data_array[1]  # TODO confirm this works
    directory = g.diropenbox("Select directory to save to","Select save directory")
    i = 0
    for row in tqdm(image_array): #iterates row by row
        for pixel in row: #each pixel in each row
            pixel.astype(np.uint16) #sets image type to 16 bit
            if output_resolution is not None: #handles rescaling if used
                pixel = cv2.resize(pixel,[output_resolution,output_resolution])
            filename = f"{directory}\\4D_stem_{i:06}.tiff" #names files and defines format
            cv2.imwrite(filename,pixel)  #saves the image
            i += 1 #increments file name


def save_data(data_array,format=None):

    if type(data_array) is tuple: #checks for metadata dictionary #TODO test this
        image_array = data_array[0]
        metadata = data_array[0]
        print("Metadata exists")
    else:
        image_array = data_array
        metadata=None
        print("Metadata not present")

    #TODO take tuple called image_array and strip out metadata, then add metadata json save
    """Handles data saving for scan4D_basic"""
    print("Preparing for data saving")
    directory = g.diropenbox("select directory to save to", "select save directory")
    time_now = datetime.now()
    filename = directory + "\\4D-STEM_" + time_now.strftime("%d_%m_%Y %H_%M")
    formats=["All images as TIFFs","Numpy array","Pickle"]
    if format == None:
        format = g.choicebox("Select format for data to be saved","select format for data to be saved",formats,preselect="All images as TIFFs")

    shape_4D = image_array.shape
    if format == "All images as TIFFs":
        print("Saving",shape_4D[0]*shape_4D[1],"files as .TIFF")
        i = 0
        #if camera_frequency_hz>=2250: #faster than 2250 FPS, dectris camera only produces 8 bit images
        #    data_type = np.uint8
        #else:
        #    data_type = np.uint16 #if slower than 2250 FPS, camera can produce 16 bit images
        for row in image_array:
            print("Saving row",row,"of",shape_4D[0])
            for image in row:
                #image.astype(data_type) # sets data type based on camera speed
                filename = f"{directory}\\4D_stem_{i:06}.tiff" #increments the number of the frame
                cv2.imwrite(filename, image) #writes the frame
                i += 1 #increases the number for the next frame
        print("Saving complete") #status update
    elif format == "Numpy array":
        print("Saving to numpy array")
        np.save(filename, image_array)
        print("Saving complete")
    elif format == "Pickle":
        print("saving as Pickle")
        with open (f"{filename}.pdat","wb")as f:
            p.dump(image_array,f)
        print("Pickling complete")



def import_tiff_series(scan_width=None):
    """Loads in a folder of TIFFs and creates a 4D-array for use with other functions"""
    directory = g.diropenbox("Select directory","Select Directory")
    if scan_width==None: #if scan width variable is empty, prompt user to enter it
        scan_width=g.integerbox("Enter scan width in pixels","Enter scan width in pixels")
    folder = os.listdir(directory) #formats the directory into proper syntax
    image_list = [] #opens empty list
    for file in tqdm(folder): #iterates through folder with a progress bar
        path = directory+"\\"+file #picks individual images
        image = cv2.imread(path,-1) #loads them with openCV
        image_list.append(image) #adds them to a list of all images

    array = np.asarray(image_list) #converts the list to an array
    reshaped_array = np.reshape(array,(scan_width,scan_width)) #reshapes the array to 4D dataset shape
    metadata=None
    return (reshaped_array,metadata)

#dataset = np.load("C:\\Users\\robert.hooley\\Desktop\\Coding\\4D-STEM_18_12_2023 15_31.npy")
#
#buffer,sum,df = multi_VDF(dataset)