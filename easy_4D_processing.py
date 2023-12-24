#%%threaded
import numpy as np
import matplotlib.pyplot as plt
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

global save_location #TODO is this needed?

#TODO - add comments throughout

#TODO - fix set and make directory? Is it necessary now?

#TODO clean up code and imports

#TODO check imports for non-standard dependancies - easygui etc

#TODO test on F4 without new STEM server? Is there any point with the update coming so soon?

#TODO test refactored scan_4D_basic on microscope


def get_microscope_parameters(scan_width_px,use_precession,camera_frequency_hz):
    illumination_parameters = grpc_client.illumination.get_spot_size() #take illumination parameters from the microscope hardware
    fov = grpc_client.scanning.get_field_width() #get the current scanning FOV
    pixel_size_nm = fov*1e9/scan_width_px #work out the pixel size in nanometers
    microscope_info = { #creates a dictionary of microscope parameters
    "FOV in microns" : fov*1e6,
    "pixel size nm" : pixel_size_nm,
    "probe current in picoamps" : illumination_parameters[0]*1e12, #amps
    "convergence semiangle mrad" : illumination_parameters[1]*1e3,
    "d50 in nanometers" : illumination_parameters[2]*1e9,
    "defocus in meters" : grpc_client.illumination.get_condenser_defocus(),
    "diffraction size in mrad" : grpc_client.projection.get_max_camera_angle()*1e3,
    "Dwell time in ms": (1/camera_frequency_hz)*1e3}
    if use_precession is True:
        microscope_info["Using precssion"]="True"
        microscope_info["precession angle in mrad"] = grpc_client.scanning.get_precession_angle()*1e3
        microscope_info["Precession Frequency in HZ"] = grpc_client.scanning.get_precession_frequency()
    return microscope_info


def set_and_make_directory(): # TODO this is a semi-functional mess
    global save_location
    directory = str(Path.home()) + ("\\Desktop\\ExpertPI data\\")  # path to a data folder on the desktop
    now = datetime.now()  # time of data save process
    todays_date = now.strftime("%d_%m_%Y")  # date as a string
    date_and_time = now.strftime("%H_%M")  # time as a string
    save_location = directory + str(todays_date)  # adds the date to the data folder
    is_existing = os.path.exists(save_location)  # checks to see if there is an existing folder with this name and location
    if not is_existing:  # if the folder is not present
        os.makedirs(save_location)  # makes the folder
        #os.makedirs(final_directory)  # makes a subfolder for the time of the acquisition
    print("Save directory is", save_location)  # prints the save directory to the console
    return save_location, date_and_time

def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask

def coordinates_to_index(coordinates_list,num_pixels): #TODO can this be replaced with np.ravel_multi_index? #TODO Yes
    index_baseline = num_pixels
    coordinates_list = coordinates_list
    index_list = []
    for item in coordinates_list:
        index = item[0]+(index_baseline[0]*item[1])
        index_list.append(index)
    return index_list


def scan_4D_basic(scan_width_px=100,camera_frequency_hz=1000,use_precession=True,save_as=None): #refactored but partially functional on F4
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
    #data_folder_location,acquisition_time = set_and_make_directory() #creates a sub folder to save data to
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

    print("reshaping array")
    image_array = np.asarray(image_list) #converts the image list to an array
    image_array = np.reshape(image_array, (scan_width_px, scan_width_px, camera_size[0], camera_size[1])) #reshapes the array to match the acquisition

    if save_as == None:
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
        print("Pickling complete")
    return image_array



def selected_area_diffraction(image_array):

    camera_data_shape = image_array[0][0].shape #shape of first image to get image dimensions almost always 512x512
    print("Cam data shape",camera_data_shape)
    dataset_shape = image_array.shape[0],image_array.shape[1] #scanned region shape
    print("dataset shape",dataset_shape)
    radius = 30  # pixels
    VBF_intensity_list = []
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1], radius=radius)
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

def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask

def multi_point_VDF(image_array,radius=10):
    """
     Produces virtual dark field images from user selected points
     Arguments:
         image_list : numpy array of all diffraction patterns
         radius : virtual aperture radius in pixels, default is 10 pixels

    Returns:
        Summed diffraction pattern taken from all pixels
        List of Virtual Dark Field images
     """
    dataset_shape = image_array.shape[0], image_array.shape[1]
    dp_shape = image_array[0][0].shape


    num_pixels = dataset_shape[0]*dataset_shape[1]

    subset_images = []
    for i in range(0,3*dataset_shape[0]): #take a number of random images from the dataset
        random_image = image_array[np.random.randint(0,dataset_shape[0])][np.random.randint(0,dataset_shape[1])]
        subset_images.append(random_image.astype(np.float64))

    sum_diffraction = sum(subset_images)
    av_int = np.average(sum_diffraction)

    plt.title("Click to place virtual apertures, right click to remove and middle click to finish")
    plt.imshow(sum_diffraction,vmax=av_int*10)
    mask_list = plt.ginput(n=-1,show_clicks=True,timeout=0)
    plt.close()
    all_intensities = []
    for i in tqdm(image_array):
        for j in i:
            mask_intensities = []
            for mask_coords in mask_list:
                integration_mask = create_circular_mask(dp_shape[0], dp_shape[1], center=mask_coords, radius=radius)
                mask_intensity = np.sum(j[integration_mask])  # measures the intensity in the masked image
                mask_intensities.append(mask_intensity)

            all_intensities.append(mask_intensities)

    DF_images = []
    for mask in range(len(mask_list)):
        DF_output = [i[mask] for i in all_intensities]
        DF_output = np.reshape(DF_output,(dataset_shape))
        DF_images.append(DF_output)

    grid_side_length = ceil((len(mask_list)+1)**(1/2))
    plot_grid = gridspec.GridSpec(grid_side_length, grid_side_length) #TODO make the grid not always square
    fig = plt.figure(figsize=(10,10)) #TODO optimise this

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


def selected_area_diffraction_from_TIFF_series(x_pixels): #TODO fully update this to use new ideas

    input_directory = g.diropenbox("Find directory","find the file directory")
    src_images = input_directory.strip() + '\\*.tif'
    all_files = []
    for item in glob.glob(src_images):
        all_files.append(item)
    print(len(all_files))
    x_pixels = x_pixels
    y_pixels = int(len(all_files)/x_pixels)
    scan_dimensions = (x_pixels,y_pixels)
    print("dataset shape",x_pixels,"by",y_pixels,"pixels")
    radius = 20  # pixels
    VBF_intensity_list = []
    DP_list = []
    shape_check = cv2.imread(all_files[0],-1) #TODO do I need to know this?
    shape_check=(shape_check.shape[0],shape_check.shape[1])

    integration_mask = create_circular_mask(shape_check[0], shape_check[1], radius=radius)

    for file in range(len(all_files)):  # change to folder iteration loop
        dp = cv2.imread(all_files[file], -1)
        DP_list.append(dp)  # adds diffraction pattern to a list
        VBF_intensity = np.sum(dp[integration_mask])  # measures the intensity in the masked image - for VBF image
        VBF_intensity_list.append(VBF_intensity)

    VBF_intensity_list = np.asarray(VBF_intensity_list)
    VBF_intensity_list = np.reshape(VBF_intensity_list, scan_dimensions)
    plt.imshow(VBF_intensity_list)
    plt.title("Click to add points, right click to remove previous point, middle click to finsh and complete polygon")
    plt.gray()
    #plt.show()
    coords = plt.ginput(n=-1,show_clicks=True) #use 4 clicks to define the integration region
    plt.close()

    polygon = patches.Polygon(coords,fill=False,edgecolor="red")
    poly = matpath(coords)
    all_pixel_coordinates = []
    inside_pixels = []
    points = scan_dimensions
    for i in range(0,points[0]):
        for j in range(0,points[1]):
            all_pixel_coordinates.append([i,j])  # this can probably be done more elegantly
    for pixel in all_pixel_coordinates:
        is_inside = poly.contains_point(pixel)
        if is_inside == True:
            #print("True")
            inside_pixels.append(pixel)

    index_list = coordinates_to_index(inside_pixels,scan_dimensions) #todo replace with ravel
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.title.set_text("Integration region")
    ax1.add_patch(polygon)
    ax1.imshow(VBF_intensity_list)

    subset_DP_list = []
    for index in index_list:
        pattern = DP_list[index].astype(np.float64)
        subset_DP_list.append(pattern)

    subset_summed_DP = sum(subset_DP_list)
    subset_summed_DP = np.asarray(subset_summed_DP,dtype=np.float64)

    zero_excluded_average = np.average(subset_summed_DP[~integration_mask])
    zero_excluded_max = max(subset_summed_DP[~integration_mask])
    #print("max intensity outside of zero order disk",zero_excluded_max)
    #print("average outside of zero order disk",zero_excluded_average)

    ax2.imshow(subset_summed_DP,vmin=0,vmax=zero_excluded_max)
    title_text = "Summed diffraction pattern from {} patterns".format(len(inside_pixels))
    ax2.title.set_text(title_text)
    plt.show()

    return subset_summed_DP, polygon, VBF_intensity_list


def save_as_tiffs(image_array,output_resolution=None):
    directory = g.diropenbox("select directory to save to","select save directory")
    i = 0
    for row in image_array:
        for pixel in row:
            pixel.astype(np.uint16)
            if output_resolution is not None:
                pixel = cv2.resize(pixel,[output_resolution,output_resolution])
            filename = f"{directory}\\4D_stem_{i:06}.tiff"
            cv2.imwrite(filename,pixel)
            i += 1

def tiffs_to_array(scan_width):
    directory = g.diropenbox("Select directory","Select Directory")
    print(directory)
    something = os.listdir(directory)
    list = []
    for file in tqdm(something):
        #print(file)
        path = directory+"\\"+file
        image = cv2.imread(path,-1)
        list.append(image)

    frame = list[0]
    shape = frame.shape()
    array = np.asarray(list)
    array = np.reshape(array,(scan_width,scan_width,shape[0],shape[1]))
    return array

def tiffs_to_array(scan_width):
    directory = g.diropenbox("Select directory","Select Directory")
    print(directory)
    something = os.listdir(directory)
    list = []
    for file in tqdm(something):
        #print(file)
        path = directory+"\\"+file
        image = cv2.imread(path,-1)
        list.append(image)

    return list


#array = tiffs_to_array(256)