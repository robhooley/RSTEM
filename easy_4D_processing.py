import numpy as np
import matplotlib.pyplot as plt
import json


from matplotlib import patches, gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle as p
import easygui as g
import cv2 as cv2
import os
from time import sleep
from tqdm import tqdm
from matplotlib.path import Path as matpath
import fnmatch
import matplotlib.colors as mcolors

#comment this out depending on where the script is located
from expert_pi.RSTEM.utilities import create_circular_mask,get_microscope_parameters,spot_radius_in_px,check_memory,collect_metadata #utilities file in RSTEM directory
#from utilities import create_circular_mask,get_microscope_parameters,spot_radius_in_px,check_memory

from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT, CondenserFocusType as CFT
from serving_manager.api import TorchserveRestManager
from expert_pi.app import app
from expert_pi.gui import main_window

window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client





def scan_4D_basic(scan_width_px=128,camera_frequency_hz=4500,use_precession=False):
    """Parameters
    scan width: pixels
    camera_frequency: camera speed in frames per second up to 72000
    use_precession: True or False
    returns a tuple of (image_array, metadata)
    """

    sufficient_RAM = check_memory(camera_frequency_hz,scan_width_px)
    if sufficient_RAM == False:
        print("This dataset might not fit into RAM, trying anyway")

    #metadata = get_microscope_parameters(scan_width_px,use_precession,camera_frequency_hz) #gets the microscope and acquisition metadata
    metadata = collect_metadata(acquisition_type="Camera",scan_width_px=scan_width_px,use_precession=use_precession,pixel_time=1/camera_frequency_hz,scan_rotation=0,edx_enabled=False,camera_pixels=512)

    if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(DT.HAADF) == True: #if either STEM detector is inserted
        grpc_client.stem_detector.set_is_inserted(DT.BF,False) #retract BF detector
        grpc_client.stem_detector.set_is_inserted(DT.HAADF, False) #retract ADF detector
        for i in tqdm(range(5),desc="Stabilising after STEM detector retraction",unit=""):
            sleep(1) #wait for 5 seconds
    grpc_client.projection.set_is_off_axis_stem_enabled(False) #puts the beam back on the camera if in off-axis mode
    sleep(0.2)  # stabilisation after deflector change
    scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/camera_frequency_hz, 8), total_size=scan_width_px, frames=1, detectors=[DT.Camera], is_precession_enabled=use_precession)
    print("Acquiring",scan_width_px,"x",scan_width_px,"px dataset at",camera_frequency_hz,"frames per second")
    image_list = [] #empty list to take diffraction data
    for i in tqdm(range(scan_width_px),desc="Retrieving data from cache",total=scan_width_px,unit="chunks"): #retrives data one scan row at a time to avoid crashes
        header, data = cache_client.get_item(scan_id, scan_width_px)  # cache retrieval in rows
        camera_size = data["cameraData"].shape[1],data["cameraData"].shape[2] #gets shape of diffraction patterns
        for j in range(scan_width_px): #for each pixel in that row
            image_data = data["cameraData"][j] #take the data for that pixel
            image_data = np.asarray(image_data) #convers to numpy array
            image_data = np.reshape(image_data,camera_size) #reshapes data to an individual image
            image_list.append(image_data) #adds it to the list of images

    print("reshaping array") #reshaping the array to match the 4D STEM acquisition
    image_array = np.asarray(image_list) #converts the image list to an array
    del image_list #flush image list to clear out RAM
    image_array = np.reshape(image_array, (scan_width_px, scan_width_px, camera_size[0], camera_size[1])) #reshapes the array to match the acquisition
    print("Array reshaped")

    return (image_array,metadata) #tuple with image data and metadata



def selected_area_diffraction(data_array):
    """Takes a 4D data array as produced by scan_4D_basic and allows the user to select virtual apertures in the image
    to integrate diffraction from"""

    if type(data_array) is tuple: #checks for metadata dictionary
        image_array = data_array[0]
    else:
        image_array = data_array

    camera_data_shape = image_array[0][0].shape #shape of first image to get image dimensions
    dataset_shape = image_array.shape[0],image_array.shape[1] #scanned region shape
    radius = 30  # pixels for rough VBF image construction
    VBF_intensity_list = [] #empty list to take virtual bright field image sigals
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1], mask_radius=radius)
    for row in image_array: #iterates through array rows
        for pixel in row: #in each row iterates through pixels
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
    subset_DP_list = [] #empty list to take the diffraction patterns inside the polygon
    for pixel in inside_pixels: #for each pixel inside the polygon
        pattern = image_array[pixel[1]][pixel[0]] #take the diffraction pattern
        pattern.astype(np.float64) #convert it to 64 bit (better for summation)
        subset_DP_list.append(pattern) #add it to the empty list
    subset_array = np.asarray(subset_DP_list)
    subset_summed_DP = np.sum(subset_array,0,dtype=np.float64)
    zero_excluded_max = max(subset_summed_DP[~integration_mask]) #highest intensity outside the zero order disk

    export_polygon = patches.Polygon(coords,fill=False,edgecolor="red") #creates a polygon from the user defined integration region
    export_fig = plt.figure(figsize=(10, 10)) #defines an empty figure to add the export to
    export_axis = export_fig.add_subplot() #add plot to exporting figure
    plt.setp(export_axis, xticks=[], yticks=[])  # removes tick markers
    export_axis.add_patch(export_polygon) #adds a circle for every mask used

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0) #removes whitespace
    plt.margins(0, 0) #removes white space
    plt.imshow(VBF_intensity_array) #add image to canvas

    canvas = FigureCanvasAgg(export_fig) #defines a canvas
    export_fig.canvas.draw() #plots the canvas
    buffered = canvas.buffer_rgba() #writes it to a buffer
    annotated_image = np.asarray(buffered) #converts the buffer to an image
    plt.close() #close the figure

    polygon = patches.Polygon(coords, fill=False, edgecolor="red")
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10)) #defines a 1x2 plot of reasonable size
    ax1.title.set_text("Integration region")

    ax1.add_patch(polygon)  # shows the polygon of the integration window in the calibrated image
    ax1.imshow(VBF_intensity_array)
    ax2.imshow(subset_summed_DP.astype(np.float64),vmin=0,vmax=zero_excluded_max) #show the image and scale to that intensity
    ax2.title.set_text(f"Summed diffraction pattern from {number_of_summed_patterns} patterns") #add title
    plt.show() #show plot

    return subset_summed_DP, annotated_image #return the summed image, the polygon and the VBF image

#checked ok
def multi_VDF(data_array,radius=None):
    """
     Produces virtual dark field images from user selected points
     Arguments:
         image_list : numpy array of all diffraction patterns
         radius : virtual aperture radius in pixels, default is 10 pixels
    Returns:
        Summed diffraction pattern taken from all pixels
        List of Virtual Dark Field images
     """

    if type(data_array) is tuple: #checks for metadata dictionary
        image_array = data_array[0]
        metadata = data_array[1]
        if radius is None:
            radius =  (metadata["Predicted diffraction spot diameter (px)"]/2)*1.3
            print(f"The predicted spot radius is {radius} pixels")
    else:
        image_array = data_array
        #metadata=None
        #print("Metadata not present")
        if radius is None:
            radius = 20

    dataset_shape = image_array.shape[0], image_array.shape[1]
    dp_shape = image_array[0][0].shape

    subset_images = []
    for i in range(0,5*dataset_shape[0]): #take a number of random images from the dataset
        random_image = image_array[np.random.randint(0,dataset_shape[0])][np.random.randint(0,dataset_shape[1])]
        subset_images.append(random_image.astype(np.float64))

    sum_diffraction = sum(subset_images) #sums the random images to make a representative diffraction pattern
    av_int = np.average(sum_diffraction) #calculates the average intensity of this pattern
    plt.figure(figsize=(10,10)) #defines plot size
    plt.gray() #sets it to grayscale
    plt.title("Click to place virtual apertures, right click to remove and middle click to finish, max masks 8")
    plt.imshow(sum_diffraction,vmax=av_int*10) #plots the representative diffraction and scales the intensity
    mask_list = plt.ginput(n=8,show_clicks=True,timeout=0) #user interacts to define mask positions
    plt.close() #closes the plot
    all_mask_intensities = [] #empty list for the mask data

    integration_masks = []
    for mask_coords in mask_list:
        integration_mask = create_circular_mask(dp_shape[0], dp_shape[1], mask_center_coordinates=mask_coords,
                                                mask_radius=radius)
        integration_masks.append(integration_mask)

    for row in tqdm(image_array):
        for pixel in row:
            mask_intensities = [] #empty list of mask intensities per pixel
            for mask_coords in range(len(mask_list)): #for each mask
                integration_mask = integration_masks[mask_coords] #takes mask from list and applies it
                mask_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
                mask_intensities.append(mask_intensity) #adds to the list

            all_mask_intensities.append(mask_intensities) #adds each pixels list to a list (nested lists)

    DF_images = [] #empty list for DF images to be added to
    for mask in range(len(mask_list)): #for every mask in the list
        DF_output = [i[mask] for i in all_mask_intensities] #this works but I don't really understand why
        DF_output = np.reshape(DF_output,(dataset_shape)) #reshapes the DF intensities to the scan dimensions
        DF_images.append(DF_output) #adds DF images to a list


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

    plot_grid = gridspec.GridSpec(grid_rows, grid_cols) #defines the plot grid

    av_int = np.average(sum_diffraction)  # calculates the average intensity in the diffraction pattern, for display scaling
    colors = list(mcolors.TABLEAU_COLORS)  # sets a list of colours
    export_fig = plt.figure(figsize=(10, 10))
    export_axis = export_fig.add_subplot()
    plt.imshow(sum_diffraction, vmin=0, vmax=av_int*10)
    plt.setp(export_axis, xticks=[], yticks=[])  # removes tick markers
    for coords in range(len(mask_list)):
        circle = plt.Circle(mask_list[coords], radius=radius, color=colors[coords], fill=True,alpha=0.3) #defines the mask annotation
        export_axis.add_patch(circle) #adds a circle for every mask used

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0) #removes whitespace
    plt.margins(0, 0) #removes white space
    canvas = FigureCanvasAgg(export_fig) #defines a canvas
    export_fig.canvas.draw() #plots the canvas
    buffered = canvas.buffer_rgba() #writes it to a buffer
    annotated_image = np.asarray(buffered) #converts the buffer to an image
    plt.close()

    fig = plt.figure(figsize=(grid_cols*5,grid_rows*5)) #makes a figure of reasonable

    ax=fig.add_subplot(plot_grid[0]) #adds the ax axis to the plot grid in the zero position
    ax.title.set_text("Summed diffraction with integration windows") #adds title
    av_int = np.average(sum_diffraction) #calculates the average intensity in the diffraction pattern, for display scaling
    plt.imshow(annotated_image)
    plt.setp(ax, xticks=[], yticks=[]) #removes tick markers
    colors = list(mcolors.TABLEAU_COLORS) #sets a list of colours

    for i in range(len(mask_list)): #for every mask used
        ax=fig.add_subplot(plot_grid[i+1]) #selects the plotting grid position for the DF to go in
        ax.imshow(DF_images[i],cmap="gray") #adds the DF image
        ax.title.set_text(i+1) #mask number
        ax.title.set_color(colors[i]) #colour code for title
        ax.spines['bottom'].set_color(colors[i])#colour rim
        ax.spines['bottom'].set_linewidth(3)#line thickness
        ax.spines['top'].set_color(colors[i])#colour rim
        ax.spines['top'].set_linewidth(3)#line thickness
        ax.spines['right'].set_color(colors[i])#colour rim
        ax.spines['right'].set_linewidth(3)#line thickness
        ax.spines['left'].set_color(colors[i]) #colour rim
        ax.spines['left'].set_linewidth(3) #line thickness
        plt.setp(ax, xticks=[], yticks=[]) #removes ticks
    plt.gray() #sets plots to be grayscale
    plt.show() #shows the plot
    #plt.close()

    return annotated_image ,sum_diffraction,DF_images #annotated image is scaled to show the final figure scale which is small #TODO make this better, maybe plot it again before export?

def save_data(data_array,format=None,output_resolution=None):
    """Handles data saving for scan4D_basic
    Parameters
    data_array: from scan_4D_basic, either with or without metadata
    format: default None
    output_resolution:Default None, used to set scaling of diffraction patterns, enter 128 or 256, will scale to square only
    """
    if type(data_array) is tuple: #checks for metadata dictionary
        image_array = data_array[0]
        metadata = data_array[1]
        print("Metadata exists")
    else:
        image_array = data_array
        metadata=None
        print("Metadata not present")

    directory = g.diropenbox("select directory to save to", "select save directory")
    print("Preparing for data saving")
    filename = directory + "\\4D-STEM_"

    formats=["TIFFs","Numpy array","Pickle"]
    if format == None:
        format = g.choicebox("Select format for data to be saved","select format for data to be saved",formats,preselect=1)
    shape_4D = image_array.shape
    if format == "TIFFs":
        print("Saving",shape_4D[0]*shape_4D[1],"files as .tiff")
        i = 0
        for row in tqdm(image_array):
            for image in row:
                if output_resolution is not None:  # handles rescaling if used
                    image = cv2.resize(image, [output_resolution, output_resolution])
                filename = f"{directory}\\4D_stem_{i:06}.tiff" #increments the number of the frame
                cv2.imwrite(filename, image) #writes the frame
                i += 1 #increases the number for the next frame
        print("Saving complete") #status update
    elif format == "Numpy array":
        num_files_in_dir = len(fnmatch.filter(os.listdir(directory), '*.npy'))
        filename=filename+f"{num_files_in_dir+1}"
        if output_resolution is not None:
            print(f"Binning diffraction patterns to {output_resolution}x{output_resolution}px")
            resized_images = []
            for row in tqdm(image_array,unit="Chunks"):
                for image in row:
                    image_resized = cv2.resize(image,dsize=[output_resolution,output_resolution])
                    resized_images.append(image_resized)
            image_array = np.asarray(resized_images)
            image_array = np.reshape(image_array, (shape_4D[0], shape_4D[1], output_resolution, output_resolution))

        print(f"Saving to numpy array, {filename}.npy")
        np.save(filename, image_array)
        print("Saving complete")
    elif format == "Pickle":
        num_files_in_dir = len(fnmatch.filter(os.listdir(directory), '*.pdat'))
        filename = filename + f"{num_files_in_dir + 1}"
        if output_resolution is not None:
            print(f"Binning diffraction patterns to {output_resolution}x{output_resolution}px")
            resized_images = []
            for row in tqdm(image_array,unit="Chunks"):
                for image in row:
                    image_resized = cv2.resize(image,dsize=[output_resolution,output_resolution])
                    resized_images.append(image_resized)
            image_array = np.asarray(resized_images)
            image_array = np.reshape(image_array, (shape_4D[0], shape_4D[1], output_resolution, output_resolution))
        print(f"saving as Pickle with metadata {filename}.pdat file")
        with open (f"{filename}.pdat","wb")as f:
            p.dump((image_array,metadata),f)
        print("Pickling complete")

    if metadata is not None:
        metadata_name = directory + f"\\4D-STEM_metadata{num_files_in_dir+1}.json"
        open_json = open(metadata_name,"w")
        json.dump(metadata,open_json,indent=6)
        open_json.close()

#checked ok
def import_tiff_series(scan_width=None):
    """Loads in a folder of TIFFs and creates a 4D-array for use with other functions"""
    directory = g.diropenbox("Select directory","Select Directory")
    if scan_width is None: #if scan width variable is empty, prompt user to enter it
        num_files = len(fnmatch.filter(os.listdir(directory), '*.tiff')) #counts how many .tiff files are in the directory
        guessed_scan_width = int(np.sqrt(num_files)) #assumes it is a square acquisition
        scan_width=g.enterbox(f"Enter scan width in pixels, there are {num_files} TIFF files in this folder, "
                                f"scan width might be {guessed_scan_width}","Enter scan width in pixels",
                                default=str(guessed_scan_width))

        scan_width=int(scan_width)
        scan_height = int(num_files/scan_width)
    load_metadata = g.ynbox("Do you want to load the metadata .JSON file?","Do you want to load the metadata"
                                                                           " .JSON file")
    if load_metadata == False:
        metadata=None
    if load_metadata == True:
        metadata_path = g.fileopenbox("Select metadata file","Select metadata file")
        loading_container = open(metadata_path,"r")
        metadata = json.load(loading_container)

    folder = os.listdir(directory) #formats the directory into proper syntax
    image_list = [] #opens empty list
    for file in tqdm(folder): #iterates through folder with a progress bar
        path = directory+"\\"+file #picks individual images
        if file.endswith(".tiff"):
            image = cv2.imread(path,-1) #loads them with openCV
            image_list.append(image) #adds them to a list of all images

    array = np.asarray(image_list) #converts the list to an array
    cam_pixels_x,cam_pixels_y = image_list[0].shape
    reshaped_array = np.reshape(array,(scan_width,scan_height,cam_pixels_x,cam_pixels_y)) #reshapes the array to 4D dataset shape #TODO confirm ordering of scan width and height with non-square dataset
    if load_metadata == False:
        return reshaped_array #just a data array reshaped to the 4D STEM acquisition shape
    else:
        return (reshaped_array,metadata) #tuple containing the data array and the metadata dictionary
