import math
import pickle as p
import numpy as np
import easygui as g
from tqdm import tqdm
from serving_manager.api import TorchserveRestManager
import pandas as pd
from bisect import bisect_left
#from bda_functions import calculate_dose
import pickle
import matplotlib.pyplot as plt
import cv2 as cv2
from collections import Counter
import scipy

from expert_pi.RSTEM.bda_functions import calculate_dose, beam_size_matched_acquisition


#TODO V1.0 offline analysis

def pointInRect(point,rectangle):
    x1, y1, x2, y2 = rectangle
    y, x = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    else:
        return False

def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number - valor))

    return aux.index(min(aux))

def create_circular_mask(image_height, image_width, mask_center_coordinates=None, mask_radius=None):
    if mask_center_coordinates is None:  # use the middle of the image
        mask_center_coordinates = (int(image_width/2), int(image_height/2))
    if mask_radius is None:  # use the smallest distance between the center and image walls
        mask_radius = min(mask_center_coordinates[0], mask_center_coordinates[1], image_width - mask_center_coordinates[0], image_height - mask_center_coordinates[1])
    Y, X = np.ogrid[:image_height, :image_width]
    dist_from_center = np.sqrt((X - mask_center_coordinates[0])**2 + (Y - mask_center_coordinates[1])**2)
    mask = dist_from_center <= mask_radius
    return mask


def dose_series(max_dose=False,num_steps=False,directory=None,dwell_time=None):
    """This should do a calulation of the current hardware dose and run a
    time resolved 4D-STEM acquisition that sums the data to a single pattern, saving to a pdat file"""

    if dwell_time is None:
        dwell_time = 56e-6 #4 precession cycles, dwell in seconds

    current_dose_values = calculate_dose()
    current_dose_rate = current_dose_values["Probe dose rate-A-2s-1"]
    print("Current dose rate",np.round(current_dose_rate,2),"e-A-2s-1")
    print("Dose step increment",np.round(current_dose_rate*dwell_time,1),"e-A-2")

    if directory is None:
        directory = g.diropenbox("Select save directory","Save directory")
        
    filename = directory+"\\dose series.pdat"
    

    dose_steps_needed = int((max_dose/current_dose_rate)*1.1) #additional 10% buffer
    if max_dose==False:
        steps_needed=num_steps
    else :
        steps_needed=dose_steps_needed #either uses dose target or number of steps
    dataset_list = []
    dose_list = []
    dose_offset = current_dose_values["Probe dose e-A-2"]*3 #assume 3 navigation scans to get the region selected
    dose_increment = current_dose_rate*dwell_time
    dose_list.append(dose_offset) #navigation dose rather than zero dose
    
    for step in tqdm(range(1,steps_needed+1)):
        image = beam_size_matched_acquisition(pixels=8,dwell_time_s=dwell_time,output="sum")
        dataset_list.append(image)
        dose_list.append((dose_increment+dose_list[-1]))

    with open(filename,"wb") as f:
        p.dump((dataset_list,dose_list),f)

    return dataset_list,dose_list


def get_spot_positions_ML(image,threshold=0): #TODO deprecate

    image = image.astype(np.uint16) #TODO temporary rescale to uint16

    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host='172.16.2.86',
                                    image_encoder='.tiff')
    #results = manager.infer(image=image, model_name='diffraction_spot_segmentation_medium')  # spot detection
    results = manager.infer(image=image, model_name='spot_segmentation')
    spots = results["objects"]
    spot_list = []
    areas = []
    for i in range(len(spots)):
        if spots[i]["mean_intensity"] > threshold and spots[i]["area"] > 5:
            spot_coords = spots[i]["center"]
            spot_list.append(spot_coords)
            area = spots[i]["area"]
            areas.append(area)
    #for i in spot_list:
    #    plt.plot(i[0],i[1],"r+")
    #    plt.imshow(image,vmax=np.average(image*10),extent=(0,1,1,0),cmap="gray")


    #plt.show(block=False)
    return spot_list

def get_spot_intensities(dataset,spot_radius,plot=False):
    """:parameter: directory to the data series as string, filenames should increase with dose
    :parameter: spot radius in pixels, governs the size of the integration mask

    :returns: Normalised and raw integrated intensities in separate dataframes, with a list for the
    doses received by each frame"""

    global_spot_intensities = []  # holds the non-normalised data for plotting
    image_doses = dataset[1]
    del image_doses[0]
    images = dataset[0]

    template_positions = get_spot_positions_ML(dataset[0][0],threshold=0.01) #list of spot centers
    print("Number of spots",len(template_positions))

    spot_index = list(np.arange(0, len(template_positions)))

    height, width = images[0].shape[:2]

    #for i in template_positions:
    #    plt.plot(i[0], i[1], "r+")
    #    plt.imshow(images[0], vmax=np.average(images[0]*10), extent=(0, 1, 1, 0), cmap="gray")

    #plt.show(block=False)


    for frame in tqdm(range(len(images))):
        spot_intensities = []
        image = images[frame]  # load in as grayscale
        #frame_dose = image_doses[frame]  # calculate frame dose using previously specified info
        #print("Frame number",frame)
        for i in spot_index:  # for every spot in the template
            #print("Fractional position",template_positions[i])
            spot_center = (template_positions[i][0]*height,template_positions[i][1]*height) #convert from fraction of image to pixels
            integration_mask = create_circular_mask(height, width, mask_center_coordinates=spot_center, mask_radius=spot_radius)  # creates a
            integrated_intensity = np.sum(image[integration_mask])  # measures the intensity in the mask
            #print("spot center",spot_center,"spot intensity",integrated_intensity)
            spot_intensities.append(integrated_intensity)  # adds the intensity to the list of spot intensities
        global_spot_intensities.append(spot_intensities)  # writes all spot intensities to a single list, frame wise



    """ converts the lists into a dataframe and sets the dose values to be the index column"""
    integrated_spot_intensities = pd.DataFrame.from_records(global_spot_intensities, index=image_doses, exclude=None,
                                            columns=spot_index)
    integrated_spot_intensities.reset_index()

    """normalise the initial data (not nat log)"""
    normalised_intensities = integrated_spot_intensities.div(integrated_spot_intensities.iloc[0])  # normalises to the first spot intensity
    # iloc[0] is the first value in each column
    return (integrated_spot_intensities,normalised_intensities,image_doses,template_positions,images) #normalised raw intensities in separate dataframes, and doses

def count_spots(dataset,threshold=0):

    spot_count = []

    for image in tqdm(range(len(dataset[0]))):
        pattern = dataset[0][image]
        spot_fitting = get_spot_positions_ML(pattern,threshold=threshold)
        num_spots = len(spot_fitting)
        spot_count.append(num_spots)

    return spot_count


def plot_results(results,filepath=None):

    #if filepath == None:
    #    filepath = g.diropenbox("Select directry to save results","Select save directory")

    dataframe = results[0]
    normalised_dataframe = results[1]
    doses = results[2]
    spot_positions = results[3]
    images = results [4]



    """plot setup"""

    fig,ax2= plt.subplots(1, 1) #(ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.tight_layout(pad=2)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    """Plot setup"""
    #for column in dataframe.columns:
    #    ax1.set_title("Integrated intensities")
    #    ax1.scatter(doses, dataframe[column])  # plots spot intensity against accumulated dose
        # ax1.axvline(critical_dose)
    #    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
    #    ax1.set_xlabel("Accumulated dose e$^-$A$^-2$")
    #    ax1.set_ylabel("Normalised integrated intensity")

    ax2.set_title("Template image")
    ax2.imshow(images[0])

    for i in spot_positions:
        ax2.scatter(i[0],i[1],marker="+",color="r")
        ax2.imshow(images[0],vmax=np.average(images[0]*10),extent=(0,1,1,0),cmap="Greys")
        ax2.annotate(spot_positions.index(i),(i[0],i[1]),color="r")

    #ax3.imshow(images[-1],vmax=np.average(images[-1]*10),extent=(0,1,1,0),cmap="Greys")
    #ax3.set_title("Last image")

    plt.show()


def full_series_spot_tracking(dataset,threshold=0): #TODO deprecate single function and use this one
    # TODO make this work on individual patterns

    full_fits_per_frame = []
    spot_count = []
    spot_positions = []
    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host='172.16.2.86',
                                    image_encoder='.tiff')

    image_series=dataset[0]
    doses = dataset[1]

    for image in tqdm(image_series):
        image = image.astype(np.uint16) #TODO temporary rescale to uint16 can use float 32 now
        results = manager.infer(image=image, model_name='spot_segmentation')
        spots = results["objects"]
        full_fits_per_frame.append(results) #full spot_fitting
        spot_counts_frame = len(spots)
        spot_count.append(spot_counts_frame) #number of spots detected
        spot_list = [] #image by image list of spot coordinates
        for i in range(len(spots)):
            if spots[i]["mean_intensity"] > threshold:
                spot_coords = spots[i]["center"]
                spot_list.append(spot_coords)
        spot_positions.append(spot_list) #frame by frame list of spot coordinates

        return spot_count,full_fits_per_frame,spot_positions,doses,image_series

def calculate_garman_limit_spot_number(num_spots,doses):
    approx_garman = int(max(num_spots)/2)
    print("Approx Garman limit",approx_garman)

    garman_closest_index = closest(num_spots,approx_garman)

    #print(num_spots)
    print("Closest frame to the Garman limit",num_spots[garman_closest_index])
    #print(dataset[1])

    garman_dose = dataset[1][garman_closest_index]
    print("The accumulated dose at the Garman limit",garman_dose)
    print(num_spots[garman_closest_index])
    #print(len(num_spots))
    #print(len(dataset[1]))
    #print("len dataset before deletion",len(dataset[1]))
    del dataset[1][0]
    #print("len dataset after deletion",len(dataset[1]))
    print(len(num_spots))

    plt.scatter(dataset[1],num_spots,marker=".",c="red")
    plt.xlabel("Accumulated dose (e-A-2")
    plt.suptitle("Number of diffraction spots as a function of accumulated dose, at room temperature")
    plt.title(f"{max(num_spots)} initial spots, reduced by 50% after {np.round(garman_dose,1)}e-A-2")
    plt.hlines(num_spots[garman_closest_index],xmin=0,xmax=garman_dose+2,colors="black")
    plt.vlines(garman_dose,ymax=num_spots[garman_closest_index]*1.2,ymin=0,colors="black")
    plt.ylabel("Number of detected diffraction spots")
    plt.show()

    return garman_dose


def calculate_spot_lifetime(full_fitting,doses):

    energy = 100e3 #TODO get from metadata
    max_cam_angle_rads =0.14 #TODO get from metadata
    camera_frame_size = 512 #TODO get from image series or ML fitting

    phir = energy*(1 + scipy.constants.e*energy/(2*scipy.constants.m_e*scipy.constants.c**2))
    g_vec = np.sqrt(2*scipy.constants.m_e*scipy.constants.e*phir)
    k = g_vec/scipy.constants.hbar
    wavelength = (2*np.pi/k)*1e12

    pixel_size_inv_angstrom = (2*max_cam_angle_rads)*1e-3/(
            wavelength*0.01)/camera_frame_size  # assuming 512 pixels

    template_spot_rectangles = []

    template_fit = full_fitting[0]["objects"]
    spot_id_list = [*range(len(template_fit))]

    for i in tqdm(template_fit):
        rectangle = i["bounding_box"]
        template_spot_rectangles.append(rectangle)
        #spot_coords = template_fit[i]["center"]
        #template_spot_positions.append(spot_coords)

    all_spot_positions = []
    for frame in tqdm(range(len(full_fitting))):
        results = full_fitting[frame]["objects"]
        spot_positions_per_frame = []
        for spot in results:
            spot_positions = spot["center"]
            spot_positions_per_frame.append(spot_positions)
        all_spot_positions.append(spot_positions_per_frame)

    template_spot_positions = all_spot_positions[0]
    print(f"There are {len(template_spot_positions)} detected in the first frame")
    bulk_results = []

    for frame in tqdm(range(len(full_fitting))):
        #print("Frame",frame)
        #frame_list = []
        list_name = []
        for spot in range(len(all_spot_positions[frame])): #this is probably wrong
            for rectangle in range(len(template_spot_rectangles)):
                spot_center = all_spot_positions[frame][spot]
                is_in = pointInRect(spot_center,template_spot_rectangles[rectangle])
                if is_in ==True:
                    id_for_spot = spot_id_list[rectangle]
                    result = (spot_center,id_for_spot)
                    list_name.append(result)
                    #print(f"Comparing {id_for_spot} at {spot_center} to see if it is inside {template_spot_rectangles[rectangle]}, which is {is_in}")

        bulk_results.append(list_name)

    captured_spot_ids = []
    for frame in range(len(bulk_results)):
        for spot in bulk_results[frame]:
            spot_id = spot[1]
            captured_spot_ids.append(spot_id)



    lifespan_steps = Counter(captured_spot_ids)
    lifespan_doses = {}
    print(f"There are {len(lifespan_steps)} spots captured in the series, meaning {len(template_spot_positions)-len(lifespan_steps)} were lost")
    """Calculating lifespan in dose units"""
    for step in lifespan_steps.items():
        #print(step)
        lifespan_doses[step[0]] = doses[step[1]]

    #print(lifespan_doses)

    for spot in spot_id_list:
        if spot not in lifespan_steps.keys():
            print(f"Spot ID {spot} is missing")
            lifespan_doses[spot]=0 #assume intensity is zero

    #TODO find way to get zero order beam reliably

    """spot_intensities_list = []

    last_pattern = all_spot_positions[-1]

    for i in range(len(last_pattern)):
        spot_intensity = last_pattern[i]["max_intensity"]
        spot_intensities_list.append(spot_intensity)

    max_spot_intensity_pattern = spot_intensities_list.index(max(spot_intensities_list))

    brightest_spot_center = template_spot_positions[max_spot_intensity_pattern]
    brightest_spot_center = all_spot_positions[-1][-1]

    print("brightest spot position",brightest_spot_center)"""

    #brightest_spot_center = (0.54,0.5)

    most_frequent_spot_id = max(lifespan_steps,key=lifespan_steps.get)
    brightest_spot_center = template_spot_positions[most_frequent_spot_id]

    spot_resolutions_fractional = []
    spot_resolutions_mrad = []
    spot_resolutions_angstrom = []

    for spot in template_spot_positions:
        if spot is not brightest_spot_center:
            spot_resolution_fractional = math.dist(spot,brightest_spot_center)
            spot_resolution_mrad = spot_resolution_fractional*(max_cam_angle_rads*1e3)
            spot_resolution_angstrom = 1/(spot_resolution_fractional*camera_frame_size*pixel_size_inv_angstrom)
        else: #filters out a nasty divide by zero error from having zero resolution for central beam
            spot_resolution_fractional = 0
            spot_resolution_mrad = 0
            spot_resolution_angstrom = 0

        spot_resolutions_fractional.append(spot_resolution_fractional)
        spot_resolutions_mrad.append(spot_resolution_mrad)
        spot_resolutions_angstrom.append(spot_resolution_angstrom)


    df = pd.DataFrame({"spot ID":lifespan_doses.keys(),
                       "Lifespan dose":lifespan_doses.values(),
                       "spot position":template_spot_positions,
                       "spot resolution fractional":spot_resolutions_fractional,
                       "spot resolution mrad":spot_resolutions_mrad,
                       "spot resolution angstrom":spot_resolutions_angstrom})

    for i in df.iterrows():
        id = i[1][0]
        spot_pos = i[1][2]
        lifetime = i[1][1]
        plot = plt.scatter(x=spot_pos[0],y=spot_pos[1],c=lifetime,vmin=0,vmax=15)
    plt.scatter(brightest_spot_center[0],brightest_spot_center[1],marker="+",c="red")
    plt.imshow(dataset[0][0].astype(np.uint16),vmax=np.average(dataset[0][0]*5),extent=(0,1,1,0),cmap="gray")
    plt.colorbar(plot)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title("Template spot positions")
    plt.show()

    return bulk_results,lifespan_doses,df



#filepath = r"O:\TEM\Applications\Naren\Cryo-Test-P3\Dose series testing 12_7_24\Room temp 7"

#file = filepath+"\dose series.pdat"
#with open(file,"rb") as f:
#    dataset = pickle.load(f)
#print("Dataset loaded")


#for image in range(len(dataset[0])):
#    cv2.imwrite(filename=f"C:\\Users\\robert.hooley\\Desktop\\Coding\\Beam damage testing\\Cryo 5\\{image}.tiff", img=dataset[0][image])


#results = get_spot_intensities(dataset,spot_radius=3)
#plot_results(results)


#num_spots = count_spots(dataset,0)

#full_fitting = full_series_spot_tracking(dataset)

#with open(r"C:\Users\robert.hooley\Desktop\Coding\Beam damage testing\bulk output test\everything.pdat","rb")as f:
#    full_fitting = pickle.load(f)

#with open(r"C:\Users\robert.hooley\Desktop\Coding\Beam damage testing\Room temp 4\Spot cache\cache.pdat","rb")as f:
#    num_spots = pickle.load(f)

#garman_dose = calculate_garman_limit_spot_number(num_spots,dataset[1])
#is_it_in = calculate_spot_lifetime(full_fitting,dataset[1])
#print(is_it_in)

