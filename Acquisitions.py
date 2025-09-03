import matplotlib.pyplot as plt

from expert_pi.__main__ import window #TODO something is not right with this on some older versions of expi
from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT, CondenserFocusType as CFT,RoiMode as RM
from serving_manager.api import TorchserveRestManager
from expert_pi.app import app
from expert_pi.gui import main_window
from grid_strategy import strategies

from expert_pi.RSTEM.utilities import collect_metadata

window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client

#from stem_measurements import shift_measurements
from tqdm import tqdm
import numpy as np
from time import sleep
import easygui as g
import cv2 as cv2
import fnmatch
import os
import json

from serving_manager.api import registration_model
from serving_manager.api import super_resolution_model
from serving_manager.api import TorchserveRestManager


from expert_pi.RSTEM.utilities import get_microscope_parameters
#registration_model(np.concatenate([original_image,translated_image],axis=1, 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.png') #TIFF is also ok
#registration_model(image, 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.png') #TIFF is also ok
#super_resolution_model(image=image, model_name='SwinIRImageDenoiser', host='172.19.1.16', port='7447') #for denoising
#manager = TorchserveRestManager(inference_port='8080', management_port='8081', host='172.16.2.86', image_encoder='.tiff')
#manager.infer(image=image, model_name='spot_segmentation') #spot detection
#manager.list_models()

host_F2 = "172.23.158.142"
host_F4 = "192.168.51.3"
host_P3 = "172.20.32.1" #TODO confirm
host_P2 = "172.25.15.0"
host_global = '172.16.2.86'
host_local = "172.27.153.166"

host = host_local


def normalize_to_8bit(image):
    """
    Normalizes a numpy array to 8-bit for display purposes with percentile-based clipping.
    """
    lower_percentile, upper_percentile = np.percentile(image, [1, 99])  # Get 1st and 99th percentiles
    clipped_image = np.clip(image, lower_percentile, upper_percentile)  # Clip to the range
    image_min = clipped_image.min()
    image_max = clipped_image.max()
    normalized = (255 * (clipped_image - image_min) / (image_max - image_min)).astype(np.uint8)
    return normalized

#refactored to 0.1.0
def acquire_focal_series(extent_nm,steps=11,BF=True,ADF=False,num_pixels=1024,pixel_time=5e-6):
    """Parameters
    extent_nm: Range the focal series should cover split equally around the current focus value
    steps: how many total steps the focal series should cover
    BF: True or False
    ADF: True or False
    num_pixels: default 1024
    pixel_time_us: default 5us"""

    if steps % 2 == 0:
        steps=steps+1
        print("Adding a step so series passes at zero")
    else:
        pass

    current_defocus = grpc_client.illumination.get_condenser_defocus(CFT.C3)
    lower_defocus = current_defocus-(extent_nm*1e-9/2)
    higher_defocus = current_defocus+(extent_nm*1e-9/2)
    defocus_intervals = np.linspace(lower_defocus,higher_defocus,steps)

    image_series = []
    defocus_offsets = []

    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is False:
        grpc_client.projection.set_is_off_axis_stem_enabled(True)

    print("Acquiring defocus series")
    for i in tqdm(defocus_intervals,unit="Frame"):

        grpc_client.illumination.set_condenser_defocus(i,CFT.C3) #get correct command
        defocus_now = grpc_client.illumination.get_condenser_defocus(CFT.C3)

        defocus_offset = current_defocus-defocus_now #this will be in meters
        defocus_offset_nm = defocus_offset*1e9
        defocus_offsets.append(np.round(defocus_offset_nm,decimals=1))

        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=(DT.BF,DT.HAADF))
        header, data = cache_client.get_item(scan_id, num_pixels**2)  # retrive small measurements in one chunk

        if BF==True and ADF==False:
            BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
            image_series.append(BF_image)
        if ADF==True and BF == False:
            ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
            image_series.append(ADF_image)
        if BF == True and ADF == True:
            BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
            ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
            image_series.append((BF_image,ADF_image))

    grpc_client.illumination.set_condenser_defocus(current_defocus, CFT.C3)

    specs = strategies.SquareStrategy("center").get_grid(len(image_series))

    for i, subplot in enumerate(specs):
        if type(image_series[0]) is tuple:
            image=image_series[i][0]
        else:
            image = image_series[i] #should handle combined BF and DF series

        plt.subplot(subplot)
        ax = plt.gca()
        image = image_series[i]
        name = str(defocus_offsets[i])
        ax.imshow(image, cmap="gray")
        ax.set_title(name+" nm")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    plt.rcParams["figure.figsize"] = (12, 12)
    plt.show(block=False)
    return image_series,defocus_offsets

#refactored to 0.1.0
def acquire_FOV_series(scale,steps=11,num_pixels=1024,pixel_time=5e-6):
    """Parameters
    extent_nm: Range the focal series should cover split equally around the current focus value
    steps: how many total steps the focal series should cover
    BF: True or False
    ADF: True or False
    num_pixels: default 1024
    pixel_time_us: default 5us"""

    if steps % 2 == 0:
        steps=steps+1
        print("Adding a step so series passes at zero")
    else:
        pass


    current_fov = grpc_client.scanning.get_field_width()
    lower_fov = current_fov/scale
    higher_fov = current_fov*scale
    fovs_lower = np.linspace(lower_fov,current_fov,int((steps-1)/2))
    fovs_lower = fovs_lower[:-1]
    fovs_higher = np.linspace(current_fov,higher_fov, int((steps-1)/2))
    fovs = np.append(fovs_lower,fovs_higher)

    image_series = []

    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is False:
        grpc_client.projection.set_is_off_axis_stem_enabled(True)

    print("Acquiring defocus series")
    for i in tqdm(fovs,unit="Frame"):

        grpc_client.scanning.set_field_width(i) #get correct command

        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=(DT.BF,DT.HAADF))
        header, data = cache_client.get_item(scan_id, num_pixels**2)  # retrive small measurements in one chunk
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
        image_series.append((BF_image,ADF_image))

    grpc_client.scanning.set_field_width(current_fov)

    specs = strategies.SquareStrategy("center").get_grid(len(image_series))

    for i, subplot in enumerate(specs):
        plt.subplot(subplot)
        ax = plt.gca()
        image = np.concatenate((image_series[i][0],image_series[i][1]),axis=1)
        name = str(np.round((fovs[i]*1e6),3))
        ax.imshow(image, cmap="gray")
        ax.set_title(name+" um")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    plt.rcParams["figure.figsize"] = (12, 12)
    plt.show(block=False)
    return image_series,fovs

#refactored 0.1.0
def acquire_STEM(fov_um=None,pixel_time=5e-6,num_pixels=1024,scan_rotation_deg=None):
    """Acquires a single STEM image from the inserted detectors
    returns a tuple with the images and the metadata in a dictionary
    Parameters
    fov : field of view in microns
    pixel_time_us: dwell time in microseconds
    num_pixels: scan width in pixels,
    scan_rotation: scan rotation in degrees """

    if scan_rotation_deg is not None:
        grpc_client.scanning.set_rotation(np.deg2rad(scan_rotation_deg))
    if fov_um is not None:
        grpc_client.scanning.set_field_width(fov_um*1e-6) #in microns

    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is False:
        grpc_client.projection.set_is_off_axis_stem_enabled(True) #ensure off axis BF is used if both detectors are out

    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is True:
        grpc_client.projection.set_is_off_axis_stem_enabled(False) # ensure off axis is not used if both detectors are in
    if grpc_client.stem_detector.get_is_inserted(DT.BF) is True and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is True:
        grpc_client.projection.set_is_off_axis_stem_enabled(False)  #ensure off axis is not used if both detectors are in

    metadata = collect_metadata(acquisition_type="STEM",scan_width_px=num_pixels,pixel_time=pixel_time,scan_rotation=scan_rotation_deg)

    #scan_width_px = num_pixels,use_precession=False,camera_frequency_hz = None,
    #                                     STEM_dwell_time = pixel_time_us,scan_rotation=np.rad2deg(grpc_client.scanning.get_rotation()))

    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=[DT.BF,DT.HAADF])
    header, data = cache_client.get_item(scan_id, num_pixels ** 2) #retrive small measurements in one chunk

    BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
    ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
    return(BF_image,ADF_image,metadata)

#tested ok
def save_STEM(image,metadata=None,name=None,folder=None):
    """Parameters
    image: single array to be saved as a .tiff image
    metadata : optional dictionary to be saved as json
    name: optional user defined filename, otherwise will be called STEM
    folder: optional user defined folder, otherwise will show UI to select"""
    if folder == None:
        folder = g.diropenbox("Enter save location","Enter save location")
        folder + "\\"
    if name == None:
        num_files_in_dir = len(fnmatch.filter(os.listdir(folder), '*.tiff'))
        name = f"STEM000{num_files_in_dir+1}" #should increment the image number
    name = name+".tiff"
    filename = str(folder+"\\"+name)
    print(f"Saving {name} to {folder}")
    cv2.imwrite(filename,image)

    if metadata is not None:
        metadata_name = folder+"\\" + f"{name}_metadata.json"
        open_json = open(metadata_name, "w")
        json.dump(metadata, open_json, indent=6)
        open_json.close()

#tested ok 0.1.0
#TODO reintroduce scan rotation
def ML_drift_corrected_imaging(num_frames, pixel_time_us=None,num_pixels=None):
    """Parameters
    num_frames : integer number of frames to acquire
    pixel_time_us: pixel dwell time in microseconds
    num_pixels: number of pixels in scanned image
    Set the FOV and illumination conditions before calling the function, it will use whatever is set in the UI"""

    if pixel_time_us==None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6

    if num_pixels == None:
        num_pixels = 1024

    BF_images_list = []
    ADF_images_list = []
    image_offsets = []

    fov = grpc_client.scanning.get_field_width()

    initial_scan = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                               detectors=[DT.BF, DT.HAADF])
    header, data = cache_client.get_item(initial_scan, num_pixels ** 2) #retrive small measurements in one chunk

    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

    initial_BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
    BF_images_list.append(initial_BF_image)
    initial_ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
    ADF_images_list.append(initial_ADF_image)
    image_offsets.append(initial_shift)
    #TODO this needs a tracking signal image flag to handle HAADF images with BF retracted
    for frame in range(num_frames):
        print("Acquiring frame",frame,"of",num_frames)
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=[DT.BF, DT.HAADF])

        header, data = cache_client.get_item(scan_id, num_pixels ** 2) #retrive image from cache
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels) #reshape BF image
        BF_images_list.append(BF_image.astype(np.float64)) #add to list
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels) #reshape ADF image
        ADF_images_list.append(ADF_image.astype(np.float64)) #add to list

        s = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan) #get current beam shifts
        # apply drift correction between images:
        registration = registration_model(np.concatenate([BF_images_list[-1],BF_image],axis=1),
                                          'TEMRegistration', host=host, port='7443',
                                          image_encoder='.tiff') #measure offset of images
        raw_shift = registration[0]["translation"]
        real_shift_x = raw_shift[0]*fov #shifts normalised between 0,1 proportion of image, convert to meters
        real_shift_y = raw_shift[1]*fov
        grpc_client.illumination.set_shift(
                {"x": s['x'] - real_shift_x , "y": s['y'] - real_shift_y },grpc_client.illumination.DeflectorType.Scan) #apply shifts in microns to existing shifts#  TODO corrects to previous image, should it correct to previous image?

    print("Post acquisition fine correction")
    aligned_BF_series,summed_BF,_ = align_series_ML(BF_images_list)
    aligned_ADF_series,summed_ADF,_ = align_series_ML(ADF_images_list)

    return (aligned_BF_series,summed_BF), (aligned_ADF_series, summed_ADF)

def acquire_series(num_frames, pixel_time_us=None,num_pixels=None ):
    if pixel_time_us == None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6
    else:
        pixel_time = pixel_time_us/1e6

    if num_pixels == None:
        num_pixels = 1024

    BF_images_list = []
    ADF_images_list = []

    for frame in range(num_frames):
        print("Acquiring frame", frame, "of", num_frames)
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=[DT.BF, DT.HAADF])

        header, data = cache_client.get_item(scan_id, num_pixels**2)  # retrive image from cache
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)  # reshape BF image
        BF_images_list.append(BF_image.astype(np.float64))  # add to list #TODO why float 64?
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)  # reshape ADF image
        ADF_images_list.append(ADF_image.astype(np.float64))  # add to list #TODO why float 64?

    return (BF_images_list,ADF_images_list)


# tested ok without scan rotation
#TODO untested
def ML_drift_corrected_imaging_with_scan_rotation(num_frames, pixel_time=None, num_pixels=None, scan_rotation=0):
    R = np.array([[np.cos(scan_rotation), np.sin(scan_rotation)],
                  [-np.sin(scan_rotation), np.cos(scan_rotation)]])

    if scan_rotation is not None:
        grpc_client.scanning.set_rotation(np.deg2rad(scan_rotation))

    if pixel_time == None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds

    if num_pixels == None:
        num_pixels = 1024

    BF_images_list = []
    ADF_images_list = []
    image_offsets = []

    fov = grpc_client.scanning.get_field_width()
    initial_scan = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                    detectors=[DT.BF, DT.HAADF])
    header, data = cache_client.get_item(initial_scan, num_pixels**2)  # retrive small measurements in one chunk
    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)
    initial_BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
    BF_images_list.append(initial_BF_image)
    initial_ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
    ADF_images_list.append(initial_ADF_image)
    image_offsets.append(initial_shift)

    for frame in range(num_frames):
        print("Acquiring frame", frame, "of", num_frames)
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=[DT.BF, DT.HAADF])

        header, data = cache_client.get_item(scan_id, num_pixels**2)  # retrive image from cache
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)  # reshape BF image
        BF_images_list.append(BF_image.astype(np.float64))  # add to list
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)  # reshape ADF image
        ADF_images_list.append(ADF_image.astype(np.float64))  # add to list

        s = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)  # get current beam shifts
        # apply drift correction between images:
        registration = registration_model(np.concatenate([BF_images_list[-1], BF_image], axis=1),
                                          'TEMRegistration', host=host, port='7443',
                                          image_encoder='.tiff')  # measure offset of images
        raw_shift = registration[0]["translation"]
        real_shift_x = raw_shift[0]*fov  # shifts normalised between 0,1 proportion of image, convert to meters
        real_shift_y = raw_shift[1]*fov

        shift = np.dot(R, (real_shift_x,real_shift_y))  # rotate shifts back to scanning axes
        # print(i, (s0['x'] - s['x']) * 1e9, (s0['y'] - s['y']) * 1e9)
        grpc_client.illumination.set_shift(
                {"x": s['x'] + shift[0], "y": s['y'] + shift[1]},
                grpc_client.illumination.DeflectorType.Scan)  # apply shifts in microns to existing shifts

    print("Post acquisition correction")
    aligned_BF_series, summed_BF, _ = align_series_ML(BF_images_list)
    aligned_ADF_series, summed_ADF, _ = align_series_ML(ADF_images_list)

    return (aligned_BF_series, summed_BF), (aligned_ADF_series, summed_ADF)

def align_series_ML(image_series): #single series in a list

    initial_image = image_series[0]
    initial_image_shape = initial_image.shape
    shifts = []
    translated_list = []
    for image in tqdm(range(len(image_series))):
        translated_image = image_series[image].astype(np.float64)
        registration_values = registration_model(np.concatenate([initial_image,translated_image],axis=1), 'TEMRegistration', host=host, port='7443', image_encoder='.tiff')
        translation_values = registration_values[0]["translation"] #normalised between 0,1
        x_pixels_shift = translation_values[0]*initial_image_shape[0]
        y_pixels_shift = translation_values[1]*initial_image_shape[1]
        shifts.append((x_pixels_shift,y_pixels_shift))

        matrix = np.float32([[1,0,x_pixels_shift],[0,1,y_pixels_shift]])
        transposed_image = cv2.warpAffine(translated_image.astype(np.uint16),matrix,(initial_image_shape[1],initial_image_shape[0]))
        translated_list.append(transposed_image)

    summing_array = np.asarray(translated_list)
    summed_image = np.sum(summing_array, 0, dtype=np.float64)
    return translated_list,summed_image,shifts

"""def align_series_cross_correlation(image_series):
    initial_image = image_series[0]
    initial_image_shape = initial_image.shape
    fov_px = initial_image_shape[0]
    offsets = []
    correlation_coefficients = []
    translated_list = []
    for image in tqdm(range(len(image_series))):
        offset,coeffs = get_offset_of_pictures(initial_image,image_series[image],fov=fov_px,get_corr_coeff=True) #TODO remove STEM measurements function and use OpenCV instead
        offsets.append(offset)
        correlation_coefficients.append(coeffs)
        x_pixels_shift = offset[0]
        y_pixels_shift = offset[1]
        shift_matrix = np.float32([[1,0,x_pixels_shift],[0,1,y_pixels_shift]])
        translated_image = cv2.warpAffine(image_series[image].astype(np.uint16),shift_matrix,(initial_image_shape[1],initial_image_shape[0]))
        translated_list.append(translated_image)

    return translated_list,offsets,correlation_coefficients
"""

def get_number_of_nav_pixels():
    scan_field_pixels = window.scanning.size_combo.currentText()
    pixels = int(scan_field_pixels.replace(" px", ""))
    return pixels

def get_pixel_positions_pointer(): #this doesnt work when imported as a function, but why
    fov = window.scanning.fov_spin.value() #in microns
    print("FOV",fov)
    pixels = get_number_of_nav_pixels()
    print("Pixels",pixels)
    posx = window.image_view.tools["point_selector"].x()
    posy = window.image_view.tools["point_selector"].y()
    print("posX",posx,"posX",posy)
    pixel_coords = [int(posy / fov * pixels + 0.5) + pixels // 2, int(posx / fov * pixels + 0.5) + pixels // 2]
    print("pixel coords",pixel_coords)
    return pixel_coords

def start_point_acquisition(pixel_offsets,dwell_time=None):

    N=1
    overview = get_number_of_nav_pixels()

    #put in an is pixel selector active
    if dwell_time is None:
        dwell_time = 10e-3

    if pixel_offsets == None: #use center pixel
        rectangle = [overview/2,overview/2,N,N]
    else:
        rectangle = [pixel_offsets[0],pixel_offsets[1],N,N] #[11, 512, N, N]
        #rectangle = [p1_offset,p2_offset,N,N]
    grpc_client.projection.set_is_off_axis_stem_enabled(False)
    scan_id = scan_helper.start_rectangle_scan(pixel_time=dwell_time, total_size=overview, frames=1, detectors=[DT.Camera], rectangle=rectangle)
    header, data = cache_client.get_item(scan_id, N**2)
    camera_data = data['cameraData']
    image = np.reshape(camera_data,(512,512))

    plt.imshow(image)
    plt.show(block=False)

    return image


def get_spot_positions_ML(image,threshold=0):
    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                    image_encoder='.tiff')
    results = manager.infer(image=image, model_name='diffraction_spot_segmentation')  # spot detection
    spots = results["objects"]
    spot_list = []
    areas = []
    spot_radii = []
    fix,ax = plt.subplots(1,1)
    shape = image.shape
    for i in range(len(spots)):
        if spots[i]["mean_intensity"] > threshold:
            spot_coords = spots[i]["center"]
            spot_list.append(spot_coords)
            area = spots[i]["area"]
            areas.append(area)
            radius = np.sqrt(area/np.pi)/shape[0]
            spot_radii.append(radius)

    for i in spot_list:
        ax.plot(i[0],i[1],"r+")
        plt.imshow(image,vmax=np.average(image*10),extent=(0,1,1,0),cmap="gray")


    plt.show(block=False)
    output= (spot_list,spot_radii)

    return output


def pointer_spot_fit(pixel_coords=None):
    if pixel_coords is None:
        pixel_coords = get_pixel_positions_pointer()

    diff_pattern = start_point_acquisition(pixel_coords)

    results = get_spot_positions_ML(diff_pattern,threshold=0)

    print(len(results[0]),"spots were detected")
    return results