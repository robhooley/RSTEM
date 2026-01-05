from tqdm import tqdm
import numpy as np
from time import sleep
import easygui as g
import cv2 as cv2
import fnmatch
import os
import json
import matplotlib.pyplot as plt
from grid_strategy import strategies

from serving_manager.tem_models.specific_model_functions import registration_model
from serving_manager.management.torchserve_rest_manager import TorchserveRestManager

from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT

from RSTEM.app_context import get_app
from RSTEM.analysis import get_spot_positions
from RSTEM.utilities import collect_metadata

from RSTEM.utilities import crop_center_square


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


def scan_4D_basic(scan_width_px=128, camera_frequency_hz=4500, use_precession=False):
    """
    Parameters
    ----------
    scan_width_px : int
        Scan width/height in pixels (square raster).
    camera_frequency_hz : int or float
        Camera speed in frames per second (up to 72000).
    use_precession : bool
        Enable/disable precession.

    Returns
    -------
    (image_array, metadata) : tuple
        image_array has shape (scan_width_px, scan_width_px, camY, camX)
    """

    """from expertpi.application import app_state
    #unlock the state if acquisition fails
    app_state.AcqLock.unlock()"""

    app = get_app()

    pixel_time = 1.0 / camera_frequency_hz

    metadata = collect_metadata(
        acquisition_type="4D",
        scan_width_px=scan_width_px,
        use_precession=use_precession,
        pixel_time=pixel_time,
        edx_enabled=False)
    # Ensure STEM detectors are retracted and beam is on axis
    bf_in = app.api.stem_detector.get_is_inserted(DT.BF)
    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)
    if bf_in or haadf_in:
        if bf_in:
            app.detectors.stem.insert_bf(False)
        if haadf_in:
            app.detectors.stem.insert_df(False)
        for _ in tqdm(range(5), desc="Stabilising after STEM detector retraction", unit=""):
            sleep(1)
    app.scanning.set_off_axis(False)
    sleep(0.2)  # stabilisation after deflector change

    reader = app.acquisition.acquire_camera(pixel_time=np.round(pixel_time,8),total_size=scan_width_px,precession_enabled=use_precession)

    print(f"Acquiring {scan_width_px} x {scan_width_px} px dataset at {camera_frequency_hz} fps")
    #Retrieve first row to infer dtype/shape, then pre-allocate array
    first_row = reader.get_lines(1)
    row_block = first_row.camera # shape: (scan_width_px, camY, camX)
    if row_block.ndim != 4 or row_block.shape[1] != scan_width_px:
        raise RuntimeError(f"Unexpected cameraData shape: {row_block.shape}")
    camY, camX = row_block.shape[2], row_block.shape[3] #read camera size from first row of data
    dtype = row_block.dtype  # keep native dtype to avoid copies/conversions
    image_array = np.empty((scan_width_px, scan_width_px, camY, camX), dtype=dtype, order="C") #Pre-allocate target 4D array: (scanY, scanX, camY, camX)
    #Assign the first row (index 0) directly
    image_array[0, :, :, :] = row_block  # vectorized write
    #Retrieve remaining rows
    for i in tqdm(range(1, scan_width_px), desc="Retrieving data from cache", total=scan_width_px - 1, unit="rows"):
        data = reader.get_lines(1)
        row_block = data.camera  # expected shape: (scan_width_px, camY, camX)
        image_array[i, :, :, :] = row_block #assert row_block.shape == (scan_width_px, camY, camX)
    reader.close()
    print("Array ready")
    return image_array, metadata


def scan_4D(scan_width_px=128,dwell_time=5.556e-5,use_precession=False,roi_mode=128,post_crop=True,reset_after=True):#TODO refactor to pre-allocation
    """Parameters
    scan width: pixels
    camera_frequency: camera speed in frames per second up to 72000
    use_precession: True or False
    roi_mode: optional variable to enable ROI mode, either 128,256 or False
    returns a tuple of (image_array, metadata)
    """

    #sufficient_RAM = check_memory(1/dwell_time,scan_width_px,roi_mode)
    #if sufficient_RAM == False:
    #    print("This dataset will probably not fit into RAM, trying anyway but expect a crash")
     #gets the microscope and acquisition metadata
    app = get_app()


    # Ensure STEM detectors are retracted and beam is on axis
    bf_in = app.api.stem_detector.get_is_inserted(DT.BF)
    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)
    if bf_in or haadf_in:
        if bf_in:
            app.detectors.stem.insert_bf(False)
        if haadf_in:
            app.detectors.stem.insert_df(False)
        for _ in tqdm(range(5), desc="Stabilising after STEM detector retraction", unit=""):
            sleep(1)
    app.scanning.set_off_axis(False)
    sleep(0.2)  # stabilisation after deflector change


    #metadata = collect_metadata(acquisition_type="Camera",scan_width_px = scan_width_px, use_precession= use_precession, pixel_time = dwell_time)

    #sets to ROI mode
    if roi_mode==128: #512x128 px
        #app.detectors.camera.set_roi(RM.Lines_128)
        app.api.scanning.set_camera_roi(roi_mode=RM.Lines_128, use16bit=False)
        camera_shape=(128,512)
    elif roi_mode==256: #512x256 px
        #app.detectors.camera.set_roi(RM.Lines_256)
        app.api.scanning.set_camera_roi(roi_mode=RM.Lines_256, use16bit=False)
        camera_shape=(256,512)
    else:
        #app.detectors.camera.set_roi(RM.Disabled)
        app.api.scanning.set_camera_roi(roi_mode=RM.Disabled, use16bit=False)

    reader = app.acquisition.acquire_camera(pixel_time=dwell_time, total_size=scan_width_px,
                                            precession_enabled=use_precession)

    print(f"Acquiring {scan_width_px} x {scan_width_px} px dataset at {int(1/dwell_time)} fps")
    # Retrieve first row to infer dtype/shape, then pre-allocate array
    first_row = reader.get_lines(1)
    row_block = first_row.camera  # shape: (scan_width_px, camY, camX)
    if row_block.ndim != 4 or row_block.shape[1] != scan_width_px:
        raise RuntimeError(f"Unexpected cameraData shape: {row_block.shape}")

    camY, camX = row_block.shape[2], row_block.shape[3]  # read camera size from first row of data
    dtype = row_block.dtype  # keep native dtype to avoid copies/conversions
    image_array = np.empty((scan_width_px, scan_width_px, camY, camX), dtype=dtype,
                           order="C")  # Pre-allocate target 4D array: (scanY, scanX, camY, camX)
    # Assign the first row (index 0) directly
    image_array[0, :, :, :] = row_block  # vectorized write
    # Retrieve remaining rows
    for i in tqdm(range(1, scan_width_px), desc="Retrieving data from cache", total=scan_width_px - 1, unit="rows"):
        data = reader.get_lines(1)
        row_block = data.camera  # expected shape: (scan_width_px, camY, camX)
        image_array[i, :, :, :] = row_block  # assert row_block.shape == (scan_width_px, camY, camX)

    if post_crop:
        image_array = crop_center_square(image_array, roi_mode)

    if reset_after:
        #app.detectors.camera.set_roi(RM.Disabled)
        app.api.scanning.set_camera_roi(roi_mode=RM.Disabled, use16bit=False)

    #metadata = collect_metadata(
    #    acquisition_type="4D",
    #    scan_width_px=scan_width_px,
    #    use_precession=use_precession,
    #    pixel_time=dwell_time,
    #    edx_enabled=False)

    return (image_array)#,metadata) #tuple with image data and metadata

#refactored to 0.5.1
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

    app=get_app()

    current_defocus = app.api.illumination.get_condenser_defocus(CFT.C3)
    lower_defocus = current_defocus-(extent_nm*1e-9/2)
    higher_defocus = current_defocus+(extent_nm*1e-9/2)
    defocus_intervals = np.linspace(lower_defocus,higher_defocus,steps)

    image_series = []
    defocus_offsets = []

    if app.api.stem_detector.get_is_inserted(DT.BF) is False and app.api.stem_detector.get_is_inserted(DT.HAADF) is False:
        app.scanning.set_off_axis(True)

    print("Acquiring defocus series")
    for i in tqdm(defocus_intervals,unit="Frame"):

        app.api.illumination.set_condenser_defocus(i,CFT.C3) #get correct command
        defocus_now = app.api.illumination.get_condenser_defocus(CFT.C3)

        defocus_offset = current_defocus-defocus_now #this will be in meters
        defocus_offset_nm = defocus_offset*1e9
        defocus_offsets.append(np.round(defocus_offset_nm,decimals=1))

        scan = app.acquisition.acquire_stem(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=(DT.BF,DT.HAADF))
        image = scan.get_all()

        if BF==True and ADF==False:
            BF_image = image["BF"][0]
            image_series.append(BF_image)
        if ADF==True and BF == False:
            ADF_image = image["HAADF"][0]
            image_series.append(ADF_image)
        if BF == True and ADF == True:
            BF_image = image["BF"][0]
            ADF_image = image["HAADF"][0]
            image_series.append((BF_image,ADF_image))

    app.api.illumination.set_condenser_defocus(current_defocus, CFT.C3)

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

#TODO REFACTORED UNTESTED
def acquire_FOV_series(upper_fov,lower_fov,steps=10,num_pixels=None,pixel_time=None):
    """Parameters
    upper_fov: upper end of the fov series in microns #TODO needs a max fov validation adding
    lower_fov: lower end of the fov series in microns#TODO needs a min fov validation adding
    steps: how many total steps the FOV series should cover
    BF: True or False
    ADF: True or False
    num_pixels: default 1024
    pixel_time_us: default takes from UI"""

    app=get_app()

    current_fov = app.scanning.get_fov()
    fovs_lower = np.linspace(lower_fov,current_fov,int((steps-1)/2))
    fovs_lower = fovs_lower[:-1]
    fovs_higher = np.linspace(current_fov,upper_fov, int((steps-1)/2))
    fovs = np.append(fovs_lower,fovs_higher)

    image_series = []

    if pixel_time is None:
        app.scanning.get_pixel_time()
    if num_pixels is None:
        num_pixels = app.scanning.get_pixel_count().value

    if app.api.stem_detector.get_is_inserted(DT.BF) is False and app.api.stem_detector.get_is_inserted(DT.HAADF) is False:
        app.scanning.set_off_axis(True)

    print("Acquiring defocus series")
    for i in tqdm(fovs,unit="Frame"):

        app.scanning.set_fov(i) #Set FOV
        scan = app.acquisition.acquire_stem(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                            detectors=(DT.BF, DT.HAADF))
        image = scan.get_all()
        BF_image = image["BF"][0]
        ADF_image = image["HAADF"][0]
        image_series.append((BF_image,ADF_image))

    app.scanning.set_fov(current_fov)

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

#TODO REFACTORED UNTESTED
def acquire_STEM(fov=None,pixel_time=None,num_pixels=None,scan_rotation_deg=None):
    """Acquires a single STEM image from the inserted detectors
    returns a tuple with the images and the metadata in a dictionary
    Parameters
    fov : field of view in microns
    pixel_time_us: dwell time in microseconds
    num_pixels: scan width in pixels,
    scan_rotation: scan rotation in degrees """

    app = get_app()

    if scan_rotation_deg is not None:
        app.scanning.set_scanning_rotation(np.deg2rad(scan_rotation_deg))
    if fov is not None:
        app.scanning.set_fov(fov) #in meters

    if app.api.stem_detector.get_is_inserted(DT.BF) is False and app.api.stem_detector.get_is_inserted( #API call, not synced
            DT.HAADF) is False:
        app.scanning.set_off_axis(True) #ensure off axis BF is used if both detectors are out

    if app.api.stem_detector.get_is_inserted(DT.BF) is False and app.api.stem_detector.get_is_inserted( #API call, not synced
            DT.HAADF) is True:
        app.scanning.set_off_axis(False) # ensure off axis is not used if both detectors are in
    if app.api.stem_detector.get_is_inserted(DT.BF) is True and app.api.stem_detector.get_is_inserted( #API call, not synced
            DT.HAADF) is True:
        app.scanning.set_off_axis(False)  #ensure off axis is not used if both detectors are in

    if pixel_time is None:
        pixel_time = app.scanning.get_pixel_time()
    if num_pixels is None:
        num_pixels = app.scanning.get_pixel_count().value

    scan = app.acquisition.acquire_stem(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                        detectors=(DT.BF, DT.HAADF))
    image = scan.get_all()
    BF_image = image["BF"][0]
    ADF_image = image["HAADF"][0]

    #metadata = collect_metadata(acquisition_type="STEM",scan_width_px=num_pixels,pixel_time=pixel_time,scan_rotation=scan_rotation_deg) #TODO could get all these from the scan controller

    return(BF_image,ADF_image)#,metadata)

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


#TODO REFACTORED
def acquire_series(num_frames=10,pixel_time=None,num_pixels=None):

    app = get_app()

    BF_images_list = []
    ADF_images_list = []

    if pixel_time==None:
        pixel_time = app.scanning.get_pixel_time()
    if num_pixels == None:
        num_pixels = app.scaning.get_pixel_count().value


    scan = app.acquisition.acquire_stem(pixel_time=pixel_time, total_size=num_pixels, frames=num_frames,
                                            detectors=(DT.BF, DT.HAADF))
    for frame in range(num_frames):
        images = scan.get_frame(frame)
        BF_images_list.append(images["BF"])
        ADF_images_list.append(images["HAADF"])

    return (BF_images_list,ADF_images_list)


#TODO REFACTORED why not working?
def point_acquisition(pixel_offsets=None,dwell_time=None,fit_spots=False):


    app = get_app()
    N=1

    overview = app.scanning.get_pixel_count().value

    #put in an is pixel selector active
    if dwell_time is None:
        dwell_time = 10e-3

    if pixel_offsets == None: #use center pixel
        rectangle = [overview/2,overview/2,N,N]
    else:
        rectangle = [pixel_offsets[0],pixel_offsets[1],N,N]
    app.scanning.set_off_axis(False)
    pointer = app.acquisition.acquire_camera(pixel_time=dwell_time, total_size=overview, frames=1, rectangle=rectangle,precession_enabled=None)
    image = pointer.get_frame()
    camera_data = image.camera[0][0]

    if fit_spots==True:
        results = get_spot_positions(camera_data, threshold=0)

        print(len(results[0]), "Spots were detected")

        return image,results

    else:
        plt.imshow(camera_data)
        plt.show(block=False)
        return image

#TODO refactored but untested
def acquire_precession_tilt_series(upper_limit_degrees):

    """Acquires a series of precession diffraction patterns from 0 to the max angle in 0.1 degree steps"""

    app = get_app()
    filepath = g.diropenbox("Select directory to save series","Save location")
    beam_size = app.optics.get_diameter()

    app.scanning.set_fov(beam_size*2)
    pattern_list = []

    angle_list = list(np.linspace(0,upper_limit_degrees,0.1))

    bf_in = app.api.stem_detector.get_is_inserted(DT.BF)
    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)
    app.scanning.set_off_axis(False)
    if bf_in or haadf_in:
        if bf_in:
            app.detectors.stem.insert_bf(False)
        if haadf_in:
            app.detectors.stem.insert_df(False)

    scan_width_px = 8

    for i in angle_list:
        print(f"Current precession angle {i} degrees")
        radians = np.deg2rad(i)
        app.api.scanning.set_precession_angle(radians)
        reader = app.acquisition.acquire_camera(pixel_time=1e-3,total_size=scan_width_px,precession_enabled=True)

        # Retrieve first row to infer dtype/shape, then pre-allocate array
        first_row = reader.get_lines(1)
        row_block = first_row.camera  # shape: (scan_width_px, camY, camX)
        if row_block.ndim != 4 or row_block.shape[1] != scan_width_px:
            raise RuntimeError(f"Unexpected cameraData shape: {row_block.shape}")
        camY, camX = row_block.shape[2], row_block.shape[3]  # read camera size from first row of data
        dtype = row_block.dtype  # keep native dtype to avoid copies/conversions
        image_array = np.empty((scan_width_px, scan_width_px, camY, camX), dtype=dtype,
                               order="C")  # Pre-allocate target 4D array: (scanY, scanX, camY, camX)
        # Assign the first row (index 0) directly
        image_array[0, :, :, :] = row_block  # vectorized write
        # Retrieve remaining rows
        for i in range(1, scan_width_px):
            data = reader.get_lines(1)
            row_block = data.camera  # expected shape: (scan_width_px, camY, camX)
            image_array[i, :, :, :] = row_block  # assert row_block.shape == (scan_width_px, camY, camX)
        reader.close()

        single_pattern = np.sum(image_array,axis=(0,1))
        pattern_list.append(single_pattern)

    annotated_images = []
    for i in range(len(pattern_list)):
        image = pattern_list[i].astype(np.uint8)
        angle = angle_list[i]
        text = f"Precession angle {angle} degrees"
        cv2.putText(image, text, (10,10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(255, 225, 255))
        filename = filepath+f"\precession angle {angle} degrees.tiff"
        cv2.imwrite(filename,image)
        annotated_images.append(image)

    return annotated_images,angle_list
