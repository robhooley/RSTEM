from tqdm import tqdm
import numpy as np
import easygui as g
import cv2 as cv2
import fnmatch
import os
import json
import matplotlib.pyplot as plt
from grid_strategy import strategies
from time import time,sleep
import math

from serving_manager.tem_models.specific_model_functions import registration_model
from serving_manager.management.torchserve_rest_manager import TorchserveRestManager

from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT, DeflectorType as DFT

from expertpi.config import Config



#config = Config()
config = Config(r"C:\Users\stem\Documents\Rob_coding\ExpertPI-0.5.1\config.yml") # path to config file if changes have been made, otherwise comment out and use default

from importlib.util import find_spec

if find_spec("RSTEM.app_context") is not None:
    from RSTEM.app_context import get_app
    from RSTEM.utilities import (
        model_has_workers,
        downsample_diffraction,
        create_circular_mask,
        collect_metadata,
        crop_center_square,
        validate_quantity
    )
    from RSTEM.analysis import (
    align_image_series,
    get_spot_positions
    )
else:
    from app_context import get_app
    from utilities import (
        model_has_workers,
        downsample_diffraction,
        create_circular_mask,
        collect_metadata,
        crop_center_square,
        validate_quantity
    )
    from analysis import (
    align_image_series,
    get_spot_positions
    )


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

    validated_camera_frequency_hz = validate_quantity(camera_frequency_hz)
    camera_frequency_hz = validated_camera_frequency_hz[0]
    if validated_camera_frequency_hz[1] is not None:
        print(validated_camera_frequency_hz[1])


    pixel_time = 1.0 / validated_camera_frequency_hz[0]

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


def scan_4D(scan_width_px=128,dwell_time=5.556e-5,use_precession=False,roi_mode=128,post_crop=True,reset_after=True):
    """Parameters
    scan width: pixels
    camera_frequency: camera speed in frames per second up to 72000
    use_precession: True or False
    roi_mode: optional variable to enable ROI mode, either 128,256 or False
    returns a tuple of (image_array, metadata)
    """

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

    #sets to ROI mode
    if roi_mode==128: #512x128 px
        app.detectors.camera.set_roi(RM.Lines_128)
    elif roi_mode==256: #512x256 px
        app.detectors.camera.set_roi(RM.Lines_256)
    else:
        app.detectors.camera.set_roi(RM.Disabled)

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
    for i in tqdm(range(1, scan_width_px), desc="Retrieving remaining data from cache", total=scan_width_px - 1, unit="rows"):
        data = reader.get_lines(1)
        row_block = data.camera  # expected shape: (scan_width_px, camY, camX)
        image_array[i, :, :, :] = row_block  # assert row_block.shape == (scan_width_px, camY, camX)

    if post_crop:
        image_array = crop_center_square(image_array, roi_mode)

    if reset_after:
        app.detectors.camera.set_roi(RM.Disabled)

    metadata = collect_metadata(
       acquisition_type="4D",
        scan_width_px=scan_width_px,
        use_precession=use_precession,
        pixel_time=dwell_time)

    return (image_array,metadata) #tuple with image data and metadata

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
def acquire_STEM(fov=None, pixel_time=None, num_pixels=None, scan_rotation_deg=None):
    """Acquires a single STEM image from the inserted detectors
    returns (images_list, metadata_dict)
    """
    app = get_app()

    if scan_rotation_deg is not None:
        app.scanning.set_scanning_rotation(np.deg2rad(scan_rotation_deg))
    if fov is not None:
        app.scanning.set_fov(fov)

    bf_in = app.api.stem_detector.get_is_inserted(DT.BF)
    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)

    if (not bf_in) and (not haadf_in):
        app.scanning.set_off_axis(True)      # off-axis BF
        detectors = (DT.BF,)
    elif (not bf_in) and haadf_in:
        app.scanning.set_off_axis(False)
        detectors = (DT.HAADF,)
    elif bf_in and haadf_in:
        app.scanning.set_off_axis(False)
        detectors = (DT.BF, DT.HAADF)
    else:  # bf_in and (not haadf_in)
        app.scanning.set_off_axis(False)
        detectors = (DT.BF,)

    if pixel_time is None:
        pixel_time = app.scanning.get_pixel_time()
    if num_pixels is None:
        num_pixels = app.scanning.get_pixel_count().value

    scan = app.acquisition.acquire_stem(
        pixel_time=pixel_time,
        total_size=num_pixels,
        frames=1,
        detectors=list(detectors),  # pass a list of detector enums
    )
    image = scan.get_all()

    output = []
    if DT.BF in detectors:
        output.append(image["BF"][0])
    if DT.HAADF in detectors:
        output.append(image["HAADF"][0])

    metadata = collect_metadata(
        acquisition_type="STEM",
        scan_width_px=num_pixels,
        pixel_time=pixel_time
    )

    return output, metadata


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

def rotational_correction(raw_shift, fov_x, fov_y, theta_deg, y_down=True):
    """
    Map registration shift (normalized image coords) to deflector delta (physical).
    raw_shift: (dx_norm, dy_norm) where +x=right, +y=down in image.
    Returns (Δdef_x, Δdef_y) to ADD to current deflector shift (already negated to oppose drift).
    """
    dx_m = float(raw_shift[0]) * fov_x
    dy_m = float(raw_shift[1]) * fov_y
    if y_down:
        dy_m = -dy_m  # array +y down → physical +y up

    v_img = np.array([dx_m, dy_m], dtype=float)

    th = np.deg2rad(theta_deg)
    # Rotate image-vector by -θ into scan/deflector axes
    Rm = np.array([[ np.cos(-th), -np.sin(-th)],
                   [ np.sin(-th),  np.cos(-th)]], dtype=float)
    v_scan = Rm @ v_img

    # Oppose the measured drift: return the delta to ADD to current deflectors
    delta_deflector = -v_scan
    return float(delta_deflector[0]), float(delta_deflector[1])

def drift_corrected_imaging(num_frames=10, pixel_time=None, num_pixels=None, host=None,model_name="TEMRegistration",logging=False):  # TODO full refactor needed
    """Parameters
    num_frames : integer number of frames to acquire (total, including the seed frame)
    pixel_time_us: pixel dwell time in microseconds; if None, read from UI
    num_pixels: number of pixels in scanned image; default 1024

    Set the FOV and illumination conditions before calling; it uses current UI state.
    """
    app = get_app()

    images_list   = []   # list of dict frames: {"BF": np.ndarray|None, "HAADF": np.ndarray|None}
    image_offsets = []   # list of deflector shift dicts (logged before applying correction)

    try:
        fov = float(app.scanning.get_fov())
    except Exception as e:
        raise RuntimeError(f"Failed to read field width (FOV): {e}")
    if not np.isfinite(fov) or fov <= 0:
        raise RuntimeError(f"Invalid field width (FOV): {fov!r}")

    fov_x = fov_y = fov

    # --- Scan rotation (ok to default to 0° if unavailable) -----------------
    try:
        theta_val = np.rad2deg(app.scanning.get_scanning_rotation())
        theta_deg = float(theta_val) if theta_val is not None else 0.0
    except Exception:
        raise RuntimeError("Cannot read Scan Rotation from HW")

    # Decide tracking signal
    bf_in    = app.api.stem_detector.get_is_inserted(DT.BF)
    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)

    if bf_in and haadf_in:
        tracking_signal = DT.HAADF
        app.scanning.set_off_axis(False)
    elif haadf_in:
        tracking_signal = DT.HAADF
        app.scanning.set_off_axis(False)
    elif bf_in:
        tracking_signal = DT.BF
        app.scanning.set_off_axis(False)
    else:  # if nothing is inserted use off-axis BF
        tracking_signal = DT.BF
        app.scanning.set_off_axis(True)

    track_key = "BF" if tracking_signal == DT.BF else "HAADF"  # expected keys in stemData

    # Helpers
    def _detectors_for_scan(include_both_if_available=True):
        dets = []
        if bf_in: dets.append(DT.BF)
        if haadf_in: dets.append(DT.HAADF)
        if not dets:  # neither inserted → BF off-axis
            return [DT.BF]
        return dets if include_both_if_available else [tracking_signal]

    def _acquire_frame(det_list, pixel_time, num_pixels):
        # honor the detector list passed in
        scan = app.acquisition.acquire_stem(
            pixel_time=pixel_time,
            total_size=num_pixels,
            frames=1,
            detectors=tuple(det_list)
        )
        image = scan.get_all()
        BF_image  = image["BF"][0]    if "BF" in image    and len(image["BF"])    else None
        ADF_image = image["HAADF"][0] if "HAADF" in image and len(image["HAADF"]) else None
        return {"BF": BF_image, "HAADF": ADF_image}

    if host is None:
        host = config.inference.host

    if pixel_time is None:
        pixel_time = app.scanning.get_pixel_time()
    if num_pixels is None:
        num_pixels = app.scanning.get_pixel_count().value

    manager = TorchserveRestManager(
        inference_port='8080',
        management_port='8081',
        host=host,
        image_encoder='.tiff'
    )
    manager.scale(model_name=model_name)

    # --- tracking frame -----------------------------------------
    initial_shift = app.adjustments.get_illumination_shift()
    seed_frame = _acquire_frame(_detectors_for_scan(include_both_if_available=True), pixel_time, num_pixels)
    images_list.append(seed_frame)
    image_offsets.append(initial_shift)

    # --- Subsequent frames (register to previous) ---------------------------
    for frame_idx in range(1, num_frames):
        print(f"Acquiring frame {frame_idx} of {num_frames - 1}")

        curr_frame = _acquire_frame(_detectors_for_scan(include_both_if_available=True), pixel_time, num_pixels)

        prev_tracking = images_list[-1][track_key]
        curr_tracking = curr_frame[track_key]
        if prev_tracking is None or curr_tracking is None:
            # Skip correction if we lack the tracking channel
            images_list.append(curr_frame)
            image_offsets.append(app.adjustments.get_illumination_shift())
            continue

        # Registration input (current vs previous). Model predicts motion curr -> prev in image coords.
        reg_input = np.concatenate(
            [curr_tracking.astype(np.float32, copy=False),
             prev_tracking.astype(np.float32, copy=False)],
            axis=1
        )
        pre_inference = time()
        registration = registration_model(
            reg_input,
            model_name,
            host=host, port="8080",
            image_encoder=".tiff"
        )
        if logging:
            post_inference = time() - pre_inference
            print(f"Inference time: {post_inference:.3f}s")

        raw_shift = registration[0]["translation"]  # normalized (dx_norm, dy_norm)

        # ---- Update reference for next iteration NOW (loop logic fix) ----
        images_list.append(curr_frame)
        image_offsets.append(app.adjustments.get_illumination_shift())

        # ---- Apply counter-motion: invert BOTH axes before hardware mapping ----
        # raw_shift is normalized image motion; invert to counteract drift
        signed_norm = (-float(raw_shift[0]), -float(raw_shift[1]))

        # Convert normalized image shift → deflector delta (handles rotation & y-down)
        d_dx, d_dy = rotational_correction(
            signed_norm,
            fov_x=fov_x, fov_y=fov_y,
            theta_deg=theta_deg,
            y_down=True
        )

        # Apply correction relative to current deflector shift
        s = app.adjustments.get_illumination_shift()
        if logging:
            print("X shift um", s[0] * 1e6, "Y shift um", s[1] * 1e6)
        app.adjustments.set_illumination_shift(s[0] + d_dx, s[1] + d_dy)

    images_by_channel = {
        ch: [f[ch] for f in images_list if f.get(ch) is not None]
        for ch in ("BF", "HAADF")
        if any(f.get(ch) is not None for f in images_list)
    }

    results = []

    plot_flag = bool(logging)  # plot only on the first alignment we run

    if "BF" in images_by_channel and "HAADF" in images_by_channel:
        aligned_BF_series, summed_BF, _ = align_image_series(images_by_channel["BF"], plot=plot_flag)
        results.append((aligned_BF_series, summed_BF))
        plot_flag = False

        aligned_HAADF_series, summed_HAADF, _ = align_image_series(images_by_channel["HAADF"], plot=plot_flag)
        results.append((aligned_HAADF_series, summed_HAADF))

    elif "BF" in images_by_channel:
        aligned_BF_series, summed_BF, _ = align_image_series(images_by_channel["BF"], plot=plot_flag)
        results.append((aligned_BF_series, summed_BF))

    elif "HAADF" in images_by_channel:
        aligned_HAADF_series, summed_HAADF, _ = align_image_series(images_by_channel["HAADF"], plot=plot_flag)
        results.append((aligned_HAADF_series, summed_HAADF))

    return results

#TODO REFACTORED why not working?
def point_acquisition(pixel_offsets=None,dwell_time=None,return_metadata=True):

    app = get_app()
    N=1

    overview = app.scanning.get_pixel_count().value

    if pixel_offsets == None: #use center pixel
        rectangle = [overview/2,overview/2,N,N]
    else:
        rectangle = [pixel_offsets[0],pixel_offsets[1],N,N]
    app.scanning.set_off_axis(False)
    pointer = app.acquisition.acquire_camera(pixel_time=dwell_time, total_size=overview, frames=1, rectangle=rectangle,precession_enabled=None)
    image = pointer.get_frame()
    camera_data = image.camera[0][0]
    if return_metadata:
        metadata = collect_metadata(acquisition_type="4D")
        return camera_data,metadata
    else:
        return camera_data

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
        filename = filepath+f"\\precession angle {angle} degrees.tiff"
        cv2.imwrite(filename,image)
        annotated_images.append(image)

    return annotated_images,angle_list
