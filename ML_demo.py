import numpy as np
import cv2 as cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#from serving_manager.api import registration_model
#from serving_manager.api import super_resolution_model
#from serving_manager.api import TorchserveRestManager

from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT
from expert_pi.app import app
from expert_pi.gui import main_window

window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client


from serving_manager.management.torchserve_rest_manager import TorchserveRestManager #serving manager 0.9.3
from serving_manager.tem_models.specific_model_functions import registration_model #serving manager 0.9.3


host = "172.27.153.166"
#host = "192.168.51.3"

def view_all_models(host=None):
    if not host:
        host="192.168.51.3" #uses the local server host if it runs on a client PC
    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host, image_encoder='.tiff') #contacts the model manager
    models = manager.list_all_models() #gets all of the available models
    for i in models:
        print(i)

    return models

def align_image_series(image_series): #single series in a list

    initial_image = image_series[0] #first image in series
    initial_image_shape = initial_image.shape #shape of image
    shifts = [] #empty list for shifts
    translated_list = [] #empty list for output images
    for image in tqdm(range(len(image_series))): #iterate over series
        translated_image = image_series[image].astype(np.float64) #take the image and convert to 64 bit for ML model
        registration_values = registration_model(np.concatenate([initial_image,translated_image],axis=1), 'TEMRegistration', host=host, port='7443', image_encoder='.tiff') #run through registration model
        translation_values = registration_values[0]["translation"] #normalised between 0,1
        x_pixels_shift = translation_values[0]*initial_image_shape[0] #multiply by number of pixels to get pixel shift
        y_pixels_shift = translation_values[1]*initial_image_shape[1]
        shifts.append((x_pixels_shift,y_pixels_shift)) #write pixel shifts to list
        matrix = np.float32([[1,0,x_pixels_shift],[0,1,y_pixels_shift]]) #matrix to shift correct the images
        transposed_image = cv2.warpAffine(translated_image.astype(np.uint16),matrix,(initial_image_shape[1],initial_image_shape[0])) #offset the image by the measured pixel shifts
        translated_list.append(transposed_image) #add the shifted image to a list

    summing_array = np.asarray(translated_list)  #convert the list to an array
    summed_image = np.sum(summing_array, 0, dtype=np.float64)#sum the stack together


    fig,ax1,ax2,ax3 = plt.subplot(1,3)
    ax1.imshow(initial_image,cmap="grey")
    ax2.imshow(image_series[-1],cmap="grey")
    ax3.imshow(summed_image,cmap="grey")

    plt.show()

    return translated_list,summed_image,shifts #return the shifted images in a list, the summed image, and the shifts

def get_spot_positions(image,threshold=0):
    try:
        manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                        image_encoder='.tiff') #start server manager
        manager.scale(model_name="spot_segmentation",max_worker=2) #scale the server to have 2 processes
        results = manager.infer(image=image, model_name='spot_segmentation')  # send image to spot detection
        spots = results["objects"] #retrieve spot details
    except ConnectionError:
        print("Could not connect to model, check the host and if there are available workers")
    spot_list = [] #list for all spots
    areas = []
    spot_radii = []
    fix,ax = plt.subplots(1,1)
    shape = image.shape
    for i in range(len(spots)):
        if spots[i]["mean_intensity"] > threshold:
            spot_coords = spots[i]["center"] #spot center in fractions of image
            spot_list.append(spot_coords)
            area = spots[i]["area"] #spot area in fractional dimensions
            areas.append(area)
            radius = np.sqrt(area/np.pi)/shape[0] #calculate radius from area
            spot_radii.append(radius)

    for i in range(len(spot_list)):
        ax.plot(spot_list[i][0],spot_list[i][1],"r+") #plot spot center with red cross marker
        spot_marker = Circle(xy=(spot_list[i][0],spot_list[i][1]),radius=spot_radii[i],color="yellow",fill=False)
        ax.add_patch(spot_marker)
        plt.imshow(image,vmax=np.average(image*10),extent=(0,1,1,0),cmap="gray")


    plt.show(block=False)
    output= (spot_list,spot_radii)

    return output

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


def ML_drift_corrected_imaging(num_frames, pixel_time_us=None, num_pixels=None): #TODO untested live
    """Parameters
    num_frames : integer number of frames to acquire (total, including the seed frame)
    pixel_time_us: pixel dwell time in microseconds; if None, read from UI
    num_pixels: number of pixels in scanned image; default 1024

    Set the FOV and illumination conditions before calling; it uses current UI state.
    """

    # --- Inputs / defaults ---------------------------------------------------
    pixel_time = (window.scanning.pixel_time_spin.value() / 1e6) if (pixel_time_us is None) else (pixel_time_us / 1e6)
    num_pixels = 1024 if (num_pixels is None) else int(num_pixels)

    images_list   = []   # list of dict frames: {"BF": np.ndarray|None, "HAADF": np.ndarray|None}
    image_offsets = []   # list of deflector shift dicts (logged before applying correction)

    try:
        fov = float(grpc_client.scanning.get_field_width())
    except Exception as e:
        raise RuntimeError(f"Failed to read field width (FOV): {e}")
    if not np.isfinite(fov) or fov <= 0:
        raise RuntimeError(f"Invalid field width (FOV): {fov!r}")

    fov_x = fov_y = fov

    # --- Scan rotation (ok to default to 0° if unavailable) -----------------
    try:
        theta_val = np.rad2deg(grpc_client.scanning.get_scan_rotation())
        theta_deg = float(theta_val) if theta_val is not None else 0.0
    except Exception:
        raise RuntimeError(f"Cannot read Scan Rotation from HW")

    # Decide tracking signal
    bf_in    = grpc_client.stem_detector.get_is_inserted(DT.BF)
    haadf_in = grpc_client.stem_detector.get_is_inserted(DT.HAADF)

    if bf_in and haadf_in:
        tracking_signal = DT.BF
        grpc_client.projection.set_is_off_axis_stem_enabled(False)
    elif haadf_in:
        tracking_signal = DT.HAADF
        grpc_client.projection.set_is_off_axis_stem_enabled(False)
    elif bf_in:
        tracking_signal = DT.BF
        grpc_client.projection.set_is_off_axis_stem_enabled(False)
    else:
        tracking_signal = DT.BF
        grpc_client.projection.set_is_off_axis_stem_enabled(True)

    track_key = "BF" if tracking_signal == DT.BF else "HAADF"  # expected keys in stemData

    # Helpers
    def _detectors_for_scan(include_both_if_available=True):
        dets = []
        if bf_in: dets.append(DT.BF)
        if haadf_in: dets.append(DT.HAADF)
        if not dets:  # neither inserted → BF off-axis
            return [DT.BF]
        return dets if include_both_if_available else [tracking_signal]

    def _acquire_frame(det_list):
        scan_id = scan_helper.start_rectangle_scan(
            pixel_time=pixel_time,
            total_size=num_pixels,
            frames=1,
            detectors=det_list
        )
        _header, data = cache_client.get_item(scan_id, num_pixels ** 2)

        frame = {"BF": None, "HAADF": None}
        stem = data.get("stemData", {})

        # If your API uses enums as keys, adapt to: stem[DT.BF] / stem[DT.HAADF]
        if "BF" in stem and stem["BF"] is not None:
            frame["BF"] = np.asarray(stem["BF"]).reshape(num_pixels, num_pixels).astype(np.float64, copy=False)
        if "HAADF" in stem and stem["HAADF"] is not None:
            frame["HAADF"] = np.asarray(stem["HAADF"]).reshape(num_pixels, num_pixels).astype(np.float64, copy=False)

        return frame

    # --- tracking frame -----------------------------------------
    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)
    seed_frame = _acquire_frame(_detectors_for_scan(include_both_if_available=True))
    #if seed_frame[track_key] is None:
        # Fallback: acquire both if tracking-only failed
    #    seed_frame = _acquire_frame(_detectors_for_scan(include_both_if_available=True))
    #    if seed_frame[track_key] is None:
    #        raise RuntimeError(f"Tracking signal '{track_key}' not available in seed acquisition.")
    images_list.append(seed_frame)
    image_offsets.append(initial_shift)

    # --- Subsequent frames (register to previous) ---------------------------
    for frame_idx in range(1, num_frames):
        print(f"Acquiring frame {frame_idx} of {num_frames - 1}")
        curr_frame = _acquire_frame(_detectors_for_scan(include_both_if_available=True))

        prev_tracking = images_list[-1][track_key]
        curr_tracking = curr_frame[track_key]
        if prev_tracking is None or curr_tracking is None:
            # Skip correction if we lack the tracking channel
            images_list.append(curr_frame)
            image_offsets.append(grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan))
            continue

        # Registration input (current vs previous)
        reg_input = np.concatenate([curr_tracking, prev_tracking], axis=1)
        registration = registration_model(
            reg_input,
            'TEMRegistration',
            host=host, port='7443',
            image_encoder='.tiff'
        )
        raw_shift = registration[0]["translation"]  # normalized shift (dx_norm, dy_norm)

        # Convert image shift → deflector delta (θ applied here)
        d_dx, d_dy = rotational_correction(
            raw_shift, fov_x=fov_x, fov_y=fov_y, theta_deg=theta_deg, y_down=True
        )

        # Apply correction relative to current scan deflector shift
        s = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)
        grpc_client.illumination.set_shift(
            {"x": s["x"] + d_dx, "y": s["y"] + d_dy},
            grpc_client.illumination.DeflectorType.Scan
        )

        images_list.append(curr_frame)
        image_offsets.append(s)  # pre-correction log

    images_by_channel = {
        ch: [f[ch] for f in images_list if f.get(ch) is not None]
        for ch in ("BF", "HAADF")
        if any(f.get(ch) is not None for f in images_list)
    }

    results = []

    print("Post acquisition fine correction")
    if "BF" in images_by_channel:
        aligned_BF_series, summed_BF, _ = align_image_series(images_by_channel["BF"])
        results.append((aligned_BF_series,summed_BF))
    if "HAADF" in images_by_channel:
        aligned_HAADF_series, summed_HAADF, _ = align_image_series(images_by_channel["HAADF"])
        results.append((aligned_HAADF_series,summed_HAADF))

    return results