import numpy as np
import cv2 as cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from time import time

from expertpi import application, config

from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT,DeflectorType as DFT

#from expert_pi import grpc_client
#from expert_pi.app import scan_helper
#from expert_pi.grpc_client.modules._common import DetectorType as DT
#from expert_pi.app import app
#from expert_pi.gui import main_window

#window = main_window.MainWindow()
#controller = app.MainApp(window)
#cache_client = controller.cache_client
from RSTEM.app_context import get_app
from serving_manager.management.torchserve_rest_manager import TorchserveRestManager #serving manager 0.9.3
from serving_manager.tem_models.specific_model_functions import registration_model #serving manager 0.9.3

#config = Config()
config = Config(r"C:\Users\stem\Documents\Rob_coding\ExpertPI-0.5.1\config.yml") # path to config file if changes have been made, otherwise comment out and use default

def view_all_models(host=None):
    if host == None:
        host = config.inference.host

    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host, image_encoder='.tiff')
    models = manager.list_all_models() #gets all of the available models
    descriptions = []
    for i in models:
        description = manager.describe_model(i)
        print(f"{description[0]["modelName"]}, Version {description[0]["modelVersion"]}, Current workers available {description[0]["minWorkers"]}, Processor type {description[0]["deviceType"]}")
        descriptions.append((description[0]["modelName"],description[0]["minWorkers"]))
    return descriptions

def model_has_workers(model_name,host=None):

    if host == None:
        host = config.inference.host

    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                    image_encoder='.tiff')  # contacts the model manager
    model_status = manager.describe_model(model_name)
    if model_status[0]["minWorkers"] != 0:
        has_workers = True
    else: has_workers = False

    return has_workers

def align_image_series(image_series, plot=False,host=None):
    """
    Align a list of 2D images to the first frame using the TEMRegistration model
    'handles model scaling itself.
    Returns: (aligned_list, summed_image, shifts)
    """
    if host == None:
        host = config.inference.host

    if not isinstance(image_series, list) or len(image_series) == 0:
        raise Exception("Dataset must be a non-empty list of images")

    initial_image = np.array(image_series[0], dtype=np.float64, copy=False)
    H, W = initial_image.shape
    pixel_shifts = []
    translated_list = []

    manager = TorchserveRestManager(
        inference_port='8080', management_port='8081', host=host, image_encoder='.tiff')

    try:
        manager.scale(model_name="TEMRegistration")
    except Exception:
        raise Exception("Registration model cannot be contacted")
        return #if scaling does not work, end the alignment

    for idx in tqdm(range(len(image_series)), desc="Aligning frames", unit="img"):
        moving = np.array(image_series[idx], dtype=np.float64, copy=False)

        # Side-by-side concat (H, 2W), matching your registration model contract
        reg_input = np.concatenate([initial_image, moving], axis=1)
        registration_values = registration_model(reg_input,'TEMRegistration',host=host,port='7443',image_encoder='.tiff')
        dx_norm, dy_norm = registration_values[0]["translation"]
        x_pixels_shift = dx_norm * H
        y_pixels_shift = dy_norm * W
        pixel_shifts.append((x_pixels_shift, y_pixels_shift))

        M = np.float32([[1, 0, x_pixels_shift], [0, 1, y_pixels_shift]]) #affine warp matrix
        warped = cv2.warpAffine(moving.astype(np.uint16, copy=False),M,(W, H),flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        translated_list.append(warped)

    summed_image = np.sum(np.asarray(translated_list), axis=0, dtype=np.float64)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(15,5)
        ax1.imshow(initial_image, cmap="gray")
        ax1.title.set_text("Initial Image")
        plt.setp(ax1, xticks=[], yticks=[])  # removes ticks
        ax2.imshow(image_series[-1], cmap="gray")
        ax2.title.set_text("Final Image")
        plt.setp(ax2, xticks=[], yticks=[])  # removes ticks
        ax3.imshow(summed_image, cmap="gray")
        ax3.title.set_text(f"Summation of {len(image_series)+1} images")
        plt.setp(ax3, xticks=[], yticks=[])  # removes ticks

        plt.show(block=False)

    return translated_list, summed_image, pixel_shifts


def get_spot_positions(image,threshold=0,host=None,model_name="spot_segmentation",logging=True):

    if host == None:
        host = config.inference.host

    try:
        manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                        image_encoder='.tiff')  # start server manager
        if not model_has_workers(model_name,host=host):
            manager.scale(model_name=model_name) #scale the server to have 1 process

        if logging:
            process_pre = time()
        results = manager.infer(image=image, model_name=model_name)  # send image to spot detection
        if logging:
            inference_time = time()-process_pre
            print(f"Inference Time: {inference_time:3f}s")
        spots = results["objects"] #retrieve spot details
    except ConnectionError:
        print("Could not connect to model, check the model is available")
        return [],[]
    spot_list = [] #list for all spots
    areas = []
    spot_radii = []
    fig,ax = plt.subplots(1,1)
    shape = image.shape
    for spot in spots:
        if spot["mean_intensity"] > threshold:
            spot_coords = spot["center"]
            area = spot["area"]

            spot_list.append(spot_coords)
            areas.append(area)
            radius = np.sqrt(area / np.pi) / shape[0]
            spot_radii.append(radius)

    ax.imshow(image, vmax=np.average(image * 10), extent=(0, 1, 1, 0), cmap="gray")

    for coords, radius in zip(spot_list, spot_radii):
        ax.plot(coords[0], coords[1], "r+")
        spot_marker = Circle(xy=coords, radius=radius, color="yellow", fill=False)
        ax.add_patch(spot_marker)


    plt.show(block=False)

    return (spot_list,spot_radii)

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

#handles scan rotation imaging
#TODO refactored but untested
def drift_corrected_imaging(num_frames, pixel_time=None, num_pixels=None,host=None): #TODO full refactor needed
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
        fov = float(app.scanning.get_fov().value)
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
        raise RuntimeError(f"Cannot read Scan Rotation from HW")

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
    else: #if nothing is inserted use off axis BF
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

    def _acquire_frame(det_list,pixel_time,num_pixels):
        scan = app.acquisition.acquire_stem(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                            detectors=(DT.BF, DT.HAADF))
        image = scan.get_all()
        BF_image = image["BF"][0]
        ADF_image = image["HAADF"][0]

        frame = {"BF": BF_image, "HAADF": ADF_image}

        return frame

    if host is None:
        host = config.inference.host

    if pixel_time is None:
        pixel_time = app.scanning.get_pixel_time()
    if num_pixels is None:
        num_pixels = app.scanning.get_pixel_count().value

    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                        image_encoder='.tiff') #start serving manager
    manager.scale(model_name="TEMRegistration")

    # --- tracking frame -----------------------------------------
    initial_shift = app.api.illumination.get_shift(DFT.Scan)
    seed_frame = _acquire_frame(_detectors_for_scan(include_both_if_available=True),pixel_time,num_pixels)
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
        curr_frame = _acquire_frame(_detectors_for_scan(include_both_if_available=True),pixel_time,num_pixels)

        prev_tracking = images_list[-1][track_key]
        curr_tracking = curr_frame[track_key]
        if prev_tracking is None or curr_tracking is None:
            # Skip correction if we lack the tracking channel
            images_list.append(curr_frame)
            image_offsets.append(app.api.illumination.get_shift(DFT.Scan))
            continue

        # Registration input (current vs previous)
        reg_input = np.concatenate([curr_tracking, prev_tracking], axis=1)
        pre_inference = time()
        registration = registration_model(
            reg_input,
            'TEMRegistration',
            host=host, port='8080',
            image_encoder='.tiff'
        )
        post_inference = time()-pre_inference
        print(f"Inference time: {post_inference:3f}s")
        raw_shift = registration[0]["translation"]  # normalized shift (dx_norm, dy_norm)

        # Convert image shift → deflector delta #if scan rotation is present, will calculate the shifts using the rotation
        d_dx, d_dy = rotational_correction(
            raw_shift, fov_x=fov_x, fov_y=fov_y, theta_deg=theta_deg, y_down=True
        )

        # Apply correction relative to current scan deflector shift
        s = app.api.illumination.get_shift(DFT.Scan)
        app.api.illumination.set_shift(
            {"x": s["x"] + d_dx, "y": s["y"] + d_dy},
            DFT.Scan
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


def align_image_series(image_series, plot=False,host=None,model="TEMRegistration"):
    """
    Align a list of 2D images to the first frame using the TEMRegistration model
    'handles model scaling itself.
    Returns: (aligned_list, summed_image, shifts)
    """

    if host is None:
        host = config.inference.host

    #if model=="TEMRegistration":
    #    port = 7443
    #elif model == "TEMRoma" or "TEMRomaTiny":
    #    port = 8080

    if not isinstance(image_series, list) or len(image_series) == 0:
        raise Exception("Dataset must be a non-empty list of images")

    initial_image = np.array(image_series[0], dtype=np.float64, copy=False)
    H, W = initial_image.shape
    shifts = []
    translated_list = []

    # Self-contained TorchServe session (same model name)
    manager = TorchserveRestManager(
        inference_port='8080',
        management_port='8081',
        host=host,
        image_encoder='.tiff'
    )
    # Safe to call repeatedly; if already scaled/loaded, server should no-op
    try:
        manager.scale(model_name=model)
    except Exception:
        raise Exception(f"{model} cannot be contacted")
        #if scaling does not work, end the alignment

    for idx in tqdm(range(len(image_series)), desc="Aligning frames", unit="img"):
        moving = np.array(image_series[idx], dtype=np.float64, copy=False)

        # Side-by-side concat (H, 2W), matching registration model contract
        reg_input = np.concatenate([initial_image, moving], axis=1)
        registration_values = registration_model(
            reg_input,
            model,
            host=host,
            port="8080",
            image_encoder='.tiff'
        )
        # Expect [dx_norm, dy_norm] in [0..1]
        dx_norm, dy_norm = registration_values[0]["translation"]
        x_pixels_shift = dx_norm * H
        y_pixels_shift = dy_norm * W
        shifts.append((x_pixels_shift, y_pixels_shift))

        M = np.float32([[1, 0, x_pixels_shift], [0, 1, y_pixels_shift]])
        warped = cv2.warpAffine(
            moving.astype(np.uint16, copy=False),
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        translated_list.append(warped)

    summed_image = np.sum(np.asarray(translated_list), axis=0, dtype=np.float64)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(15,5)
        ax1.imshow(initial_image, cmap="gray")
        ax1.title.set_text("Initial Image")
        plt.setp(ax1, xticks=[], yticks=[])  # removes ticks
        ax2.imshow(image_series[-1], cmap="gray")
        ax2.title.set_text("Final Image")
        plt.setp(ax2, xticks=[], yticks=[])  # removes ticks
        ax3.imshow(summed_image, cmap="gray")
        ax3.title.set_text(f"Summation of {len(image_series)+1} images")
        plt.setp(ax3, xticks=[], yticks=[])  # removes ticks

        plt.show()

    return translated_list, summed_image, shifts
