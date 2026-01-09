import numpy as np
import cv2 as cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from time import time

from expertpi import application, Config

from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT,DeflectorType as DFT


from RSTEM.app_context import get_app
from serving_manager.management.torchserve_rest_manager import TorchserveRestManager #serving manager 0.9.3
from serving_manager.tem_models.specific_model_functions import registration_model #serving manager 0.9.3

#config = Config()
config = Config(r"C:\Users\stem\Documents\Rob_coding\ExpertPI-0.5.1\config.yml") # path to config file if changes have been made, otherwise comment out and use default

"""Demo scripts showing the integration of TESCAN ML models into ExpertPI"""


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

def align_image_series(image_series, plot=False, host=None, model_name="TEMRegistration"):
    """
    Align a sequence of 2D images to the first frame using a TorchServe-hosted
    image-registration model and accumulate the aligned result.

    Each image in the input series is registered to the first image via a learned registration model
    (e.g. TEMRegistration) served throughTorchServe. The model is expected to return a normalised
     translation vector, which is converted into pixel shifts and applied using an affine warp.
     All aligned frames are summed to produce an accumulated image, and the per-frame shifts are recorded.

    Optionally,plots can be generated to visualise the reference image, final
    frame, summed image, and measured drift trajectory.

    Parameters
    ----------
    image_series : list of numpy.ndarray
        Non-empty list of 2D images to be aligned. All images must have identical shape and
        represent the same field of view. The first image in the list is used as the fixed
        reference for all registrations.
    plot : bool, optional
        If ``True``, displays a multi-panel matplotlib figure showing the reference image,
        the final frame in the series, the summed aligned image, and a scatter plot of the
        measured x/y shifts. Default is ``False``.
    host : str or None, optional
        Hostname or IP address of the TorchServe inference server. If ``None``, the value is
        taken from ``config.inference.host``.
    model : str, optional
        Name of the registration model deployed.  Default is ``"TEMRegistration"``. also possible are:
        "TEMRoma", "TEMRomaTiny"

    Returns
    -------
    translated_list : list of numpy.ndarray
        List of aligned images, each warped into the reference frame and returned as
        ``uint16`` arrays.
    summed_image : numpy.ndarray
        Floating-point image formed by summing all aligned frames. No normalisation or
        averaging is applied.
    shifts : list of tuple of float
        List of measured translations for each frame in pixel units, given as
        ``(x_shift_px, y_shift_px)`` relative to the reference image.

    Raises
    ------
    Exception
        If ``image_series`` is empty or not a list, or if the registration model cannot be
        contacted on the specified host.
    RuntimeError
        If the registration model returns an invalid or malformed result (e.g. missing or
        non-finite translation values).

    """

    if host is None:
        host = config.inference.host  #pull from config

    if not isinstance(image_series, list) or len(image_series) == 0:
        raise Exception("Dataset must be a non-empty list of images")

    initial_image = np.array(image_series[0], dtype=np.float64)
    H, W = initial_image.shape
    shifts = []
    translated_list = []

    manager = TorchserveRestManager(
        inference_port='8080',
        management_port='8081',
        host=host,
        image_encoder='.tiff'
    )
    try:
        manager.scale(model_name=model_name)
    except Exception:
        raise Exception(f"{model_name} cannot be contacted")

    acc = np.zeros((H, W), dtype=np.float64)

    for idx in tqdm(range(len(image_series)), desc="Aligning frames", unit="img"):
        moving_u16 = np.array(image_series[idx])
        moving_f = moving_u16.astype(np.float32,)

        reg_input = np.concatenate([initial_image, moving_f], axis=1)
        registration_values = registration_model(
            reg_input,
            model_name,
            host=host,
            port="8080",
            image_encoder='.tiff'
        )

        if not isinstance(registration_values, (list, tuple)) or len(registration_values) == 0:
            raise RuntimeError("registration_model returned no results")
        rv0 = registration_values[0]
        if not isinstance(rv0, dict) or "translation" not in rv0:
            raise RuntimeError("registration_model result missing 'translation'")
        translation = rv0["translation"]
        if (not isinstance(translation, (list, tuple)) or len(translation) != 2 or
                not np.all(np.isfinite(translation))):
            raise RuntimeError(f"Invalid 'translation' payload: {translation!r}")

        dx_norm, dy_norm = translation

        x_pixels_shift = dx_norm * W
        y_pixels_shift = dy_norm * H
        shifts.append((x_pixels_shift, y_pixels_shift))

        M = np.float32([[1, 0, x_pixels_shift], [0, 1, y_pixels_shift]])

        warped_f = cv2.warpAffine(
            moving_f,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        acc += warped_f.astype(np.float64)

        warped_u16 = np.clip(warped_f, 0, 65535).astype(np.uint16)
        translated_list.append(warped_u16)

    summed_image = acc

    if plot:
        xs, ys = zip(*shifts) if shifts else ([], [])
        fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)
        fig.set_size_inches(15, 5)
        ax1.imshow(initial_image, cmap="gray")
        ax1.title.set_text("Initial Image")
        plt.setp(ax1, xticks=[], yticks=[])
        ax2.imshow(image_series[-1], cmap="gray")
        ax2.title.set_text(f"Final Image")
        plt.setp(ax2, xticks=[], yticks=[])
        ax3.imshow(summed_image, cmap="gray")
        ax3.title.set_text(f"Summation of {len(image_series)} images")
        plt.setp(ax3, xticks=[], yticks=[])
        ax4.scatter(xs, ys, s=10)
        xs, ys = zip(*shifts) if shifts else ([], [])
        ax4.set_title("Measured drift")
        ax4.set_xlabel("x shift [px]")
        ax4.set_ylabel("y shift [px]")
        if xs and ys:
            t = np.arange(len(xs))  # 0..N-1 (acquisition index)
            # RdYlBu maps low->red, high->blue (so start=red, end=blue)
            sc = ax4.scatter(xs, ys, c=t, cmap="RdYlBu", s=16)
            # (optional) faint trajectory line for context
            ax4.plot(xs, ys, linewidth=0.8, alpha=0.4)
            # 1:1 data scale and square axes panel
            ax4.set_aspect('equal', adjustable='datalim')
            ax4.set_box_aspect(1)
            ax4.set_anchor('C')
            ax4.grid(True, alpha=0.2)
            # Colorbar keyed to acquisition index
            cb = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
            cb.set_label("Frame index (0 = start)")
        ax4.set_anchor('C')  # center the square axis in its slot
        ax4.autoscale_view()  # respect current data limits
        plt.show()

    return translated_list, summed_image, shifts

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