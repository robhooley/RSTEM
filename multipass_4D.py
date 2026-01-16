from time import time,sleep
from importlib.util import find_spec
import os
import queue
import threading
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from time import time


from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT, DeflectorType as DFT
from expertpi.config import Config
from serving_manager.tem_models.specific_model_functions import registration_model
from serving_manager.management.torchserve_rest_manager import TorchserveRestManager


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
    from RSTEM.acquisitions import(
        rotational_correction
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
    from acquisitions import (
    rotational_correction
    )
    from analysis import (
    align_image_series,
    get_spot_positions
    )

#config = Config()
config = Config(r"C:\Users\stem\Documents\Rob_coding\ExpertPI-0.5.1\config.yml") # path to config file if changes have been made, otherwise comment out and use default

#concept
"""Take a reference STEM image, then collect a 4D-dataset at max camera speed, drift correct using fast STEM frames
then acquire subsequent 4D-STEM acquisitions with inter-frame drift correction from the fast STEM frames"""

"""After acquisition, sum the images together sequentially (1+2),(1+2+3),(1+2+3+4) to get dose sequence"""


def DCFI_4D(frames=10,scan_width=200,scan_height=None,use_precession=False,tracking_pixel_time=None,tracking_pixels=None,model_name="TEMRegistration",tracking=True,logging=True,host=None,save_to_disk=False,disk_cache_location=None):

    app = get_app()

    try:
        fov = float(app.scanning.get_fov())
    except Exception as e:
        raise RuntimeError(f"Failed to read field width (FOV): {e}")
    if not np.isfinite(fov) or fov <= 0:
        raise RuntimeError(f"Invalid field width (FOV): {fov!r}")

    fov_x = fov_y = fov

    try:
        theta_val = np.rad2deg(app.scanning.get_scanning_rotation())
        theta_deg = float(theta_val) if theta_val is not None else 0.0
    except Exception:
        raise RuntimeError(f"Cannot read Scan Rotation from HW")

    # Decide tracking signal

    bf_in = app.api.stem_detector.get_is_inserted(DT.BF)
    if bf_in:
        app.detectors.stem.insert_bf(False)
        app.detectors.stem.insert_df(False)
        for _ in tqdm(range(5), desc="Stabilising after STEM detector retraction", unit=""):
            sleep(1)

    def _start_npy_row_writer(out_npy_path: str, shape, dtype, queue_max: int = 2):
        """
        Background writer for row-chunks into a .npy (memmapped) array.

        Writes rows shaped (scanX, camY, camX) into target shaped (scanY, scanX, camY, camX).
        The queue items are (row_idx, row_block).
        """
        os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)

        q: "queue.Queue[tuple[int, np.ndarray] | None]" = queue.Queue(maxsize=queue_max)

        # .npy with header, but backed by memmap => fast, and easy to load later with np.load(mmap_mode="r")
        mm = open_memmap(out_npy_path, mode="w+", dtype=dtype, shape=shape)

        def _writer():
            try:
                while True:
                    item = q.get()
                    if item is None:
                        q.task_done()
                        break
                    row_idx, row_block = item
                    # row_block expected shape: (scanX, camY, camX)
                    mm[row_idx, :, :, :] = row_block
                    q.task_done()
                mm.flush()
            finally:
                # ensure any waiting join() isn't stuck if an exception happens
                while True:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except queue.Empty:
                        break

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        return mm, q, t

    def _acquire_frame(pixel_time,num_pixels):
        app.scanning.set_off_axis(True)
        scan = app.acquisition.acquire_stem(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                            detectors=(DT.BF))
        image = scan.get_all()

        frame = { "BF": image["BF"][0]}
        app.scanning.set_off_axis(False)
        return frame

    if host is None:
        host = config.inference.host
    if tracking:
        """Tracking acquisition setup"""

        if tracking_pixel_time is None:
            tracking_pixel_time = app.scanning.get_pixel_time()
        if tracking_pixels is None:
            tracking_pixels = app.scanning.get_pixel_count().value

        manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                            image_encoder='.tiff') #start serving manager
        manager.scale(model_name=model_name)

        images_list   = []   # list of dict frames: {"BF": np.ndarray|None, "BF": np.ndarray|None}
        image_offsets = []   # list of deflector shift dicts (logged before applying correction)


        # --- tracking frame -----------------------------------------
        initial_shift = app.adjustments.get_illumination_shift()
        seed_frame = _acquire_frame(tracking_pixel_time,tracking_pixels)
        images_list.append(seed_frame["BF"])
        image_offsets.append(initial_shift)


    if scan_height is not None:
        scan_dimensions=[scan_width,scan_height]
    else:
        scan_dimensions=[scan_width,scan_width]


    # --- Subsequent frames (register to previous) ---------------------------
    for frame_idx in range(0, frames):
        if logging:
            print(f"Acquiring frame {frame_idx} of {frames - 1}")
            """Active tracking of acquisition - optional"""

        if tracking:
            curr = _acquire_frame(tracking_pixel_time, tracking_pixels)

            prev_tracking = images_list[-1]  # last reference
            curr_tracking = curr["BF"]

            if prev_tracking is None or curr_tracking is None:
                images_list.append(curr["BF"])
                image_offsets.append(app.adjustments.get_illumination_shift())
                continue

            # 1) Measure shift (curr vs prev)
            reg_input = np.concatenate(
                [curr_tracking.astype(np.float32, copy=False),
                 prev_tracking.astype(np.float32, copy=False)],
                axis=1
            )
            pre_inference = time()
            registration = registration_model(
                reg_input,
                model_name,
                host=host, port='8080',
                image_encoder='.tiff'
            )
            post_inference = time() - pre_inference
            if logging:
                print(post_inference, "seconds")

            # 2) **Update reference for next iteration
            images_list.append(curr["BF"])

            image_offsets.append(app.adjustments.get_illumination_shift())

            dxn, dyn = registration[0]["translation"]  # normalized in image coords
            signed_norm = (-dxn, -dyn)  # <-- flip X and Y
            d_dx, d_dy = rotational_correction(
                signed_norm, fov_x=fov_x, fov_y=fov_y, theta_deg=theta_deg, y_down=True
            )

            s = app.adjustments.get_illumination_shift()
            if logging:
                print("X shift um", s[0] * 1e6, "Y shift um", s[1] * 1e6)
            app.adjustments.set_illumination_shift(s[0] + d_dx, s[1] + d_dy)

        """4D acquisition"""

        reader = app.acquisition.acquire_camera(
            pixel_time=1 / 72000,
            rectangle=(0,0,scan_dimensions[0],scan_dimensions[1]),
            total_size=(max(scan_dimensions[0], scan_dimensions[1])),
            precession_enabled=use_precession
        )

        print(f"Acquiring {scan_dimensions[0]} x {scan_dimensions[1]} px dataset at 72000 fps")

        scanX = scan_dimensions[0]
        scanY = scan_dimensions[1]

        # Retrieve first row to infer dtype/shape
        first_row = reader.get_lines(1)
        row_block = first_row.camera  # expected shape: (1, scanX, camY, camX)

        if row_block.ndim != 4 or row_block.shape[0] != 1 or row_block.shape[1] != scanX:
            raise RuntimeError(f"Unexpected cameraData shape: {row_block.shape}")

        camY, camX = row_block.shape[2], row_block.shape[3]
        dtype = row_block.dtype

        # in-RAM array; can drop it to save RAM.
        #image_array = np.empty((scanY, scanX, camY, camX), dtype=dtype, order="C")

        # Start background writer (one file per outer frame_idx)
        if save_to_disk:
            out_npy = os.path.join(disk_cache_location, f"/frame_{frame_idx:06d}.npy")
            _, write_q, write_thread = _start_npy_row_writer(
                out_npy_path=out_npy,
                shape=(scanY, scanX, camY, camX),
                dtype=dtype,
                queue_max=2,  # keep low to avoid RAM overloading
            )
        else:
            write_q = None
            write_thread = None

        # Assign the first row (row index 0)
        row0 = row_block[0]  # (scanX, camY, camX)
        #image_array[0, :, :, :] = row0
        if save_to_disk:
            write_q.put((0, row0))

        # Retrieve remaining rows
        for y in tqdm(
                range(1, scanY),
                desc="Retrieving remaining data from cache",
                total=scanY - 1,
                unit="rows"
        ):
            data = reader.get_lines(1)
            row_block = data.camera  # (1, scanX, camY, camX)
            rowy = row_block[0]  # (scanX, camY, camX)

            #image_array[y, :, :, :] = rowy
            if save_to_disk:
                write_q.put((y, rowy))

        if save_to_disk:
            # Finish writer cleanly for this dataset
            write_q.put(None)
            write_q.join()
            write_thread.join()





