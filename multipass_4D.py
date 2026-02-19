from time import time,sleep
from importlib.util import find_spec
import os
import queue
import threading
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from time import time
from numcodecs import Blosc
import zarr
import json
from natsort import natsorted


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
                                            detectors=([DT.BF]))
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

def DCFI_4D(
    frames=10,
    scan_width=200,
    roi_mode=None,
    scan_height=None,
    use_precession=False,
    tracking_pixel_time=None,
    tracking_pixels=None,
    model_name="TEMRomaTiny",
    tracking=True,
    logging=True,
    host=None,
    save_to_disk=False,
    disk_cache_location=None,
    save_format="zarr",
):
    """
    Drift-corrected frame imaging with optional 4D camera acquisition.

    save_format:
      - "npy": writes one .npy per outer frame_idx using a memmapped background writer
      - "zarr": writes one .zarr store per outer frame_idx using compressed Zarr background writer
    """

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

    def _choose_zarr_chunks(scanX: int, camY: int, camX: int, dtype, target_mb: int = 16):
        """
        Choose chunks of form (1, scanX, tileY, tileX) targeting ~target_mb per chunk.
        Avoids massive chunks like (1, scanX, 512, 512) which are too large for efficient compression.
        """
        bytes_per_px = np.dtype(dtype).itemsize
        target_bytes = target_mb * 1024 * 1024
        candidates = [256, 192, 160, 128, 96, 80, 64, 48, 32]

        best = (1, scanX, min(128, camY), min(128, camX))
        best_gap = float("inf")

        for ty in candidates:
            if ty > camY:
                continue
            for tx in candidates:
                if tx > camX:
                    continue
                chunk_bytes = 1 * scanX * ty * tx * bytes_per_px
                if chunk_bytes <= target_bytes:
                    gap = target_bytes - chunk_bytes
                    if gap < best_gap:
                        best = (1, scanX, ty, tx)
                        best_gap = gap

        return best

    def _start_zarr_row_writer(out_zarr_path: str, dataset_name: str, shape, dtype, scanX: int, queue_max: int = 4):
        """
        Background writer for row-chunks into a Zarr dataset (compressed).

        Writes rows shaped (scanX, camY, camX) into target shaped (scanY, scanX, camY, camX).
        The queue items are (row_idx, row_block).
        """
        os.makedirs(os.path.dirname(out_zarr_path), exist_ok=True)

        q: "queue.Queue[tuple[int, np.ndarray] | None]" = queue.Queue(maxsize=queue_max)

        # Good live-acquisition default: fast compression for integer data
        compressor = Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE)

        scanY, _, camY, camX = shape
        chunks = _choose_zarr_chunks(scanX=scanX, camY=camY, camX=camX, dtype=dtype, target_mb=16)

        root = zarr.open(out_zarr_path, mode="a")
        arr = root.require_dataset(
            name=dataset_name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            overwrite=True,
        )

        def _writer():
            try:
                while True:
                    item = q.get()
                    if item is None:
                        q.task_done()
                        break
                    row_idx, row_block = item
                    # row_block expected shape: (scanX, camY, camX)
                    arr[row_idx, :, :, :] = row_block
                    q.task_done()
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
        return arr, q, t, chunks

    def _acquire_frame(pixel_time, num_pixels):
        app.scanning.set_off_axis(True)
        scan = app.acquisition.acquire_stem(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=([DT.BF]))
        image = scan.get_all()
        frame = {"BF": image["BF"][0]}
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

        manager = TorchserveRestManager(
            inference_port="8080", management_port="8081", host=host, image_encoder=".tiff"
        )  # start serving manager
        manager.scale(model_name=model_name)

        images_list = []    # list of tracking frames: [np.ndarray]
        image_offsets = []  # list of illumination shifts (logged before applying correction)

        # --- tracking seed frame -----------------------------------------
        initial_shift = app.adjustments.get_illumination_shift()
        seed_frame = _acquire_frame(tracking_pixel_time, tracking_pixels)
        images_list.append(seed_frame["BF"])
        image_offsets.append(initial_shift)

    if scan_height is not None:
        scan_dimensions = [scan_width, scan_height]
    else:
        scan_dimensions = [scan_width, scan_width]

    if save_to_disk:
        if disk_cache_location is None:
            raise RuntimeError("save_to_disk=True but disk_cache_location is None")
        if save_format not in ("npy", "zarr"):
            raise ValueError(f"Invalid save_format '{save_format}', must be 'npy' or 'zarr'")

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
                [curr_tracking.astype(np.float32, copy=False), prev_tracking.astype(np.float32, copy=False)],
                axis=1,
            )
            pre_inference = time()
            registration = registration_model(
                reg_input,
                model_name,
                host=host,
                port="8080",
                image_encoder=".tiff",
            )
            post_inference = time() - pre_inference
            if logging:
                print(post_inference, "seconds")

            # 2) Update reference for next iteration
            images_list.append(curr["BF"])
            image_offsets.append(app.adjustments.get_illumination_shift())

            dxn, dyn = registration[0]["translation"]  # normalized in image coords
            signed_norm = (-dxn, -dyn)  # flip X and Y
            d_dx, d_dy = rotational_correction(
                signed_norm, fov_x=fov_x, fov_y=fov_y, theta_deg=theta_deg, y_down=True
            )

            s = app.adjustments.get_illumination_shift()
            if logging:
                print("X shift um", s[0] * 1e6, "Y shift um", s[1] * 1e6)
            app.adjustments.set_illumination_shift(s[0] + d_dx, s[1] + d_dy)

        """4D acquisition setup"""

        if roi_mode == 128:  # 512x128 px
            app.detectors.camera.set_roi(RM.Lines_128)
            dwell_time=1/18000
        elif roi_mode == 256:  # 512x256 px
            app.detectors.camera.set_roi(RM.Lines_256)
            dwell_time=1/9000
        else:
            app.detectors.camera.set_roi(RM.Disabled)
            dwell_time = 1/72000



        reader = app.acquisition.acquire_camera(
            pixel_time=np.round(dwell_time,8),
            rectangle=(0, 0, scan_dimensions[0], scan_dimensions[1]),
            total_size=(max(scan_dimensions[0], scan_dimensions[1])),
            precession_enabled=use_precession,
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

        # Start background writer (one file/store per outer frame_idx)
        if save_to_disk:
            if save_format == "zarr":
                out_zarr = os.path.join(disk_cache_location, f"frame_{frame_idx:06d}.zarr")
                _, write_q, write_thread, zarr_chunks = _start_zarr_row_writer(
                    out_zarr_path=out_zarr,
                    dataset_name="raw",
                    shape=(scanY, scanX, camY, camX),
                    dtype=dtype,
                    scanX=scanX,
                    queue_max=4,
                )
                if logging:
                    print(f"Saving to Zarr ({dtype}, chunks={zarr_chunks}) -> {out_zarr}")
            else:  # "npy"
                out_npy = os.path.join(disk_cache_location, f"frame_{frame_idx:06d}.npy")
                _, write_q, write_thread = _start_npy_row_writer(
                    out_npy_path=out_npy,
                    shape=(scanY, scanX, camY, camX),
                    dtype=dtype,
                    queue_max=2,  # keep low to avoid RAM overloading
                )
                if logging:
                    print(f"Saving to NPY -> {out_npy}")

        else:
            write_q = None
            write_thread = None

        # Enqueue the first row (row index 0)
        row0 = row_block[0]  # (scanX, camY, camX)
        if save_to_disk:
            write_q.put((0, row0))

        # Retrieve remaining rows
        for y in tqdm(
            range(1, scanY),
            desc="Retrieving remaining data from cache",
            total=scanY - 1,
            unit="rows",
        ):
            data = reader.get_lines(1)
            row_block = data.camera  # (1, scanX, camY, camX)
            rowy = row_block[0]      # (scanX, camY, camX)

            if save_to_disk:
                write_q.put((y, rowy))

        if save_to_disk:
            # Finish writer
            write_q.put(None)
            write_q.join()
            write_thread.join()

    #TODO test
    metadata = collect_metadata(acquisition_type="4D",scan_width_px=scan_width,num_frames=frames)
    out_metadata = os.path.join(disk_cache_location, f"multipass_4D_metadata.json")
    open_json = open(out_metadata, "w")
    json.dump(metadata, open_json, indent=6)
    open_json.close()

    #reset ROI mode after acquisition
    app.detectors.camera.set_roi(RM.Disabled)

def cumulative_sum_frames_to_individual_outputs(
    input_dir: str,
    output_dir: str,
    *,
    input_zarr_dataset_name: str = "raw",
    output_format: str = "zarr",            # "zarr" or "npy"
    output_zarr_dataset_name: str = "sum",  # dataset name inside each output .zarr
    accum_dtype=np.uint16,                  # safe for <=257 uint8 frames
    compressor=None,                        # for Zarr outputs only
    chunk_target_mb: int = 16,              # for Zarr outputs only
    validate_shapes: bool = True,
):
    """
    Creates cumulative sums as individual outputs:
      cumsum_000001 = frame0 + frame1
      cumsum_000002 = frame0 + frame1 + frame2
      ...

    Input directory must contain files like:
      - frame_000000.npy
      - frame_000000.zarr  (dataset inside = input_zarr_dataset_name)

    Output directory will receive either:
      - cumsum_000001.npy, cumsum_000002.npy, ...
      - cumsum_000001.zarr, cumsum_000002.zarr, ...  (dataset inside = output_zarr_dataset_name)
    """

    # ---- validate inputs (strings only, no Path objects) -------------------
    if not isinstance(input_dir, str) or not input_dir.strip():
        raise ValueError("input_dir must be a non-empty string")
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty string")
    os.makedirs(output_dir, exist_ok=True)

    if output_format not in ("zarr", "npy"):
        raise ValueError("output_format must be 'zarr' or 'npy'")

    # ---- discover + natsort frames (inlined, as requested) -----------------
    frames = natsorted([
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().startswith("frame_") and name.lower().endswith((".npy", ".zarr"))
    ])

    if len(frames) < 2:
        raise ValueError(
            f"Need at least 2 frames in {input_dir}. Expected files like frame_000000.npy/.zarr"
        )

    # ---- compressor default (Zarr outputs only) ----------------------------
    if compressor is None:
        # Good balance for integer cumulative sums
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    # ---- open first frame (memmap/zarr) to get shape -----------------------
    first_path = frames[0]
    if first_path.lower().endswith(".npy"):
        first = np.load(first_path, mmap_mode="r")
    else:
        root = zarr.open(first_path, mode="r")
        if input_zarr_dataset_name not in root:
            raise KeyError(f"Zarr store {first_path} does not contain dataset '{input_zarr_dataset_name}'")
        first = root[input_zarr_dataset_name]

    frame_shape = tuple(first.shape)
    if len(frame_shape) != 4:
        raise RuntimeError(f"Expected 4D frames (scanY, scanX, camY, camX), got {frame_shape}")

    scanY, scanX, camY, camX = frame_shape

    # ---- safety for uint16 accumulator with uint8 inputs -------------------
    # If your inputs are uint8 and frames<200, uint16 is perfect (safe up to 257 frames).
    if accum_dtype == np.uint16:
        max_frames_safe = 65535 // 255  # 257
        if len(frames) > max_frames_safe:
            raise ValueError(
                f"uint16 accumulator only safe up to {max_frames_safe} uint8 frames; found {len(frames)}. "
                f"Use accum_dtype=np.uint32."
            )

    # ---- disk-backed accumulator (RAM-stable) ------------------------------
    acc_path = os.path.join(output_dir, "_accumulator_tmp.npy")
    acc = np.lib.format.open_memmap(acc_path, mode="w+", dtype=accum_dtype, shape=frame_shape)
    acc[:] = 0
    acc.flush()

    # Add frame0 once up front (no dtype-cast temporary)
    np.add(acc, first, out=acc, casting="unsafe")
    acc.flush()

    # ---- choose Zarr output chunks (if needed) -----------------------------
    out_chunks = None
    tileY = tileX = None
    if output_format == "zarr":
        # inline chunk chooser (keeps ~chunk_target_mb per chunk)
        bytes_per_px = np.dtype(accum_dtype).itemsize
        target_bytes = int(chunk_target_mb * 1024 * 1024)
        candidates = [256, 192, 160, 128, 96, 80, 64, 48, 32]

        best = (1, scanX, min(128, camY), min(128, camX))
        best_gap = float("inf")
        for ty in candidates:
            if ty > camY:
                continue
            for tx in candidates:
                if tx > camX:
                    continue
                chunk_bytes = 1 * scanX * ty * tx * bytes_per_px
                if chunk_bytes <= target_bytes:
                    gap = target_bytes - chunk_bytes
                    if gap < best_gap:
                        best = (1, scanX, ty, tx)
                        best_gap = gap

        out_chunks = best
        _, _, tileY, tileX = out_chunks

    written = []

    # ---- main loop: frame1..end, each step writes one cumulative output ----
    for idx in tqdm(range(1, len(frames)), desc="Summing frames", unit="frame", total=len(frames) - 1):
        frame_path = frames[idx]

        if frame_path.lower().endswith(".npy"):
            frame = np.load(frame_path, mmap_mode="r")
        else:
            root = zarr.open(frame_path, mode="r")
            if input_zarr_dataset_name not in root:
                raise KeyError(f"Zarr store {frame_path} does not contain dataset '{input_zarr_dataset_name}'")
            frame = root[input_zarr_dataset_name]

        if validate_shapes and tuple(frame.shape) != frame_shape:
            raise RuntimeError(
                f"Shape mismatch in {os.path.basename(frame_path)}: {tuple(frame.shape)} vs {frame_shape}"
            )

        # Update accumulator on disk without allocating dtype-cast temporaries
        np.add(acc, frame, out=acc, casting="unsafe")
        acc.flush()

        out_idx = idx  # idx=1 -> cumsum_000001 (frame0+frame1)

        if output_format == "npy":
            out_path = os.path.join(output_dir, f"cumsum_{out_idx:06d}.npy")
            out_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=accum_dtype, shape=frame_shape)
            out_mm[:] = acc
            out_mm.flush()
            written.append(out_path)
        else:
            out_store = os.path.join(output_dir, f"cumsum_{out_idx:06d}.zarr")
            out_root = zarr.open(out_store, mode="w")
            dset = out_root.create_dataset(
                name=output_zarr_dataset_name,
                shape=frame_shape,
                chunks=out_chunks,
                dtype=accum_dtype,
                compressor=compressor,
                overwrite=True,
            )

            # Write chunk-by-chunk (bounded RAM, compression-friendly)
            for y in range(scanY):
                for y0 in range(0, camY, tileY):
                    y1 = min(y0 + tileY, camY)
                    for x0 in range(0, camX, tileX):
                        x1 = min(x0 + tileX, camX)
                        dset[y, :, y0:y1, x0:x1] = acc[y, :, y0:y1, x0:x1]

            written.append(out_store)

    # ---- cleanup temp accumulator ------------------------------------------
    try:
        os.remove(acc_path)
    except Exception:
        pass

    return written

def load_frame(path: str, dataset_name="raw"):
    if path.lower().endswith(".npy"):
        return np.load(path, mmap_mode="r")
    elif path.lower().endswith(".zarr"):
        return zarr.open(path, mode="r")[dataset_name]
    else:
        raise ValueError(path)
