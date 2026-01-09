import os
import pickle
import numpy as np
import easygui as g
import cv2 as cv2
from tqdm import tqdm
from sys import getsizeof
import matplotlib.colors as mcolors
from collections import defaultdict
from time import sleep,time
import xraydb as xdb
import scipy.signal
import matplotlib.pyplot as plt
from grid_strategy import strategies
from PIL import Image
#from serving_manager.api import registration_model
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_float32
import hyperspy.api as hs
import numpy as np

from importlib.util import find_spec

from expertpi import application
from expertpi.config import Config
from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT,DeflectorType as DFT


#config = Config()
config = Config(r"C:\Users\stem\Documents\Rob_coding\ExpertPI-0.5.1\config.yml") # path to config file if changes have been made, otherwise comment out and use default

from serving_manager.management.torchserve_rest_manager import TorchserveRestManager #serving manager 0.9.3
from serving_manager.tem_models.specific_model_functions import registration_model #serving manager 0.9.3


if find_spec("RSTEM.app_context") is not None:
    from RSTEM.app_context import get_app
    from RSTEM.utilities import (
        model_has_workers,
        collect_metadata,
        generate_colourlist,
        generate_colourmaps,
        rotational_correction
        )
    from RSTEM.analysis import(
    align_image_series)
else:
    from app_context import get_app
    from utilities import (
        model_has_workers,
        collect_metadata,
        generate_colourlist,
        generate_colourmaps,
        rotational_correction)
    from analysis import (align_image_series)


#from expert_pi.measurements import edx_processing

#TODO update to ML with scan rotation
def acquire_EDX_map(frames=10,pixel_time=10e-6,num_pixels=None,host=None,model_name="TEMRegistration",logging=True):
    """Parameters
    frames: number of scans
    pixel_time: in seconds
    fov: in microns
    scan_rotation in degrees
    num_pixels: scan dimensions
    drift_correction_method: either "patches" for openCV template matching, "ML" uses trained AI drift correction"""


    """Parameters
    num_frames : integer number of frames to acquire (total, including the seed frame)
    pixel_time_us: pixel dwell time in microseconds; if None, read from UI
    num_pixels: number of pixels in scanned image; default 1024

    Set the FOV and illumination conditions before calling; it uses current UI state.
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

    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)
    if haadf_in == False:
        app.detectors.stem.insert_df(True)
        app.scanning.set_off_axis(False)
        sleep(5)

    #tracking_signal = DT.HAADF
    #track_key = "HAADF"  # expected keys in stemData

    # Helpers


    def _acquire_frame(pixel_time,num_pixels):
        scan = app.acquisition.acquire_stem(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                            detectors=(DT.BF,DT.HAADF),edx_detectors=(DT.EDX0,DT.EDX1))
        image = scan.get_all()
        EDX_data = scan.edx.get_frame(0)

        frame = { "HAADF": image["HAADF"][0], "EDX": EDX_data}
        return frame

    if host is None:
        host = config.inference.host

    if pixel_time is None:
        pixel_time = app.scanning.get_pixel_time()
    if num_pixels is None:
        num_pixels = app.scanning.get_pixel_count().value

    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                        image_encoder='.tiff') #start serving manager
    manager.scale(model_name=model_name)

    images_list   = []   # list of dict frames: {"BF": np.ndarray|None, "HAADF": np.ndarray|None}
    image_offsets = []   # list of deflector shift dicts (logged before applying correction)
    map_data_list = []

    # --- tracking frame -----------------------------------------
    initial_shift = app.adjustments.get_illumination_shift()
    seed_frame = _acquire_frame(pixel_time,num_pixels)
    images_list.append(seed_frame["HAADF"])
    map_data_list.append(seed_frame["EDX"])
    image_offsets.append(initial_shift)

    # --- Subsequent frames (register to previous) ---------------------------
    for frame_idx in range(1, frames):
        if logging:
            print(f"Acquiring frame {frame_idx} of {frames - 1}")

        curr = _acquire_frame(pixel_time, num_pixels)

        prev_tracking = images_list[-1]  # last reference
        curr_tracking = curr["HAADF"]

        if prev_tracking is None or curr_tracking is None:
            images_list.append(curr["HAADF"])
            map_data_list.append(curr["EDX"])
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

        #raw_shift = registration[0]["translation"]  # normalized (dx_norm, dy_norm)

        # 2) **Update reference for next iteration NOW**
        images_list.append(curr["HAADF"])
        map_data_list.append(curr["EDX"])
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

    aligned_HAADF_series, summed_HAADF,pixel_shifts = align_image_series(images_list,model_name=model_name,plot=logging)

    results = aligned_HAADF_series,summed_HAADF,images_list,map_data_list,pixel_shifts

    return results





    """map_data = []
    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=[DT.BF, DT.HAADF, DT.EDX1, DT.EDX0],use_precession=precession)
    header, data = cache_client.get_item(scan_id, num_pixels**2)
    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)
    map_data.append((header,data,initial_shift))

    with open(f"{folder}{file_name}_0.pdat", "wb") as f:
        pickle.dump(map_data[0], f) #first item from map data list of scans

    image_list = []
    anchor_image = data["stemData"][tracking_signal].reshape(num_pixels, num_pixels)
    image_list.append(anchor_image)

    for i in range(1, frames):
        print(f"Acquiring frame {i} of {frames}")
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=[DT.BF, DT.HAADF, DT.EDX1, DT.EDX0])
        header, data = cache_client.get_item(scan_id, num_pixels**2)
        current_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

        with open(f"{folder}{file_name}_{i}.pdat", "wb") as f:
            pickle.dump((header, data, current_shift), f)

        # apply drift correction between images:
        series_image = data["stemData"][tracking_signal].reshape(num_pixels, num_pixels)

        #TODO tidy up the shift offset inconsistencies


        image_list.append(series_image)
        map_data.append((header,data,current_shift))



    return map_data"""


def construct_maps(map_data=None, elements=None, end_of_series=None, mode=None):
    """parameters
    map_data : if map_data is stored in RAM, or left as None to load from file
    elements : List of elements in string format ["Cu","Al","Ti"]
    mode : Keep as None, mode="Explore" uses Explores color palette"""

    if elements is None:
        elements = ["Cu"]
    edx_data = []
    BFs = []
    ADFs = []

    if map_data == None:
        map_data = []
        data_folder = g.diropenbox("data folder","data folder")
        file_path = os.listdir(data_folder)
        for file in tqdm(file_path):  # iterates through folder with a progress bar
            path = data_folder + "\\" + file  # picks individual images
            if file.endswith(".pdat"):
                with open(path, "rb") as f:
                    header, data, s0 = pickle.load(f)
                    map_data.append((header,data))

    scan_pixels = map_data[0][0]["scanDimensions"]

    if end_of_series is not None: #default is to use all frames
        num_frames = end_of_series #sets the end frame of the series

    for i in range(num_frames): #map data stored as tuple of (header,data,shifts)
        frame = map_data[i]
        data = frame[1]
        edx_data.append(data["edxData"]["EDX0"]["energy"])
        edx_data.append(data["edxData"]["EDX1"]["energy"])
        BFs.append(data["stemData"]["BF"].reshape(scan_pixels[1], scan_pixels[2]))
        ADFs.append(data["stemData"]["HAADF"].reshape(scan_pixels[1], scan_pixels[2]))

    energies = np.concatenate([e[0] for e in edx_data])
    pixels = np.concatenate([e[1] for e in edx_data])

    def generate_base_function(energy, energies, MnFWHM=140):
        def gauss(x, *p):
            A, mu, sigma = p
            return A/sigma*np.exp(-(x - mu)**2/(2.*sigma**2))

        mu = energy
        sigma = MnFWHM/2.3548/np.sqrt(5900.3)*np.sqrt(mu)
        result = gauss(energies, 1, mu, sigma)

        return result/np.sum(result)

    # generate gaussians for each line of elements:
    lines = get_xray_lines(elements)#edx_processing.get_edx_filters(elements)
    E = np.array(range(0, 2**16))  # max energy 2**16 kV
    bases = []

    map_names = []
    map_energies = []
    for name,value in lines.items():
        #map_names.append(name)
        for map_energy in value:
            if map_energy not in map_energies:
                map_energies.append(map_energy)
                map_names.append(name)

    for value in map_energies:
        base = generate_base_function(value, E)
        print(base)
        bases.append(base)
    N = len(bases)
    bases = np.array(bases)
    # generate basis matrix:
    M = []
    for i in range(N):
        M.append([])
        for j in range(N):
            M[i].append(np.sum(bases[i]*bases[j]))
    Mi = np.linalg.inv(M) #this cannot handle duplicate bases, so need to filter out duplicate energies

    # now we remap the energy events to the probabilities of the element (it is same as linear fitting):
    print("number of X-ray events -",len(energies))
    batch = 400_000
    channels = np.zeros((N, (scan_pixels[1]* scan_pixels[2])))

    i = 0
    while i < len(energies):
        probabilities = bases[:, energies[i:i + batch]]
        channels[:, pixels[i:i + batch]] += np.dot(Mi, probabilities)
        i += batch
        print(f"energies->elements {i/len(energies)*100:6.3f}%", end="\r")
    channels = channels.reshape(N, scan_pixels[1], scan_pixels[2])

    # filter images
    median_filter = 5
    dilation = 2
    kernel = np.ones((3, 3), np.uint8)

    channels_filtered = channels*0
    print("filtering")
    for i in range(N):
        channels_filtered[i, :, :] = cv2.dilate(channels[i, :, :], kernel, iterations=dilation)
        channels_filtered[i, :, :] = scipy.signal.medfilt2d(channels_filtered[i, :, :], median_filter)
        print(i, end="\r")

    imgs_all = {}

    for i in range(N):
        imgs_all[map_names[i]]=channels_filtered[i, :, :]

    # plot them
    num_maps = len(imgs_all)

    imgs_list = []
    names_list = []
    for name in imgs_all.keys():
        img = imgs_all[name]
        imgs_list.append(img)
        names_list.append(name)

    specs = strategies.SquareStrategy("center").get_grid(num_maps)
    color_maps = generate_colourmaps(num_maps, mode=mode)

    for i,subplot in enumerate(specs):
        plt.subplot(subplot)
        ax = plt.gca()
        image = imgs_list[i]
        name = names_list[i]
        ax.imshow(image,cmap=color_maps[i])
        if mode == "Explore":
            ax.set_title(name,color="pink")

    if mode == "Explore":
        plt.suptitle("EDX Mapping, Explore style",color="pink")

    plt.show()

    # saving:
    save_folder = g.diropenbox("Select folder to save maps into")
    for name, img in imgs_all.items():
        img = img.astype("float")
        img[img < 0] = 0
        im = Image.fromarray((img/np.max(img)*(2**15 - 1)).astype('uint16'))
        im.save(save_folder + '/' + str(name) + '.tif')


    BF = BFs[0]
    cv2.imwrite(save_folder + '/BF_0.tif',BF)

    HAADF = ADFs[0]
    cv2.imwrite(save_folder + '/HAADF_0.tif',HAADF)


def produce_spectrum(map_data=None,elements=None,normalise=False):
    edx_data = []

    print("loading data")
    if map_data == None: #loads map data from disk
        map_data = []
        data_folder = g.diropenbox("data folder","data folder")
        file_path = os.listdir(data_folder) #file path
        for file in tqdm(file_path):  # iterates through folder with a progress bar
            path = data_folder + "\\" + file  # picks individual images
            if file.endswith(".pdat"): #only collects .pdat files
                with open(path, "rb") as f:
                    header, data, s0 = pickle.load(f)
                    map_data.append((header,data)) #adds to list of pdat files

    for i in range(len(map_data)): #map data stored as tuple of (header,data,shifts)
        frame = map_data[i]
        data = frame[1]
        edx_data.append(data["edxData"]["EDX0"]["energy"])
        edx_data.append(data["edxData"]["EDX1"]["energy"])

    energies = np.concatenate([e[0] for e in edx_data])

    histogram,bin_edges = np.histogram(energies,bins=4000,range=(0,20000))

    if normalise:
        max_count = max(histogram[200:])
        histogram = histogram/max_count
        plt.stairs(histogram, bin_edges, color="black")
        plt.ylabel("Normalised intensity")
    else:
        plt.stairs(histogram,bin_edges,color="black")

        plt.ylabel("Intensity (Counts)")

    plt.xlim(left=200)
    plt.xlabel("Energy (eV)")



    if elements is not None:
        lines = get_xray_lines(elements=elements)#,lines=['Ka1','La1',"Ma"])
        #print(lines)
        colors = generate_colourlist(len(lines))
        i=0
        for name,line in zip(lines.keys(),lines.values()):
            energy = line
            plt.axvline(energy,linestyle="dotted",color=colors[i],label=name)
            plt.text(energy,1.1,name,rotation="vertical")
            i+=1

    plt.legend()
    plt.show()
    return (histogram,bin_edges)


    #TODO functional but needs commments and refactoring

def get_xray_lines(elements=[],intensity_threshold=0.05):
    """Params
    elements : List of elements in individual strings ["Al","Cu","Fe"],
    intensity_threshold : minimum peak intensity normalised to the strongest peak
    returns a dictionary with the X-ray line family and energy"""

    def list_duplicates(seq):
        tally = defaultdict(list)
        for i, item in enumerate(seq):
            tally[item].append(i)
        return ((key, locs) for key, locs in tally.items()
                if len(locs) > 1)

    line_family_list = []
    family_energy_list = []
    line_dictionary = {} #opens empty dictionary to hold lines and energies
    for element in elements: #for each element
        line_details = [] #empty list for line names
        line_energies = [] #empty list for line energies
        lines = xdb.xray_lines(element) #gets all lines for an element
        for line_name,info in zip(lines.keys(),lines.values()): #for each line
            if info.intensity >= intensity_threshold and 20000 >= info.energy >= 200: #filters out low,high energy and minor lines
                line_name = line_name[:2] #cuts it to just the first two characters ie ka2 to ka
                line_detail = (element+" "+line_name) #adds the name to the list
                #print(line_detail)
                line_details.append(line_detail) #adds the name to the list
                line_energies.append(info.energy) #adds the energy to the list

        for duplicate in sorted(list_duplicates(line_details)): #TODO refactor and tidy up
                line_family_indexes = duplicate[1]
                family_energy = []
                for index in line_family_indexes:
                    #print(line_energies[index])
                    family_energy.append(line_energies[index])
                family_energy_list.append(family_energy)
                line_family_list.append(duplicate[0])
    #for name,energy in zip(line_details,line_energies): #converts the lists to a dictionary
    for name,energy in zip(line_family_list,family_energy_list): #converts the lists to a dictionary
            line_dictionary[name]=energy

    line_tuple_list = []
    for item in range(len(line_family_list)):
        line_tuple_list.append((line_family_list[item],family_energy_list[item]))

    return line_dictionary

def integrate_energy_window(signal, energy_lower, energy_higher):
    """
    Integrate a HyperSpy 2D signal within a given energy window.

    Parameters
    ----------
    signal : hyperspy.signal.Signal1D or hyperspy.signal.Signal2D
        HyperSpy signal with an energy axis (e.g., EELS spectrum image).
        Typically has dimensions (ny, nx, E) where E is the energy-loss axis.
    energy_lower : float
        Lower bound of the energy window (in same units as signal.axes_manager).
    e_high : float
        Upper bound of the energy window.
    average : bool, optional
        If True, return the *mean* intensity over the window instead of the sum.

    Returns
    -------
    map2d : numpy.ndarray
        2D array (ny × nx) of integrated intensity within the specified window.

    Raises
    ------
    ValueError
        If energy window lies outside the signal’s energy range.
    """

    # --- Validate signal type ---
    if not hasattr(signal, "axes_manager"):
        raise TypeError("Input must be a HyperSpy/Exspy Signal with an axes_manager.")

    # --- Identify energy axis ---
    energy_axis = signal.axes_manager.signal_axes[0]
    energies = energy_axis.axis

    # --- Check bounds ---
    if energy_lower < energies.min() or energy_higher > energies.max():
        raise ValueError(
            f"Energy window ({energy_lower}, {energy_higher}) is outside signal range "
            f"({energies.min()}, {energies.max()})."
        )

    # --- Select slice within energy range ---
    mask = (energies >= energy_lower) & (energies <= energy_higher)
    if not np.any(mask):
        raise ValueError("No data points within specified energy window.")

    # --- Extract subset of the signal ---
    sub_signal = signal.isig[energy_lower:energy_higher]

    # --- Integrate or average ---
    data = sub_signal.data  # shape: (ny, nx, nE)
    map2d = data.sum(axis=-1)

    return map2d

def summed_spectrum_from_coords(signal, coords, average=False):
    """
    Extract and sum spectra from a list of (y, x) coordinates in a 2D HyperSpy signal.

    Parameters
    ----------
    signal : hyperspy.signal.Signal
        A 2D spectrum image, typically shape (ny, nx, E).
    coords : list of tuple(int, int)
        List of coordinates as (y, x) pairs. Index order matches signal.data.
    average : bool, optional
        If True, return the *mean* spectrum instead of the sum.

    Returns
    -------
    summed_spectrum : hyperspy.signals.Signal1D
        Summed (or averaged) spectrum as a HyperSpy Signal1D object, preserving
        the original spectral axis calibration and metadata.

    Raises
    ------
    ValueError
        If any coordinate is out of bounds or if no coordinates are given.
    """

    # --- Validate ---
    if not hasattr(signal, "axes_manager"):
        raise TypeError("Input must be a HyperSpy Signal object.")

    if not coords or len(coords) == 0:
        raise ValueError("You must provide at least one coordinate.")

    ny, nx = signal.data.shape[:2]
    for (y, x) in coords:
        if not (0 <= y < ny and 0 <= x < nx):
            raise ValueError(f"Coordinate {(y, x)} is out of bounds for shape {(ny, nx)}")

    # --- Extract individual spectra and sum ---
    spectra = [signal.inav[y, x] for (y, x) in coords]

    # Stack into a Signal1D object
    stacked = hs.stack(spectra, axis=0)  # shape: (N, E)

    # Sum or average
    summed_data = stacked.data.sum(axis=0) if not average else stacked.data.mean(axis=0)

    # --- Construct new Signal1D preserving axis calibration ---
    summed_spectrum = hs.signals.Signal1D(
        summed_data,
        axes=[signal.axes_manager.signal_axes[0].copy()]
    )

    # Copy essential metadata (so energy calibration etc. stays correct)
    summed_spectrum.metadata = signal.metadata.deepcopy()

    return summed_spectrum

def post_align_EDX_series(image_series,map_list, plot=True,model="TEMRomaTiny",host="192.168.51.3",measured_shifts=None,port=8080):

    #TODO handle dictionary of multiple maps with element labels
    #TODO test passing previous shifts in to skip inference
    #TODO add in catch so if registration fails it skips the image and does not warp it or the maps

    """
    Align a list of EDX maps based on the simultaneously acquired STEM images
     Works from the first frame using the TEMRegistration model
    'handles model scaling itself.

    Returns: (aligned_list, summed_image, shifts)
    """

    if not isinstance(image_series, list) or len(image_series) == 0:
        raise Exception("Dataset must be a non-empty list of images")

    initial_image = np.array(image_series[0], dtype=np.float64, copy=False)
    H, W = initial_image.shape
    shifts = []
    translated_list = []
    translated_map_list = []

    # Self-contained TorchServe session (same model name)
    if measured_shifts is None: #do inference
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
            raise Exception(f"{model} model cannot be contacted")
            #if scaling does not work, end the alignment
            return

    for idx in tqdm(range(len(image_series)), desc="Aligning frames", unit="img"):
        if measured_shifts is None:
            translated_image = np.array(image_series[idx], dtype=np.float64, copy=False)

            translated_map = np.array(map_list[idx], dtype=np.float64, copy=False)

            # Side-by-side concat (H, 2W)
            reg_input = np.concatenate([initial_image, translated_image], axis=1)
            registration_values = registration_model(
                reg_input,
                model,
                host=host,
                port=port,
                image_encoder='.tiff'
            )
            # Expect [dx_norm, dy_norm] in [0..1]
            dx_norm, dy_norm = registration_values[0]["translation"]
            homography = registration_values[0]["homography_fine"]

            x_pixels_shift = dx_norm * H
            y_pixels_shift = dy_norm * W
            shifts.append((x_pixels_shift, y_pixels_shift))

            M = np.float32(homography)
            warped_STEM = cv2.warpPerspective(
                translated_image.astype(np.uint16, copy=False),M,(W, H),flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            translated_list.append(warped_STEM)

            warped_map = cv2.warpPerspective(
                translated_map.astype(np.uint16, copy=False),
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        if measured_shifts:
            x_pixels_shift = measured_shifts[idx][0]
            y_pixels_shift = measured_shifts[idx][0]

            print("X",x_pixels_shift,"Y",y_pixels_shift)


            M = np.float32([[1, 0, x_pixels_shift], [0, 1, y_pixels_shift]])

            warped_STEM = cv2.warpAffine(
                translated_image.astype(np.uint16, copy=False),
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            translated_list.append(warped_STEM)

            warped_map = cv2.warpAffine(
                translated_map.astype(np.uint16, copy=False),
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        translated_list.append(warped_STEM)
        translated_map_list.append(warped_map)

    summed_map = np.sum(np.asarray(translated_map_list), axis=0, dtype=np.float64)
    summed_image = np.sum(np.asarray(translated_list), axis=0, dtype=np.float64)

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        ax1, ax2, ax3, ax4 = axs.ravel()
        ax1.imshow(initial_image, cmap="gray")
        ax1.title.set_text("Initial Image")
        plt.setp(ax1, xticks=[], yticks=[])  # removes ticks
        ax2.imshow(summed_image, cmap="gray")
        ax2.title.set_text(f"Summed Image from {len(image_series)} frames")
        plt.setp(ax2, xticks=[], yticks=[])  # removes ticks
        ax3.imshow(map_list[0], cmap="gray")
        ax3.title.set_text(f"One map frame")
        plt.setp(ax3, xticks=[], yticks=[])  # removes ticks
        ax4.imshow(summed_map, cmap="gray")
        ax4.title.set_text(f"Map from summation of {len(image_series)} frames")
        plt.setp(ax4, xticks=[], yticks=[])  # removes ticks

        plt.show()

    return [translated_list,translated_map_list, summed_map,summed_image,shifts]

def non_local_means_filter(image,patch_size=7,patch_distance=11,h=None,sigma=None,fast_mode=True,preserve_range=True):
    """
    Apply Non-Local Means (NLM) denoising to a 2D (grayscale) or 3D (color) image.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale (H, W) or 3D color (H, W, C) image.
    patch_size : int, optional
        Size of patches used for denoising (odd). Typical: 5–7.
    patch_distance : int, optional
        Maximal distance in pixels to search similar patches. Typical: 7–15.
    h : float or sequence, optional
        Filtering strength. If None, it will be set from `sigma`.
        Higher = stronger smoothing. For color, can be (h_ch1, h_ch2, ...).
    sigma : float or sequence, optional
        Estimated noise standard deviation. If None, auto-estimated.
    fast_mode : bool, optional
        If True, use the fast approximate algorithm (big speedup, similar quality).
    preserve_range : bool, optional
        If True, do not rescale image intensity to [0, 1] internally.

    Returns
    -------
    denoised : np.ndarray
        Denoised image, same shape as input, dtype float32.

    Notes
    -----
    - For best results, give a reasonable `sigma` (noise std) or let it auto-estimate.
    - `h` is typically ~0.8–1.2 * sigma (grayscale). For color, scikit-image accepts a tuple.
    """
    try:
        from skimage.restoration import denoise_nl_means, estimate_sigma
        from skimage.util import img_as_float32
    except ImportError as e:
        raise ImportError(
            "scikit-image is required for this function. Install with `pip install scikit-image`."
        ) from e

    img = img_as_float32(image)  # keeps range if preserve_range=True below

    multichannel = (img.ndim == 3 and img.shape[-1] in (3, 4))

    # Noise estimate per-channel if not provided
    if sigma is None:
        sigma = estimate_sigma(img, channel_axis=-1 if multichannel else None, average_sigmas=not multichannel)
        # scikit-image <=0.20 used `multichannel`; >=0.21 uses `channel_axis`.

    # If h not provided, set relative to sigma
    if h is None:
        # If per-channel sigma array, scale each; else single float
        if np.isscalar(sigma):
            h = 1.0 * float(sigma)
        else:
            h = 1.0 * np.asarray(sigma, dtype=np.float32)

    # Call denoise_nl_means with compatibility for skimage >=0.21 (channel_axis)
    try:
        denoised = denoise_nl_means(
            img,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
            fast_mode=fast_mode,
            preserve_range=preserve_range,
            channel_axis=-1 if multichannel else None,
        )
    except TypeError:
        # Fallback for older scikit-image that used `multichannel`
        denoised = denoise_nl_means(
            img,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
            fast_mode=fast_mode,
            preserve_range=preserve_range,
            multichannel=multichannel,
        )

    return denoised.astype(np.float32)
