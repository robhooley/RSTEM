import os

#from expert_pi import grpc_client
#from expert_pi.controllers import scan_helper
#from expert_pi.stream_clients import cache_client
from stem_measurements import shift_measurements
import pickle
from time import sleep
import numpy as np
#from expert_pi.grpc_client.modules.scanning import DetectorType as DT
import easygui as g
import cv2 as cv2
from tqdm import tqdm
import scipy.signal
import matplotlib.pyplot as plt
from grid_strategy import strategies
from PIL import Image
from serving_manager.api import registration_model
from stem_measurements import edx_processing

host_F4 = ""
host_P3 = "172.20.32.1" #TODO confirm
host_P2 = ""
host_global = '172.16.2.86'


host = host_global



#TODO untested
#TODO check reasonable map sizes with RAM usage

def acquire_EDX_map(frames=100,pixel_time=5e-6,fov=None,scan_rotation=0,num_pixels=1024,drift_correction_method="patches"):
    """Parameters
    frames: number of scans
    pixel_time: in seconds
    fov: in microns
    scan_rotation in degrees
    num_pixels: scan dimensions
    drift_correction_method: either "patches" for openCV template matching, "ML" uses trained AI drift correction"""
    folder = g.diropenbox("Select folder to save mapping layers into")
    file_name = "\\EDX_map_frame"
    print("Predicted measurement time:", pixel_time*num_pixels**2*frames/60, "min")

    R = np.array([[np.cos(scan_rotation), np.sin(scan_rotation)],
                 [-np.sin(scan_rotation), np.cos(scan_rotation)]])

    grpc_client.scanning.set_rotation(np.deg2rad(scan_rotation))
    if fov is None:
        fov=grpc_client.scanning.get_field_width() #in meters
    if fov is not None:
        grpc_client.scanning.set_field_width(fov) #in meters

    if grpc_client.stem_detector.get_is_inserted(DT.BF) is True and grpc_client.stem_detector.get_is_inserted(
                DT.HAADF) is False:
        tracking_signal = "BF"
    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is False:
        tracking_signal = "BF"
        grpc_client.projection.set_is_off_axis_stem_enabled(True) #use off axis BF for tracking if both detectors are out
    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is True:
        tracking_signal = "HAADF"
    if grpc_client.stem_detector.get_is_inserted(DT.BF) is True and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is True:
        tracking_signal = "HAADF"
    print(f"Image tracking using {tracking_signal} images")

    map_data = []
    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=[DT.BF, DT.HAADF, DT.EDX1, DT.EDX0])
    header, data = cache_client.get_item(scan_id, num_pixels**2)
    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)
    map_data.append((header,data,initial_shift))

    with open(f"{folder}{file_name}_0.pdat", "wb") as f:
        pickle.dump(map_data[0], f) #first item from map data list of scans


    image_list = []
    anchor_image = data["stemData"][tracking_signal].reshape(num_pixels, num_pixels)
    image_list.append(anchor_image)

    if drift_correction_method == "ML":
        drift_correction_method_named = "TESCAN's machine learning"
    if drift_correction_method == "patches" or "Patches":
        drift_correction_method_named = "OpenCV sub-image template matching"

    if drift_correction_method != "patches" or "Patches" or "ML":
        print("Drift correction is not being performed")
    print(f"Drift correction is using {drift_correction_method_named} for image registration")



    for i in range(1, frames):
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=[DT.BF, DT.HAADF, DT.EDX1, DT.EDX0])
        header, data = cache_client.get_item(scan_id, num_pixels**2)
        current_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

        with open(f"{folder}{file_name}_{i}.pdat", "wb") as f:
            pickle.dump((header, data, current_shift), f)

        # apply drift correction between images:
        series_image = data["stemData"][tracking_signal].reshape(num_pixels, num_pixels)
        #TODO tidy up the shift offset inconsistencies
        if drift_correction_method == "ML":
            registration = registration_model(np.concatenate([anchor_image, series_image], axis=1),
                                              'TEMRegistration', host=host, port='7443',
                                              image_encoder='.tiff')  # measure offset of images # TODO corrects to first image, should it correct to previous image?
            raw_shift = registration[0]["translation"]
            #real_shift_x = raw_shift[0]*fov  # shifts normalised between 0,1 proportion of image, convert to meters
            #real_shift_y = raw_shift[1]*fov
            shift_offset = (raw_shift[0]*fov,raw_shift[1]*fov)
            grpc_client.illumination.set_shift({"x": current_shift['x'] + shift_offset[0], "y": current_shift['y'] + shift_offset[1]},grpc_client.illumination.DeflectorType.Scan)  # apply shifts in microns to existing shifts #TODO check if this should be - shift or + shift
            print(f"frame {i}, X shift {current_shift['x'] + shift_offset[0]*1e9} nm, Y shift {current_shift['y'] + shift_offset[1]*1e9} nm")
        if drift_correction_method == "patches" or "Patches":
            shift_offset = shift_measurements.get_offset_of_pictures(anchor_image, series_image, fov, method=shift_measurements.Method.PatchesPass2) # TODO corrects to first image, should it correct to previous image?
            shift_offset = np.dot(R, shift_offset)  # rotate back
            print(f"frame {i}, X shift {current_shift['x'] + shift_offset[0]*1e9} nm, Y shift {current_shift['y'] + shift_offset[1]*1e9} nm")
            grpc_client.illumination.set_shift({"x": current_shift['x'] + shift_offset[0], "y": current_shift['y'] + shift_offset[1]}, grpc_client.illumination.DeflectorType.Scan) #TODO check if it should be - or + shifts
        else:
            pass
        image_list.append(series_image)
        map_data.append((header,data,current_shift))


    return map_data


def sketchy_map_processing(map_data=None,elements=[""]):
    edx_data = []
    BFs = []
    ADFs = []

    #TODO handle loading data from folder
    print("loading data")
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
                #TODO untested
    scan_pixels = map_data[0][0]["scanDimensions"]
    num_frames = len(map_data)

    for i in range(len(map_data)): #map data stored as tuple of (header,data,shifts)
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
    lines = edx_processing.get_edx_filters(elements)
    E = np.array(range(0, 2**16))  # max energy 2**16 kV
    bases = []
    for name, value in lines.items():
        bases.append(generate_base_function(value, E))
    N = len(bases)
    bases = np.array(bases)

    # generate basis matrix:

    M = []
    for i in range(N):
        M.append([])
        for j in range(N):
            M[i].append(np.sum(bases[i]*bases[j]))
    Mi = np.linalg.inv(M)

    # now we remap the energy events to the probabilities of the element (it is same as linear fitting):
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
        imgs_all[list(lines.keys())[i]] = channels_filtered[i, :, :]

    # plot them

    num_maps = len(imgs_all)

    imgs_list = []
    names_list = []
    for name in imgs_all.keys():
        img = imgs_all[name]
        imgs_list.append(img)
        names_list.append(name)

        #ax[i%4, i//4].set_title(name)
        #ax[i%4, i//4].imshow(img)


    specs = strategies.SquareStrategy("center").get_grid(num_maps)

    for i,subplot in enumerate(specs):
        plt.subplot(subplot)
        ax = plt.gca()
        image = imgs_list[i]
        name = names_list[i]
        ax.imshow(image)
        ax.set_title(name)
    plt.show()

    """    f, ax = plt.subplots(4, 5, sharex=True, sharey=True)
    i = 0
    
        i += 1"""


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



def produce_spectrum(map_data=None,elements=None):
    edx_data = []

    print("loading data")
    if map_data == None:
        map_data = []
        data_folder = g.diropenbox("data folder","data folder")
        file_path = os.listdir(data_folder)
        for file in tqdm(file_path):  # iterates through folder with a progress bar
            path = data_folder + "\\" + file  # picks individual images
            print(path)
            if file.endswith(".pdat"):
                with open(path, "rb") as f:
                    header, data, s0 = pickle.load(f)
                    map_data.append((header,data))

    for i in range(len(map_data)): #map data stored as tuple of (header,data,shifts)
        frame = map_data[i]
        data = frame[1]
        edx_data.append(data["edxData"]["EDX0"]["energy"])
        edx_data.append(data["edxData"]["EDX1"]["energy"])

    energies = np.concatenate([e[0] for e in edx_data])
    #pixels = np.concatenate([e[1] for e in edx_data])



    histogram,bin_edges = np.histogram(energies,bins=4000,range=(0,20000))
    plt.stairs(histogram,bin_edges)

    plt.show()

produce_spectrum()
