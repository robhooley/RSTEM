import os

from expert_pi import grpc_client
from expert_pi.controllers import scan_helper
from expert_pi.stream_clients import cache_client
from stem_measurements import shift_measurements
import pickle
import numpy as np
from expert_pi.grpc_client.modules.scanning import DetectorType as DT
import easygui as g
import cv2 as cv2
from tqdm import tqdm
import scipy.signal
import matplotlib.pyplot as plt
from PIL import Image
import numba
from serving_manager.api import registration_model
from stem_measurements import edx_processing

host_F4 = ""
host_P3 = "172.20.32.1" #TODO confirm
host_P2 = ""
host_global = '172.16.2.86'


host = host_global



#TODO untested
#TODO check reasonable map sizes with RAM usage

def acquire_EDX_map(frames=100,pixel_time=5e-6,fov=None,scan_rotation=0,num_pixels=1024,drift_correction_method="patches",tracking_image="BF"):
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
        grpc_client.scanning.set_field_width(fov) #convert microns to meters

    map_data = []
    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=[DT.BF, DT.HAADF, DT.EDX1, DT.EDX0])
    header, data = cache_client.get_item(scan_id, num_pixels**2)
    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)
    map_data.append((header,data,initial_shift))

    with open(f"{folder}{file_name}_0.pdat", "wb") as f:
        pickle.dump(map_data[0], f) #first item from map data list of scans

    if tracking_image == "BF" or "bf":
        tracking_signal= "BF"
    elif tracking_image == "ADF" or "HAADF" or "adf" or "haadf":
        tracking_signal = "HAADF"

    anchor_image = data["stemData"][tracking_signal].reshape(num_pixels, num_pixels)

    if drift_correction_method is not "patches" or "Patches" or "ML":
        print("Drift correction method",drift_correction_method)
        print("Drift correction is not being performed")

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
            real_shift_x = raw_shift[0]*fov  # shifts normalised between 0,1 proportion of image, convert to meters
            real_shift_y = raw_shift[1]*fov
            grpc_client.illumination.set_shift({"x": current_shift['x'] + real_shift_x, "y": current_shift['y'] + real_shift_y},grpc_client.illumination.DeflectorType.Scan)  # apply shifts in microns to existing shifts
        if drift_correction_method == "patches" or "Patches":
            shift_offset = shift_measurements.get_offset_of_pictures(anchor_image, series_image, fov, method=shift_measurements.Method.PatchesPass2)
            shift_offset = np.dot(R, shift_offset)  # rotate back
            print(f"frame {i}, X shift {current_shift['x'] - shift_offset[0]*1e9} nm, Y shift {current_shift['y'] - shift_offset[1]*1e9} nm")
            grpc_client.illumination.set_shift({"x": current_shift['x'] - shift_offset[0], "y": current_shift['y'] - shift_offset[1]}, grpc_client.illumination.DeflectorType.Scan)
        else:
            pass

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
            #print(path)
            if file.endswith(".pdat"):
                with open(path, "rb") as f:
                    header, data, s0 = pickle.load(f)
            map_data.append((header,data))
                #TODO untested

    for i in range(len(map_data)): #map data stored as tuple of (header,data,shifts)
        frame = map_data[i]
        data = frame[1]
        edx_data.append(data["edxData"]["EDX0"]["energy"])
        edx_data.append(data["edxData"]["EDX1"]["energy"])
        BFs.append(data["stemData"]["BF"].reshape(1024, 1024))

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
    channels = np.zeros((N, 1024*1024))

    i = 0
    while i < len(energies):
        probabilities = bases[:, energies[i:i + batch]]
        channels[:, pixels[i:i + batch]] += np.dot(Mi, probabilities)
        i += batch
        print(f"energies->elements {i/len(energies)*100:6.3f}%", end="\r")
    channels = channels.reshape(N, 1024, 1024)

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

    f, ax = plt.subplots(4, 5, sharex=True, sharey=True)
    i = 0
    for name in imgs_all.keys():
        img = imgs_all[name]
        ax[i%4, i//4].set_title(name)
        ax[i%4, i//4].imshow(img)
        i += 1
    plt.show()

    # saving:
    save_folder = g.diropenbox("Select folder to save maps into")
    for name, img in imgs_all.items():
        img = img.astype("float")
        img[img < 0] = 0
        im = Image.fromarray((img/np.max(img)*(2**15 - 1)).astype('uint16'))
        im.save(save_folder + '/' + str(name) + '.tif')


    """im = Image.fromarray(data["stemData"]["BF"].reshape(1024, 1024))
    im.save(save_folder + '/BF.tif')

    BF_summed = (np.sum(BFs, axis=0).reshape(1024, 1024)/total_frames).astype("uint16")
    im = Image.fromarray(BF_summed)
    im.save(save_folder + '/BF_summed.tif')

    im = Image.fromarray(data["stemData"]["HAADF"].reshape(1024, 1024))
    im.save(save_folder + '/HAADF.tif')"""

def produce_spectrum(map_data=None):
    edx_data = []
    BFs = []
    ADFs = []

    print("loading data")
    if map_data == None:
        map_data = []
        data_folder = g.diropenbox("data folder","data folder")
        file_path = os.listdir(data_folder)
        for file in tqdm(file_path):  # iterates through folder with a progress bar
            path = data_folder + "\\" + file  # picks individual images
            #print(path)
            if file.endswith(".pdat"):
                with open(path, "rb") as f:
                    header, data, s0 = pickle.load(f)
            map_data.append((header,data))

    for i in range(len(map_data)): #map data stored as tuple of (header,data,shifts)
        frame = map_data[i]
        data = frame[1]
        edx_data.append(data["edxData"]["EDX0"]["energy"])
        edx_data.append(data["edxData"]["EDX1"]["energy"])
        BFs.append(data["stemData"]["BF"].reshape(1024, 1024))

    energies = np.concatenate([e[0] for e in edx_data])
    pixels = np.concatenate([e[1] for e in edx_data])

    histogram,bin_edges = np.histogram(energies,bins=2000,range=(0,20000))
    bincenters = np.mean(np.vstack([bin_edges[0:-1], bin_edges[1:]]), axis=0)

    plt.bar(histogram,bincenters)


