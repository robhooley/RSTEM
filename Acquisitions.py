
from expert_pi import grpc_client
from expert_pi.controllers import scan_helper
from expert_pi.stream_clients import cache_client
from expert_pi.grpc_client.modules._common import DetectorType as DT
import numpy as np
from critical_dose_measurements import get_ui_dose_values
def drift_corrected_imaging(num_frames, output="Summed",shift_method="ML",num_pixels=None):

    scan_rotation=0 #TODO is this scan rotation part really needed?
    R = np.array([[np.cos(scan_rotation), np.sin(scan_rotation)],
                  [-np.sin(scan_rotation), np.cos(scan_rotation)]])
    grpc_client.scanning.set_rotation(scan_rotation)


    pixel_time = window.scanning.pixel_time_spin.value()/1e3  # gets in us and convert to ns
    print("Pixel time",pixel_time,"ns")
    fov = grpc_client.scanning.get_field_width() #in meters
    fov = fov*1e-6
    print("Field of View in um",fov)

    #TODO add in option to calulcate series dose?
    dose_values_pixel,dose_values_probe  = get_ui_dose_values()

    if num_pixels == None:
        num_pixels = 1024

    BF_images_list = []
    ADF_images_list = []
    image_offsets = []

    initial_scan = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                               detectors=[DT.BF, DT.HAADF])
    header, data = cache_client.get_item(initial_scan, num_pixels ** 2) #retrive small measurements in one chunk

    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

    initial_BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
    BF_images_list.append(initial_BF_image)
    initial_ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
    ADF_images_list.append(initial_ADF_image)
    image_offsets.append(initial_shift)

    #TODO check other shift measurement methods and add them properly
    if shift_method == "ML":
        shift_method = shift_measurements.Method.PatchesPass2
    elif shift_method =="patches":
        shift_method = shift_measurements.Method.PatchesPass2

    for frame in range(len(num_frames)):
        print("Acquiring frame",frame,"of",num_frames)
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=[DT.BF, DT.HAADF, DT.EDX1, DT.EDX0])

        header, data = cache_client.get_item(scan_id, num_pixels ** 2) #retrive image from cache
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels) #reshape BF image
        BF_images_list.append(BF_image) #add to list
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels) #reshape ADF image
        ADF_images_list.append(ADF_image) #add to list

        s = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan) #get current beam shifts

        # apply drift correction between images:
        shift = shift_measurements.get_offset_of_pictures(initial_BF_image, BF_image, fov, method=shift_method) #measure offset of images
        shift = np.dot(R, shift)  # rotate shifts back to scanning axes
        #print(i, (s0['x'] - s['x']) * 1e9, (s0['y'] - s['y']) * 1e9)
        grpc_client.illumination.set_shift(
                {"x": s['x'] - shift[0] * 1e-6, "y": s['y'] - shift[1] * 1e-6},grpc_client.illumination.DeflectorType.Scan) #apply shifts in microns to existing shits





    BF_summed = sum(BF_images_list)
    ADF_summed = sum(ADF_images_list)

    if output == "Summed" or None or False:
        return BF_summed, ADF_summed

    if output == "All" or True:
        return BF_images_list, ADF_images_list
