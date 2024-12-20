import numpy as np
import math


def create_circular_mask(image_height, image_width, mask_center_coordinates=None, mask_radius=None):
    if mask_center_coordinates is None:  # use the middle of the image
        mask_center_coordinates = (int(image_width/2), int(image_height/2))
    if mask_radius is None:  # use the smallest distance between the center and image walls
        mask_radius = min(mask_center_coordinates[0], mask_center_coordinates[1], image_width - mask_center_coordinates[0], image_height - mask_center_coordinates[1])
    Y, X = np.ogrid[:image_height, :image_width]
    dist_from_center = np.sqrt((X - mask_center_coordinates[0])**2 + (Y - mask_center_coordinates[1])**2)
    mask = dist_from_center <= mask_radius
    return mask

def VBF(image_array,radius,center):
    camera_data_shape = image_array[0][0].shape  # shape of first image to get image dimensions
    dataset_shape = image_array.shape[0], image_array.shape[1]  # scanned region shape
    VBF_intensity_list = []  # empty list to take virtual bright field image sigals
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1], mask_center_coordinates=center,mask_radius=radius)
    for row in image_array:  # iterates through array rows
        for pixel in row:  # in each row iterates through pixels
            VBF_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
            VBF_intensity_list.append(VBF_intensity)  # adds to the list

    VBF_intensity_array = np.asarray(VBF_intensity_list)  # converts list to array
    VBF_intensity_array = np.reshape(VBF_intensity_array, (
        dataset_shape[0], dataset_shape[1]))  # reshapes array to match image dimensions
    return VBF_intensity_array


def VADF(image_array,radius,center):
    camera_data_shape = image_array[0][0].shape  # shape of first image to get image dimensions
    dataset_shape = image_array.shape[0], image_array.shape[1]  # scanned region shape
    #radius = 30  # pixels for rough VBF image construction
    ADF_intensity_list = []  # empty list to take virtual bright field image sigals
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1],mask_center_coordinates=center, mask_radius=radius)
    for row in image_array:  # iterates through array rows
        for pixel in row:  # in each row iterates through pixels
            BF_intensity = np.sum(pixel[integration_mask])
            total_intensity = np.sum(pixel)
            ADF_intensity = total_intensity - BF_intensity  # measures the intensity in the masked image
            ADF_intensity_list.append(ADF_intensity)  # adds to the list

    VADF_intensity_array = np.asarray(ADF_intensity_list)  # converts list to array
    VADF_intensity_array = np.reshape(VADF_intensity_array, (
        dataset_shape[0], dataset_shape[1]))  # reshapes array to match image dimensions
    return VADF_intensity_array


def VDF(image_array,radius,center):
    camera_data_shape = image_array[0][0].shape  # shape of first image to get image dimensions
    dataset_shape = image_array.shape[0], image_array.shape[1]  # scanned region shape
    DF_intensity_list = []  # empty list to take virtual bright field image sigals
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1],mask_center_coordinates=center ,mask_radius=radius)
    for row in image_array:  # iterates through array rows
        for pixel in row:  # in each row iterates through pixels
            DF_intensity = np.sum(pixel[integration_mask])
            DF_intensity_list.append(DF_intensity)  # adds to the list

    DF_intensity_array = np.asarray(DF_intensity_list)  # converts list to array
    DF_intensity_array = np.reshape(DF_intensity_array, (
        dataset_shape[0], dataset_shape[1]))  # reshapes array to match image dimensions
    return DF_intensity_array