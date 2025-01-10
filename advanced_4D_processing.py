from time import sleep

import numpy as np
from tqdm import tqdm

from expert_pi import grpc_client
from expert_pi.app import app
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT, RoiMode as RM
from expert_pi.gui import main_window

window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client

from expert_pi.RSTEM.utilities import check_memory,collect_metadata

#TODO needs STEM server 0.10

def crop_images_to_fixed_square(image_rows, square_size):
    """
    Crop a nested list of images (as NumPy arrays) to a fixed square size, centered around the middle
    of each image, processing the entire list at once.

    :param image_rows: Nested list of NumPy array images (rows of images)
    :param square_size: The fixed size of the square to crop (int)
    :return: Nested list of cropped NumPy array images (rows of cropped images)
    """
    cropped_rows = []  # List to store cropped rows

    while image_rows:
        # Remove the first row from the list
        row = image_rows.pop(0)

        # Assume all images in the row have the same dimensions (use the first image)
        height, width = row[0].shape[:2]

        # Validate the square size once for the row
        if square_size > height or square_size > width:
            raise ValueError(
                f"Square size {square_size} is too large for images with dimensions ({height}, {width})"
            )

        # Calculate the top-left corner once for the row
        top = (height - square_size) // 2
        left = (width - square_size) // 2

        # Process the row
        cropped_row = []
        for img in row:
            # Crop the image using the precomputed coordinates
            cropped_img = img[top:top + square_size, left:left + square_size]
            cropped_row.append(cropped_img)

        # Add the processed row to the cropped_rows list
        cropped_rows.append(cropped_row)

    return cropped_rows


def scan_4D_fast(scan_width_px=128,camera_frequency_hz=18000,use_precession=False,roi_mode=128):
    """Parameters
    scan width: pixels
    camera_frequency: camera speed in frames per second up to 72000
    use_precession: True or False
    roi_mode: optional variable to enable ROI mode, either 128,256 or False
    returns a tuple of (image_array, metadata)
    """

    sufficient_RAM = check_memory(camera_frequency_hz,scan_width_px,roi_mode)
    if sufficient_RAM == False:
        print("This dataset will probably not fit into RAM, trying anyway but expect a crash")
     #gets the microscope and acquisition metadata
    if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(DT.HAADF) == True: #if either STEM detector is inserted
        grpc_client.stem_detector.set_is_inserted(DT.BF,False) #retract BF detector
        grpc_client.stem_detector.set_is_inserted(DT.HAADF, False) #retract ADF detector
        for i in tqdm(range(5),desc="stabilising after detector retraction",unit=""):
            sleep(1) #wait for 5 seconds
    grpc_client.projection.set_is_off_axis_stem_enabled(False) #puts the beam back on the camera if in off-axis mode
    sleep(0.2)  # stabilisation after deflector change
    metadata = collect_metadata(acquisition_type="Camera",scan_width_px = scan_width_px, use_precession= use_precession, pixel_time = 1/camera_frequency_hz,scan_rotation= 0,camera_pixels = roi_mode)
    #sets to ROI mode
    if roi_mode==128: #512x128 px
        grpc_client.scanning.set_camera_roi(roi_mode=RM.Lines_128, use16bit=False)
        camera_shape=(128,512)
    elif roi_mode==256: #512x256 px
        grpc_client.scanning.set_camera_roi(roi_mode=RM.Lines_256,use16bit=False)
        camera_shape=(256,512)
    else:
        grpc_client.scanning.set_camera_roi(roi_mode=RM.Disabled,use16bit=True)

    scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/camera_frequency_hz, 8), total_size=scan_width_px, frames=1, detectors=[DT.Camera], is_precession_enabled=use_precession)
    print("Acquiring",scan_width_px,"x",scan_width_px,"px dataset at",camera_frequency_hz,"frames per second")
    image_list = [] #empty list to take diffraction data
    for i in tqdm(range(scan_width_px),desc="Retrieving data from cache",total=scan_width_px,unit="rows"): #retrives data one scan row at a time to avoid crashes
        header, data = cache_client.get_item(scan_id, scan_width_px)  # cache retrieval in rows
        camera_size = camera_shape#data["cameraData"].shape[1],data["cameraData"].shape[2] #gets shape of diffraction patterns
        image_data = data["cameraData"] #take the data for that row
        image_row = np.reshape(image_data,(scan_width_px,camera_size[0],camera_size[1])) #reshapes data to an individual image #TODO necessary?
        image_list.append(image_row) #adds it to the list of images

    if roi_mode==128 or 256:
        print("Cropping images to square")
        image_list = crop_images_to_fixed_square(image_list,camera_shape[0])

    image_array = np.asarray(image_list) #converts the image list to an array
    del image_list #flush image list to clear out RAM
    return (image_array,metadata) #tuple with image data and metadata