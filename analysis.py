from tqdm import tqdm
import numpy as np
from time import sleep
import easygui as g
import cv2 as cv2
import fnmatch
import os
import json
import matplotlib.pyplot as plt
from grid_strategy import strategies
from matplotlib.patches import Circle
from time import time

from serving_manager.tem_models.specific_model_functions import registration_model
from serving_manager.management.torchserve_rest_manager import TorchserveRestManager

from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT
from expertpi.config import Config

from RSTEM.app_context import get_app
from RSTEM.utilities import model_has_workers,view_all_models

#config = Config()
config = Config(r"C:\Users\stem\Documents\Rob_coding\ExpertPI-0.5.1\config.yml") # path to config file if changes have been made, otherwise comment out and use default


#TODO Refactored
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

def align_image_series(image_series, plot=False, host=None, model="TEMRegistration"):
    """
    Align a list of 2D images to the first frame using the TEMRegistration model.
    Returns: (aligned_list, summed_image, shifts)
    """

    if host is None:
        host = config.inference.host  #pull from config

    if not isinstance(image_series, list) or len(image_series) == 0:
        raise Exception("Dataset must be a non-empty list of images")

    initial_image = np.array(image_series[0], dtype=np.float64, copy=False)
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
        manager.scale(model_name=model)
    except Exception:
        raise Exception(f"{model} cannot be contacted")

    acc = np.zeros((H, W), dtype=np.float64)

    for idx in tqdm(range(len(image_series)), desc="Aligning frames", unit="img"):
        moving_u16 = np.array(image_series[idx], copy=False)
        moving_f = moving_u16.astype(np.float32, copy=False)

        reg_input = np.concatenate([initial_image, moving_f], axis=1)
        registration_values = registration_model(
            reg_input,
            model,
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

        acc += warped_f.astype(np.float64, copy=False)

        warped_u16 = np.clip(warped_f, 0, 65535).astype(np.uint16, copy=False)
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
        ax4.set_title("Measured drift")
        ax4.set_xlabel("x shift [px]")
        ax4.set_ylabel("y shift [px]")
        ax4.set_aspect('equal', adjustable='datalim')
        ax4.grid(True, alpha=0.2)
        ax4.set_box_aspect(1)
        ax4.set_anchor('C')  # center the square axis in its slot
        ax4.autoscale_view()  # respect current data limits
        plt.show()

    return translated_list, summed_image, shifts