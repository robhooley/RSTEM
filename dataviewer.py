#%%threaded
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Circle
from matplotlib.patches import Annulus
from matplotlib import patches, gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import easygui as g
import cv2 as cv2
import time
from time import sleep
from statistics import mean
from tqdm import tqdm
from skimage.transform import radon
from skimage import exposure
from pathlib import Path
from math import ceil
from matplotlib.path import Path as matpath
import matplotlib.colors as mcolors

from matplotlib.widgets import Button, Slider

#TODO refactor background image to navigation image
#TODO clean up imports
#TODO consider having acquisition module and separate analysis module

dataset = np.load("C:\\Users\\robert.hooley\\Desktop\\HR_4D_STEM.npy")

def dataviewer_4D(dataset,background_image=None):
    def spot_marker(xposition, yposition): #sets the spot position in the virtual image
        ax[0].scatter(int(yposition.val), int(xposition.val),marker="+",c="red") #adds a red cross at user selected XY

        return

    def build_VBF(dataset): #calculates a rough virtual bright field image to use as a navigation image
        intensity_list = [] #empty list to hold intensity values
        for scan_row in dataset:
            for pixel in scan_row:
                intensity = np.sum(pixel) #sums the entire image
                intensity_list.append(intensity) #adds it to the list

        intensity_array = np.asarray(intensity_list) #changes from list to array
        navigation_image = np.reshape(intensity_array, (dataset_shape[0], dataset_shape[1])) #reshapes the array to match dataset dimensions

        return navigation_image #gives out the navgation image

    def set_axes():
        ax[0].set_aspect(1) #sets square aspect ratio for the plot
        ax[1].set_aspect(1) #sets square aspect ratio for the plot
        ax[0].set_xlim(0, dataset_pixels[1]) #axes limited to size of dataset with no excess
        ax[0].set_ylim(0, dataset_pixels[0]) #axes limited to size of dataset with no excess
        ax[1].set_xlim(0, camera_pixels[0]) #axes size limits to number of pixels in camera
        ax[1].set_ylim(0, camera_pixels[1]) # axes size limits to number of pixels in camera
        ax[0].imshow(background_image,cmap="gray") #sets the colourmap to grayscale

    # The function to be called anytime a slider's value changes
    def update(val):
        ax[0].clear() #clears the old data from the navigation plot
        ax[1].clear() #clears the old data from the diffraction plot
        set_axes() #rebuilds the axes
        ax[1].imshow(dataset[yposition.val][xposition.val],cmap="gray") #adds the new diffraction pattern
        ax[1].set_title(("Pattern from", int(xposition.val), int(yposition.val))) #Adds title to diffraction plot
        ax[0].set_title("Navigation image") #adds title to navigation image
        spot_marker(yposition,xposition) #adds the new spot marker in the navigation image
        fig.canvas.draw_idle() #stops the interactive plotting

    def reset(val):
        xposition.reset() #if the reset command is sent, send the sliders back to the start point
        yposition.reset() #if the reset command is sent, send the sliders back to the start point

    def save(val): #when the save button is clicked adds the DP and position to a list for saving in one go
        print("Adding to save list")
        dp_save_list.append(dataset[yposition.val][xposition.val]) #puts the diffraction pattern in a list
        position_save_list.append((yposition.val,xposition.val)) #adds the xy position of the DP to a list
        VBF_save_list.append(background_image) #saves the navigation image to a list #TODO can be removed

    def save_list(dp_list,positions,VBF_list): #this function handles the saving of all patterns #TODO clean up refactor
        directory = g.diropenbox("Save directory","Save directory") #prompts users to select a folder to save into
        print("Saving data")
        for i in range(len(dp_list)):
            diffraction = dp_list[i] #gets the diffraction pattern
            VBF = VBF_list[i] #gets the navigation image
            position = positions[i] #gets the XY position tuple
            fig,ax = plt.subplots(1) #builds a plot to annotate the navigation image
            plt.tight_layout() #reduces white space
            marker = plt.Circle((position[1], position[0]), radius=1,color="red") #adds a position marker at the selected XY
            ax.imshow(VBF,cmap="gray") #adds the navigation image to the plot
            plt.setp(ax, xticks=[], yticks=[]) #removes tick markers
            ax.add_patch(marker) #adds the postion marker to the navigaton image
            """handles the saving of the plot""" #TODO see if the code below works
            #time_start = time.monotonic()
            #canvas = FigureCanvasAgg(fig)
            #fig.canvas.draw()
            #annotated = canvas.buffer_rgba()
            #annotated_VBF = np.asarray(annotated)
            #annotated_VBF = cv2.flip(annotated_VBF,0) #flips the image to deal with some openCV related stuff #TODO check after refactor
            #plt.close() #closes the plot as it's no longer needed
            filename_VBF = directory+f"\\VBF_x_{position[1]}_y_{position[0]}.png" #sets the filename for the navigation image
            #cv2.imwrite(filename_VBF,annotated_VBF) #saves the navigation image


            #filename_diffraction = directory+f"\\diffraction_x_{position[1]}_y_{position[0]}.TIFF" #filename for DP
            #cv2.imwrite(filename_diffraction, diffraction) #saves the DP
            """This does work, but needs to be flipped vertically again...""" #TODO maybe just revert?
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
            plt.margins(0,0)
            ax.invert_yaxis()
            plt.savefig(filename_VBF,bbox_inches='tight',pad_inches = 0)

            end_time_savefig = time.monotonic()
            #savefig_time = end_time_savefig - time_start

            #print("time for savefig processing without diffraction", savefig_time)


    dataset_shape = dataset.shape #gets the shape of the incoming dataset array
    dataset_pixels = dataset_shape[0],dataset_shape[1] #works out scanning pixels
    camera_pixels = (dataset_shape[2],dataset_shape[3]) #works out camera resolution

    if background_image is None: #if navigation image is not provided, create one
        background_image = build_VBF(dataset) #builds a quick virtual bright field image

    if background_image.shape[0] != dataset_pixels[0] or background_image.shape[1] != dataset_pixels[1]: #if image shape is not equal to dataset shape #TODO test more
        print("Provided navigation image shape is",background_image.shape)
        print("Dataset dimensions are",dataset_pixels)
        print("Navigation image shape does not match dataset shape")
        print("Creating new navigation image")
        background_image = build_VBF(dataset)

    # Define initial plotting space
    fig, ax = plt.subplots(1,2,figsize=(20,10)) #builds a figure with 2 subplots
    set_axes() # sets the axis scales
    ax[0].scatter(0,0,marker="+",c="red") #sets an initial position marker
    ax[1].imshow(dataset[0][0],cmap="gray") #shows the pattern from 0,0 in the DP array
    plt.setp(ax[1], xticks=[], yticks=[]) #removes the tick marks from the diffraction pattern display
    ax[1].set_title("Pattern from 0,0")  # Adds title to diffraction plot
    ax[0].set_title("Navigation image")  # adds title to navigation image
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the position of the pattern in the x axis.
    xpos_allowed = np.arange(start=0,stop=dataset_pixels[1],step=1) #slider range is capped to integer number of pixels
    xpos = fig.add_axes([0.1, 0.1, 0.5, 0.03]) #size of slider in plot
    xposition = Slider(ax=xpos,label='X position',valmin=0,valstep=xpos_allowed,valmax=xpos_allowed[-1],valinit=0) #creates the slider

    # Make a vertically oriented slider to control the position of the pattern in the y axis
    ypos_allowed = np.arange(start=0,stop=dataset_pixels[0],step=1) #slider range is capped to integer number of pixels
    ypos = fig.add_axes([0.1, 0.25, 0.0225, 0.63]) #size of slider in plot
    yposition = Slider(ax=ypos,label="Y position",valmin=0,valmax=ypos_allowed[-1],valstep=ypos_allowed,valinit=0,orientation="vertical") #creates the slider

    # register the update function with each slider
    xposition.on_changed(update) #tracks the sliders to see if they have changed, if so, runs the update function
    yposition.on_changed(update) #tracks the sliders to see if they have changed, if so, runs the update function

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    reset_axes = fig.add_axes([0.8, 0.025, 0.1, 0.04]) #adds a spot for the reset button
    resetbutton = Button(reset_axes, 'Reset', hovercolor='0.975') #adds the reset button
    save_image = fig.add_axes([0.2, 0.025, 0.1, 0.04]) #adds a spot for the save button
    savebutton = Button(save_image, 'Save', hovercolor='0.975') #adds the save button

    resetbutton.on_clicked(reset) #triggers the reset function if the button is clicked
    plt.show(block=False)

    dp_save_list =[] #list for diffraction patterns
    position_save_list = [] #list for xy positions
    VBF_save_list = [] #list for navigation images #TODO rename and remove?
    savebutton.on_clicked(save)

    plt.show()
    if len(dp_save_list) != 0: #if the save button has been pressed
        save_list(dp_save_list,position_save_list,VBF_save_list) #save the data to disk


def virtual_ADF(image_array,camera_size=None):

    def put_diffraction(diffraction):
        ax[0].imshow(diffraction,cmap="gray")

    def update(val):
        ax[0].clear()
        put_diffraction(diffraction)
        outer_circle = Circle(xy=(256,256),radius=outer_mask_value.val,color="red",fill=None)
        inner_circle = Circle(xy=(256,256),radius=inner_mask_value.val,color="red",fill=None)
        fill = Annulus(xy=(256,256),r=outer_mask_value.val,width=outer_mask_value.val-inner_mask_value.val,color="red",alpha=0.2)
        ax[0].add_patch(outer_circle)
        ax[0].add_patch(inner_circle)
        ax[0].add_patch(fill)
        plt.setp(ax[0], xticks=[], yticks=[])  # removes the tick marks from the diffraction pattern display
        ax[1].set_title("Virtual ADF")  # Adds title to diffraction plot
        ax[0].set_title("Diffraction")  # adds title to navigation image
        ax[0].invert_yaxis()
        fig.canvas.draw_idle() #stops the interactive plotting

    def set_axes(generated_image):
        ax[0].set_aspect(1) #sets square aspect ratio for the plot
        ax[1].set_aspect(1) #sets square aspect ratio for the plot
        ax[1].set_xlim(0, dataset_pixels[1]) #axes limited to size of dataset with no excess
        ax[1].set_ylim(0, dataset_pixels[0]) #axes limited to size of dataset with no excess
        ax[0].set_xlim(0, camera_pixels[0]) #axes size limits to number of pixels in camera
        ax[0].set_ylim(0, camera_pixels[1]) # axes size limits to number of pixels in camera
        ax[1].imshow(generated_image,cmap="gray") #sets the colourmap to grayscale

    def generate_image(inner_radius,outer_radius):
        print("inner",inner_radius)
        print("outer",outer_radius)
        inner_mask = create_circular_mask(dataset_shape[2], dataset_shape[3],center=None,radius=inner_radius)
        outer_mask = create_circular_mask(dataset_shape[2], dataset_shape[3],center=None,radius=outer_radius)
        VDF_intensity_list = []
        for scan_row in image_array:
            for pixel in scan_row:
                inner_intensity = np.sum(pixel[inner_mask])
                outer_intensity = np.sum(pixel[outer_mask])
                VDF_intensity = outer_intensity-inner_intensity
                VDF_intensity_list.append(VDF_intensity)
        VDF_array = np.asarray(VDF_intensity_list)
        generated_image = np.reshape(VDF_array,(dataset_shape[0],dataset_shape[1]))
        return generated_image

    def create_circular_mask(h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    def generating(val):
        inner_radius = inner_mask_value.val
        outer_radius = outer_mask_value.val
        generated_image = generate_image(inner_radius,outer_radius)
        ax[1].clear()
        ax[1].set_title("Virtual ADF")
        ax[1].set_aspect(1)  # sets square aspect ratio for the plot
        ax[1].set_xlim(0, dataset_pixels[1])  # axes limited to size of dataset with no excess
        ax[1].set_ylim(0, dataset_pixels[0])  # axes limited to size of dataset with no excess
        plt.setp(ax[1], xticks=[], yticks=[])  # removes the tick marks from the VDF
        plt.setp(ax[0], xticks=[], yticks=[])  # removes the tick marks from the VDF
        ax[1].imshow(generated_image, cmap="gray")
        fig.canvas.draw_idle()  # stops the interactive plotting
        return generated_image

    def save(val): #when the save button is clicked adds the DP and position to a list for saving in one go
        radii.append((inner_mask_value.val,outer_mask_value.val))
        generated_image = generate_image(inner_mask_value.val,outer_mask_value.val) #recalculates the VDF
        VDF_list.append(generated_image) #saves the VDF image to a list
        print("Adding image to save list")

    def save_list(radii,VDF_list): #TODO fix this and tidy up
        directory = g.diropenbox("Save directory","Save directory") #prompts users to select a folder to save into
        print("Saving data")
        for i in range(len(VDF_list)):
            VDF = VDF_list[i] #gets the generated image
            radiuses = radii[i] #gets the annulus masks
            inner_radius = radiuses[0]
            outer_radius = radiuses[1]
            fig,ax = plt.subplots(1) #builds a plot to annotate the navigation image
            plt.tight_layout() #reduces white space
            inner = plt.Circle((256,256), radius=inner_radius,color="red",fill=None) #adds a position marker at the selected XY
            ax.add_patch(inner)
            outer = plt.Circle((256,256), radius=outer_radius,color="red",fill=None) #adds a position marker at the selected XY
            fill = Annulus(xy=256,r=outer_radius,width=outer_radius-inner_radius,color="red",alpha=0.2)
            ax.add_patch(fill)
            ax.add_patch(outer)

            ax.imshow(diffraction,cmap="gray") #adds the diffraction pattern to the plot
            plt.setp(ax, xticks=[], yticks=[]) #removes tick markers

            """handles the saving of the plot"""
            filename_VDF = directory+f"\\VDF_{inner_radius}to{outer_radius}.png" #sets the filename for the navigation image

            """This does work, but needs to be flipped vertically again...""" #TODO maybe just revert?
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
            plt.margins(0,0)
            #ax.invert_yaxis()
            plt.savefig(filename_VDF,bbox_inches='tight',pad_inches = 0)
            cv2.imwrite(filename_VDF,VDF.astype(np.uint16))

    max_angle = math.hypot(256,256)
    min_angle = 0


    dataset_shape = image_array.shape  # gets the shape of the incoming dataset array
    dataset_pixels = dataset_shape[0], dataset_shape[1]  # works out scanning pixels
    camera_pixels = (dataset_shape[2], dataset_shape[3])  # works out camera resolution

    # Define initial plotting space
    fig, ax = plt.subplots(1,2,figsize=(20,10)) #builds a figure with 2 subplots
    fig.suptitle("Use the sliders to set VDF mask, Use generate button to make images"
                 ", click the save button to add images to prepare images for saving"
                 ", close the display to save all images")
    radii = []
    VDF_list = []

    initial_inner = 20
    initial_outer = 100

    initial_image = generate_image(initial_inner,initial_outer)
    set_axes(generated_image=initial_image) # sets the axis scales
    
    ax[0].imshow(image_array[0][0],cmap="gray") #shows the pattern from 0,0 in the DP array

    ax[1].set_title("Virtual ADF")  # Adds title to diffraction plot
    ax[0].set_title("Diffraction")  # adds title to navigation image
    ax[1].imshow(initial_image,cmap="gray")  # produces an image based off the initial slider values
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    ax[0].set_aspect(1)  # sets square aspect ratio for the plot
    ax[1].set_aspect(1)  # sets square aspect ratio for the plot
    ax[1].set_xlim(0, dataset_pixels[1])  # axes limited to size of dataset with no excess
    ax[1].set_ylim(0, dataset_pixels[0])  # axes limited to size of dataset with no excess
    ax[0].set_xlim(0, camera_pixels[0])  # axes size limits to number of pixels in camera
    ax[0].set_ylim(0, camera_pixels[1])  # axes size limits to number of pixels in camera
    plt.setp(ax[0], xticks=[], yticks=[])  # removes the tick marks from the diffraction display
    plt.setp(ax[1], xticks=[], yticks=[])  # removes the tick marks from the VDF display




    # Make a horizontal slider to control the inner angle slider.

    inner_mask_slider = fig.add_axes([0.1, 0.1, 0.5, 0.03]) #size of slider in plot
    inner_mask_value = Slider(ax=inner_mask_slider,label='Inner angle',valmin=0,valmax=max_angle-1,valstep=1,valinit=20) #creates the slider

    # Make a vertically oriented slider to control the outer angle slider

    outer_mask_slider = fig.add_axes([0.1, 0.25, 0.0225, 0.63]) #size of slider in plot
    outer_mask_value = Slider(ax=outer_mask_slider,label="Outer angle",valmin=min_angle+1,valmax=max_angle,valstep=1,valinit=100,orientation="vertical") #creates the slider

    """Temporary"""
    diffraction = image_array[0][0]
    # register the update function with each slider
    inner_mask_value.on_changed(update)  # tracks the sliders to see if they have changed, runs the update function
    outer_mask_value.on_changed(update)  # tracks the sliders to see if they have changed, runs the update function

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    generate_images_axes_object = fig.add_axes([0.8, 0.025, 0.1, 0.04])  # adds a spot for the reset button
    generatebutton = Button(generate_images_axes_object, 'Generate', hovercolor='0.975')  # adds the reset button
    save_image = fig.add_axes([0.2, 0.025, 0.1, 0.04])  # adds a spot for the save button
    savebutton = Button(save_image, 'Save', hovercolor='0.975')  # adds the save button

    generatebutton.on_clicked(generating)
    savebutton.on_clicked(save)
    update(val=False) #runs the update function

    plt.show()

    if len(VDF_list) != 0: #if the save button has been pressed
        save_list(radii,VDF_list) #save the data to disk

virtual_ADF(dataset)