import pickle as p
import numpy as np
import easygui as g
from expert_pi.RSTEM.bda_functions import beam_size_matched_acquisition
from expert_pi.RSTEM.utilities import get_microscope_parameters, calculate_dose


#TODO V1.0 offline analysis


def dose_series(max_dose=False,num_steps=False):
    """This should do a calulation of the current hardware dose and run a
    time resolved 4D-STEM acquisition that sums the data to a single pattern"""

    dwell_time = 1e-3 #s
    current_dose_values = calculate_dose()
    current_dose_rate = current_dose_values["Probe dose rate-A-2s-1"]
    print("Current dose rate",current_dose_rate,"e-A-2s-1")
    dose_steps_needed = int((max_dose/current_dose_rate)*1.1) #additional 10% buffer
    if max_dose==False:
        steps_needed=num_steps
    else :
        steps_needed=dose_steps_needed #either uses dose target or number of steps
    dataset_list = []
    dose_list = []
    dose_offset = current_dose_values["Probe dose e-A-2"]*3

    dose_list.append(dose_offset) #navigation dose
    for step in range(0,steps_needed):
        image = beam_size_matched_acquisition(pixels=16,dwell_time_s=dwell_time,output="sum")
        dataset_list.append(image)
        dose_increment = ((current_dose_rate*dwell_time)*step)+dose_offset #rate to increment
        dose_list.append(dose_increment)

    filename = r"C:\temp\dose_series\test_series.pdat"

    with open(filename,"wb") as f:
        p.dump((dataset_list,dose_list),f)

    return dataset_list,dose_list

dataset_list,dose_list = dose_series(max_dose=50)
template_pattern = dataset_list[0]

