import pickle as p
import numpy as np
import easygui as g
from tqdm import tqdm
from expert_pi.RSTEM.bda_functions import beam_size_matched_acquisition
from expert_pi.RSTEM.utilities import get_microscope_parameters, calculate_dose


#TODO V1.0 offline analysis


def dose_series(max_dose=False,num_steps=False,directory=None,dwell_time=None):
    """This should do a calulation of the current hardware dose and run a
    time resolved 4D-STEM acquisition that sums the data to a single pattern, saving to a pdat file"""

    if dwell_time is None:
        dwell_time = 56e-6 #4 precession cycles, dwell in seconds

    current_dose_values = calculate_dose()
    current_dose_rate = current_dose_values["Probe dose rate-A-2s-1"]
    print("Current dose rate",np.round(current_dose_rate,2),"e-A-2s-1")
    print("Dose step increment",np.round(current_dose_rate*dwell_time,1),"e-A-2")

    if directory is None:
        directory = g.diropenbox("Select save directory","Save directory")
        
    filename = directory+"\\dose series.pdat"
    

    dose_steps_needed = int((max_dose/current_dose_rate)*1.1) #additional 10% buffer
    if max_dose==False:
        steps_needed=num_steps
    else :
        steps_needed=dose_steps_needed #either uses dose target or number of steps
    dataset_list = []
    dose_list = []
    dose_offset = current_dose_values["Probe dose e-A-2"]*3 #assume 3 navigation scans to get the region selected
    dose_increment = current_dose_rate*dwell_time
    dose_list.append(dose_offset) #navigation dose rather than zero dose
    
    for step in tqdm(range(1,steps_needed+1)):
        image = beam_size_matched_acquisition(pixels=32,dwell_time_s=dwell_time,output="sum")
        dataset_list.append(image)
        dose_list.append((dose_increment+dose_list[-1]))

    with open(filename,"wb") as f:
        p.dump((dataset_list,dose_list),f)

    return dataset_list,dose_list

