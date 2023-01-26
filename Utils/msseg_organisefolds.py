import os
from glob import glob
import numpy as np
from tqdm import tqdm
import random
import nibabel as nib

root = "/media/Enterprise/ProbUNet_MultiSc/Data/Original"

all_labels = glob(f"{root}/**/Consensus.nii.gz", recursive=True)
random.shuffle(all_labels)

meta_dict = []
for l in all_labels:
    label = nib.load(l).get_fdata()
    brain_mask = nib.load(l.replace("Consensus.nii.gz", "Brain_Mask.nii.gz")).get_fdata()