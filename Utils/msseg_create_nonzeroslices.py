import os
from glob import glob
import numpy as np
from tqdm import tqdm
import random
import nibabel as nib

root = "/project/schatter/Ale/MS/Data"

all_labels = glob(f"{root}/**/*_0p7Bicub_crop_pad256X288/Consensus.nii.gz", recursive=True)
# all_labels = glob(f"{root}/**/Masks_0p7Bicubic_cropped/Consensus.nii.gz", recursive=True)
random.shuffle(all_labels)

meta_dict = []
x = []
y = []
z = []
for l in tqdm(all_labels):
    brain_mask = nib.load(l.replace("Consensus.nii.gz", "Brain_Mask.nii.gz")).get_fdata()
    x.append(brain_mask.shape[0])
    y.append(brain_mask.shape[1])
    z.append(brain_mask.shape[2])
    # print(brain_mask.shape)
    # coords = np.argwhere(brain_mask)
    # x_min, y_min, z_min = coords.min(axis=0)
    # x_max, y_max, z_max = coords.max(axis=0)

    # # brain_mask_cropped = brain_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    # allNIIs = glob(l.replace("Consensus.nii.gz", "*.nii.gz")) + glob(l.replace("Masks_0p7Bicubic", "Preprocessed_Data_0p7Bicubic").replace("Consensus.nii.gz", "*.nii.gz"))
    # for n in allNIIs:
    #     img = nib.load(n)
    #     data = img.get_fdata()
    #     data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    #     n_new = n.replace("_0p7Bicubic", "_0p7Bicubic_cropped")
    #     os.makedirs(os.path.split(n_new)[0], exist_ok=True)

    #     img_new = nib.Nifti1Image(data, img.affine, header=img.header)
    #     nib.save(img_new, n_new)

s