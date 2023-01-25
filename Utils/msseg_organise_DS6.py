import os
from glob import glob
import numpy as np
from tqdm import tqdm
import random
import nibabel as nib
import shutil
import pandas as pd

root = "/project/schatter/Ale/MS/Data"
destin_root = "/project/schatter/FranziVSeg/Data/MSSEG_Organised/FLAIR_Fold0"
inp_con = "FLAIR" 

all_labels = glob(
    f"{root}/**/*_0p7Bicub_crop_pad224X288/Consensus.nii.gz", recursive=True)
# all_labels = glob(f"{root}/**/Masks_0p7Bicubic_cropped/Consensus.nii.gz", recursive=True)
random.seed(1701)
random.shuffle(all_labels)

contrasts = ["T1", "T2", "FLAIR", "DP", "GADO"]
contrasts.remove(inp_con)

n_train = 32
n_val = 7

meta_dict = []
SubID = 0
for l in tqdm(all_labels):
    origLbl = nib.load(l.replace("_0p7Bicub_crop_pad224X288", ""))
    origX, origY, origZ = origLbl.shape
    resX, resY, resZ = origLbl.header.get_zooms()

    interpX, interpY, interpZ = nib.load(l.replace("_0p7Bicub_crop_pad224X288", "_0p7Bicubic")).shape

    path_parts = os.path.normpath(l).split(os.sep)
    datum = {"CentreID": int(path_parts[-4].split("_")[1]), "PatientID": int(path_parts[-3].split("_")[1]), "SubID": SubID, "origSplit": path_parts[-5],
            "origX": origX, "origY": origY, "origZ": origZ, "resX": resX, "resY": resY, "resZ": resZ, "interpX": interpX, "interpY": interpY, "interpZ": interpZ}
    meta_dict.append(datum)

    if SubID < n_train:
        split = "train"
    elif SubID < (n_train+n_val):
        split = "validate"
    else:
        split = "test"

    l_new = f"{destin_root}/{split}_label/sub{str(SubID).zfill(3)}.nii.gz"
    os.makedirs(os.path.dirname(l_new), exist_ok=True)
    shutil.copyfile(l, l_new)

    inp_old = l.replace("Masks_0p7Bicub_crop_pad224X288", "Preprocessed_Data_0p7Bicub_crop_pad224X288").replace("Consensus.nii.gz", f"{inp_con}_preprocessed.nii.gz")
    inp_new = f"{destin_root}/{split}/sub{str(SubID).zfill(3)}.nii.gz"
    os.makedirs(os.path.dirname(inp_new), exist_ok=True)
    shutil.copyfile(inp_old, inp_new)

    inp_old_vol = nib.load(inp_old)
    mCon = [ inp_old_vol.get_fdata() ]
    for c in contrasts:
        mCon.append(nib.load(inp_old.replace(f"{inp_con}_preprocessed.nii.gz", c+"_preprocessed.nii.gz")).get_fdata())
    mCon = np.stack(mCon, axis=-1)
    mCon_path = f"{destin_root}/{split}_multiCon/sub{str(SubID).zfill(3)}.nii.gz"
    os.makedirs(os.path.dirname(mCon_path), exist_ok=True)
    nib.save(nib.Nifti1Image(mCon, inp_old_vol.affine, header=inp_old_vol.header), mCon_path)

    for m in range(1,8):
        pL_old = l.replace("Consensus.nii.gz", f"ManualSegmentation_{m}.nii.gz")
        pL_new = f"{destin_root}/{split}_plausiblelabel/sub{str(SubID).zfill(3)}_Segmentation_{m-1}.nii.gz"
        os.makedirs(os.path.dirname(pL_new), exist_ok=True)
        shutil.copyfile(pL_old, pL_new)
        
    SubID += 1
    
df = pd.DataFrame.from_dict(meta_dict)
df.to_csv(os.path.dirname(destin_root)+"/meta.csv")
