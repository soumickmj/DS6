from glob import glob
import nibabel as nib
import numpy as np
import shutil

root = r"/home/schatter/Soumick/FranziVSeg/DS_Original/Vols/Forrest_Original"
new_root = r"/home/schatter/Soumick/FranziVSeg/DS_Original/Vols/Forrest_Organised/Fold0"



files = glob(f"{new_root}/**/*.nii*", recursive=True)

for f in files:
    print(f)
    im = nib.load(f)
    # print(im.header["pixdim"].round(1))
    # shutil.copyfile(f, f"{new_root}/{f.replace(root,'').split('/')[1]}.nii.gz")
    print(im.shape)