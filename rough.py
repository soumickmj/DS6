from glob import glob
import nibabel as nib
import numpy as np
import shutil
import os
from tqdm import tqdm

root = "/project/schatter/FranziVSeg/Data/Forrest_Organised/Fold0/test_plausablelabel"
files = glob(root+"/**.nii.gz")
root_orig = "/project/schatter/FranziVSeg/Data/Forrest_Organised/Fold0/test"
for f in tqdm(files):
    im = nib.load(f)
    im_orig = nib.load(f"{root_orig}/{os.path.split(f)[-1].split('_')[0]+'.nii.gz'}")
    im_new = nib.Nifti1Image(im.get_fdata(), im_orig.affine, header=im_orig.header)
    nib.save(im_new, f)


root = r"/home/schatter/Soumick/FranziVSeg/DS_Original/Vols/Forrest_Original"
new_root = r"/home/schatter/Soumick/FranziVSeg/DS_Original/Vols/Forrest_Organised/Fold0"



files = glob(f"{new_root}/**/*.nii*", recursive=True)

for f in files:
    print(f)
    im = nib.load(f)
    # print(im.header["pixdim"].round(1))
    # shutil.copyfile(f, f"{new_root}/{f.replace(root,'').split('/')[1]}.nii.gz")
    print(im.shape)