import os
from glob import glob
import numpy as np
from tqdm import tqdm
import random
import nibabel as nib

root = "/project/schatter/Ale/MS/Data"

all_labels = glob(f"{root}/**/*_0p7Bicubic_cropped/*.nii.gz", recursive=True)
random.shuffle(all_labels)

target_sliceshp = (224,288)
# x = []
# y = []
# z = []
for l in tqdm(all_labels):
    img = nib.load(l)
    data = img.get_fdata()
    diff_shp = np.subtract(target_sliceshp, data.shape[:2])
    diff_oneside = diff_shp // 2

    data_new = np.pad(data, ((diff_oneside[0], diff_shp[0]-diff_oneside[0]),(diff_oneside[1], diff_shp[1]-diff_oneside[1]),(0,0)))

    l_new = l.replace("_0p7Bicubic_cropped", f"_0p7Bicub_crop_pad{target_sliceshp[0]}X{target_sliceshp[1]}")
    os.makedirs(os.path.split(l_new)[0], exist_ok=True)

    img_new = nib.Nifti1Image(data_new, img.affine, header=img.header)
    nib.save(img_new, l_new)

