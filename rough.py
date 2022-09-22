from glob import glob
import nibabel as nib
import numpy as np
import shutil
import os
import torch
from tqdm import tqdm
from Models.ProbUNetV2.eval import SinkhornSolver, calc_energy_distances, calc_energy_distances_pyT, get_energy_distance_components, get_energy_distance_components_pyT
from Utils.fid.fidloss import FastFID, PartialResNeXt
from torch.cuda.amp import GradScaler, autocast
import time
from torch.autograd import gradcheck
import geomloss

# op_distloss = FastFID(useInceptionNet=True,batch_size=4, gradient_checkpointing=True)
# op_distloss = SinkhornSolver(epsilon)
op_distloss = geomloss.SamplesLoss(loss='sinkhorn')
op_distloss.cuda()

# x = torch.rand(100,1,640,780, device="cuda", dtype=torch.float32)
x = torch.rand(10,10,499200, device="cuda", dtype=torch.float32)
# y = torch.rand(100,1,640,780, device="cuda", dtype=torch.float32)
y = torch.rand(10,10,499200, device="cuda", dtype=torch.float32)

times = []
with autocast(enabled=True):
    for i in range(10):
        start = time.time()
        o = op_distloss(x,y)
        # o = calc_energy_distances_pyT(get_energy_distance_components_pyT(x, y, eval_class_ids=[1]))

        times.append(time.time() - start)
print(np.median(times))

hajabarola

ob = PartialResNeXt()

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