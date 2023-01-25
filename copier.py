import os
import gc

commands = [
                "cp -r /mnt/FIN_DBMS_V100/home/schatter/Soumick/FranziVSeg/Output/Forrest_ManualSeg_Fold0/ProbU3Dv2_At0 /mnt/URZ/fnw/ifp/bmmr/data/Franzi/ProbUNet/Forrest_ManualSeg_Fold0/3D",
                "cp -r /mnt/FIN_DBMS_V100/home/schatter/Soumick/FranziVSeg/Output/Forrest_ManualSeg_Fold0/ProbU2Dv2_DistLossHausdorffEng_At5_pLBL4TrainANDMan4Val_DistVal /mnt/URZ/fnw/ifp/bmmr/data/Franzi/ProbUNet/Forrest_ManualSeg_Fold0/2D",
                
                # "cp -r /media/Enterprise/FranziVSeg/Output/Forrest_ManualSeg_Fold0/ProbU2Dv2_DistLossPureFID_At3_pLBL4TrainANDMan4Val_DistVal /mnt/URZ/fnw/ifp/bmmr/data/Franzi/ProbUNet/Forrest_ManualSeg_Fold0/2D",
                
                "cp -r /mnt/FCMn0301/project/schatter/FranziVSeg/Output/Forrest_ManualSeg_Fold0/ProbU2Dv2_At1_pLBLnMan4TrainVal /mnt/URZ/fnw/ifp/bmmr/data/Franzi/ProbUNet/Forrest_ManualSeg_Fold0/2D",
                "cp -r /mnt/FCMn0301/project/schatter/FranziVSeg/Output/Forrest_ManualSeg_Fold0/ProbU2Dv2_At0 /mnt/URZ/fnw/ifp/bmmr/data/Franzi/ProbUNet/Forrest_ManualSeg_Fold0/2D",
                "cp -r /mnt/FCMn0301/project/schatter/FranziVSeg/Output/Forrest_ManualSeg_Fold0/ProbU2Dv2_At4_pLBL4TrainANDVal /mnt/URZ/fnw/ifp/bmmr/data/Franzi/ProbUNet/Forrest_ManualSeg_Fold0/2D",
                
                
            ]

for command in commands:
    print(command)
    try:
        os.system(command)
        gc.collect()
    except:
        pass

