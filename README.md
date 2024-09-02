# DS6
Official code of the paper "DS6, Deformation-aware Semi-supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data" (https://arxiv.org/abs/2006.10802)
Recording of the live-stream of the talk at Stanford University: https://youtu.be/p1RrvlMMhOI

Docker container and pre-trained weights will soon be made available.

The name of this project "DS6" was selected as a tribute to Star Trek: Deep Space Nine (DS9).
DS6 = Deformation-aware Segmentation, trained on 6 MR Volumes. 

 ## Model Weights
The weights of the models trained during this research, as well as models trained on the SMILE-UHURA challenge dataset during the challenge to create baselines, have been made publicly available on Huggingface, and they can be found in the collection: [https://huggingface.co/collections/soumickmj/ds6-66d623af4a69536bcaf3c377](https://huggingface.co/collections/soumickmj/ds6-66d623af4a69536bcaf3c377). The designations "woDeform" and "wDeform" within the model names indicate that the respective model was trained without (baseline) and with (DS6) deformation-aware learning, respectively. Model names starting with "SMILEUHURA_DS6_" signify that they were trained on the SMILE-UHURA dataset, while names starting with only "DS6_" signify that they are the models trained during the original paper. 

The weights can be directly be used pulling from Huggingface with the updated version of this pipeline, or the weights can be downloaded using the AutoModel class from the transformers package, saved as a checkpoint, and then the path to this saved checkpoint can be supplied to the pipeline using "-load_path" argument.

Here's an example of how to use directly use weights from huggingface:
```bash
-load_huggingface soumickmj/DS6_UNetMSS3D_wDeform
```
Additional parameter "-load_huggingface" must be supplied along with the other desired paramters. Technically, this paramter can also be used to supply segmentation models other than the models used in DS6. 

Here is an example of how to save the weights locally (must be saved with .pth extension) and then use it with this pipeline:
```python
from transformers import AutoModel
modelHF = AutoModel.from_pretrained("soumickmj/DS6_UNetMSS3D_wDeform", trust_remote_code=True)
torch.save({'state_dict': modelHF.model.state_dict()}, "/path/to/checkpoint/model.pth")
```
To run this pipeline with these downloaded weights, the path to the checkpoint must then be passed as preweights_path, as an additional parameter along with the other desired parameters:
```bash
-load_path /path/to/checkpoint/model.pth
```

## Credits

If you like this repository, please click on Star!

If you use this approach in your research or use codes from this repository, please cite the following in your publications:

> [Chatterjee S, Prabhu K, Pattadkal M, Bortsova G, Sarasaen C, Dubost F, Mattern H, de Bruijne M, Speck O, Nürnberger A. DS6, Deformation-Aware Semi-Supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data. Journal of Imaging. 2022; 8(10):259.](https://doi.org/10.3390/jimaging8100259)

BibTeX entry:

```bibtex
@Article{chatterjee2022ds6,
          AUTHOR = {Chatterjee, Soumick and Prabhu, Kartik and Pattadkal, Mahantesh and Bortsova, Gerda and Sarasaen, Chompunuch and Dubost, Florian and Mattern, Hendrik and de Bruijne, Marleen and Speck, Oliver and Nürnberger, Andreas},
          TITLE = {DS6, Deformation-Aware Semi-Supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data},
          JOURNAL = {Journal of Imaging},
          VOLUME = {8},
          YEAR = {2022},
          NUMBER = {10},
          ARTICLE-NUMBER = {259},
          URL = {https://www.mdpi.com/2313-433X/8/10/259},
          ISSN = {2313-433X},
          DOI = {10.3390/jimaging8100259}
}

```
Thank you so much for your support.

