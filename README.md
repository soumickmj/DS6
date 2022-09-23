# DS6
Official code of the paper "DS6, Deformation-aware Semi-supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data" (https://arxiv.org/abs/2006.10802)
Recording of the live-stream of the talk at Stanford University: https://youtu.be/p1RrvlMMhOI

Docker container and pre-trained weights will soon be made available.

The name of this project "DS6" was selected as a tribute to Star Trek: Deep Space Nine (DS9).
DS6 = Deformation-aware Segmentation, trained on 6 MR Volumes. 

## Credits

If you like this repository, please click on Star!

If you use this approach in your research or use codes from this repository, please cite the following in your publications:

> [Chatterjee S, Prabhu K, Pattadkal M, Bortsova G, Sarasaen C, Dubost F, Mattern H, de Bruijne M, Speck O, Nürnberger A. DS6: Deformation-Aware Semi-Supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data. Journal of Imaging. 2022; 8(10):259.](https://doi.org/10.3390/jimaging8100259)

BibTeX entry:

```bibtex
@Article{chatterjee2022ds6,
AUTHOR = {Chatterjee, Soumick and Prabhu, Kartik and Pattadkal, Mahantesh and Bortsova, Gerda and Sarasaen, Chompunuch and Dubost, Florian and Mattern, Hendrik and de Bruijne, Marleen and Speck, Oliver and Nürnberger, Andreas},
TITLE = {DS6: Deformation-Aware Semi-Supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data},
JOURNAL = {Journal of Imaging},
VOLUME = {8},
YEAR = {2022},
NUMBER = {10},
ARTICLE-NUMBER = {259},
URL = {https://www.mdpi.com/2313-433X/8/10/259},
ISSN = {2313-433X},
ABSTRACT = {Blood vessels of the brain provide the human brain with the required nutrients and oxygen. As a vulnerable part of the cerebral blood supply, pathology of small vessels can cause serious problems such as Cerebral Small Vessel Diseases (CSVD). It has also been shown that CSVD is related to neurodegeneration, such as Alzheimer&rsquo;s disease. With the advancement of 7 Tesla MRI systems, higher spatial image resolution can be achieved, enabling the depiction of very small vessels in the brain. Non-Deep Learning-based approaches for vessel segmentation, e.g., Frangi&rsquo;s vessel enhancement with subsequent thresholding, are capable of segmenting medium to large vessels but often fail to segment small vessels. The sensitivity of these methods to small vessels can be increased by extensive parameter tuning or by manual corrections, albeit making them time-consuming, laborious, and not feasible for larger datasets. This paper proposes a deep learning architecture to automatically segment small vessels in 7 Tesla 3D Time-of-Flight (ToF) Magnetic Resonance Angiography (MRA) data. The algorithm was trained and evaluated on a small imperfect semi-automatically segmented dataset of only 11 subjects; using six for training, two for validation, and three for testing. The deep learning model based on U-Net Multi-Scale Supervision was trained using the training subset and was made equivariant to elastic deformations in a self-supervised manner using deformation-aware learning to improve the generalisation performance. The proposed technique was evaluated quantitatively and qualitatively against the test set and achieved a Dice score of 80.44 &plusmn; 0.83. Furthermore, the result of the proposed method was compared against a selected manually segmented region (62.07 resultant Dice) and has shown a considerable improvement (18.98%) with deformation-aware learning.},
DOI = {10.3390/jimaging8100259}
}

```
Thank you so much for your support.

