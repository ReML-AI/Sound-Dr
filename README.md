# Sound-Dr: Reliable Sound Dataset and Baseline Artificial Intelligence System for Respiratory Illnesses
This project implements the baseline and benchmarks on Sound-Dr Dataset. 

>As the burden of respiratory diseases continues to fall on society worldwide, this paper proposes a high-quality and reliable dataset of human sounds for studying respiratory illnesses, including pneumonia and COVID-19. It consists of coughing, mouth breathing, and nose breathing sounds together with metadata on related clinical characteristics. We also develop a proof-of-concept system for establishing baselines and benchmarking against multiple datasets, such as Coswara and COUGHVID. Our comprehensive experiments show that the Sound-Dr dataset has richer features, better performance, and is more robust to dataset shifts in various machine learning tasks. It is promising for a wide range of real-time applications on mobile devices. The proposed dataset and system will serve as practical tools to support healthcare professionals in diagnosing respiratory disorders.

### Dataset
Link: [Detail here](./sounddr_data/README.md)

### About this implementation
This repository contains the official implementation (in Tensorflow+Keras) of the **Sound-Dr: Reliable Sound Dataset and Baseline Artificial Intelligence System for Respiratory Illnesses** and keep some results in paper.

Our main source code was written and ran on python and [JupyterLab](https://jupyter.org) in the following directory:

- [Baseline System](./SoundDr_BaselineSystem.py)
    + Edit ```config.py``` for settings, parameters
    + Train a baseline with command : ```python3 main.py```

- [Some results in paper](./sounddr_data/output). We do not clean up them and keep some cache files(Feature, fold split csv) because we want to keep original results.

- [Overall system](./SoundDr_cough.ipynb). Overall system, include Unsupervised(Isolation Forest, XGBOD) at last this notebook.
    + Chooose dataset dataset_type='SoundDr'
    + chooose pretrain model to extract Feature PRETRAIN="FRILL"
    + Chooose model to classify Classifier="SVM"
    + Set seed seed=2022"

### Requirements
Please, install the following packages
- numpy
- tqdm
- pandas
- zipfile
- pickle
- h5py
- joblib
- librosa
- opensmile
- xgboost
- tensorflow_hub
- scipy
- sklearn


### How to cite this work?
```
@inproceedings{Truong:2023,
    title={Sound-Dr: Reliable Sound Dataset and Baseline Artificial Intelligence System for Respiratory Illnesses},
    author={Truong, Hoang Van and Quang, Nguyen Huu and Cuong, Nguyen Quoc and Phong, Nguyen Xuan and Hoang, D. Nguyen},
    title={4th Asia Pacific Conference of the Prognostics and Health Management Society (PHMAP)},
    year={2023},
}
```

## Reference
This project is based on the following implementations:

- https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark
- https://github.com/audeering/opensmile
- https://github.com/DeepSpectrum/DeepSpectrum
- https://github.com/steverab/failing-loudly

