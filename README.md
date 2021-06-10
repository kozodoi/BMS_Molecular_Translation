# BMS Molecular Translation

The top-5% solution to the [BMS Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation) Kaggle Competition.

![sample](https://i.postimg.cc/t4KqLMNC/inchi.jpg)


## Summary

Organic chemists frequently draw molecular work using structural graph notations. As a result, decades of scanned publications and medical documents contain drawings not annotated with chemical formulas. Currently, time-consuming manual work of experts is required to reliably convert such images into a machine-readable formula format. Automated recognition of optical chemical structures could speed up research and development in the field.

The goal of this project is to develop a deep learning based algorithm for chemical image captioning. In other words, the project aims at translating unlabeled chemical images into the text formula strings. To do that, I work with a large dataset of more than 4 million chemical images provided by Bristol-Myers Squibb.

My solution is an ensemble of seven CNN-LSTM Encoder-Decoder models. All models are implemented in `PyTorch `. The table below summarizes the main architecture and training parameters. The solution reaches the test score of 1.31 LD and places 47th out of 874 competing teams. The detailed summary is provided in [this writeup](https://www.kaggle.com/c/bms-molecular-translation/discussion/243845).

![models](https://i.postimg.cc/cLrTp1Pc/Screen-2021-06-04-at-10-17-02.jpg)


## Project structure

The project has the following structure:
- `codes/`: `.py` main scripts with data, model, training and inference modules
- `notebooks/`: `.ipynb` Colab-friendly notebooks for data augmentation and model training
- `input/`: input data (not included due to size constraints, can be downloaded [here](https://www.kaggle.com/c/bms-molecular-translation/data))
- `output/`: model configurations, weights and figures exported from the notebooks


## Working with the repo

### Environment

To work with the repo, I recommend to create a virtual Conda environment from the `environment.yml` file:
```
conda env create --name bms --file environment.yml
conda activate bms
```

### Reproducing solution

The solution can then be reproduced in the following steps:
1. Download [competition data](https://www.kaggle.com/c/bms-molecular-translation/data) and place it in the `input/` folder.
2. Run `01_preprocessing_v1.ipynb` to preprocess the data and define chemical tokenizer.
3. Run `02_gen_extra_data.ipynb` and `03_preprocessing_v2.ipynb` to construct additional synthetic images.
4. Run training notebooks `04_model_v6.ipynb` - `10_model_v33.ipynb` to obtain weights of base models.
5. Perform normalization of each model predictions using `11_normalization.ipynb`.
6. Run the ensembling notebook `12_ensembling.ipynb` to obtain the final predictions.

All training notebooks have the same structure and differ in model/data parameters. Different versions are included to ensure reproducibility. To understand the training process, it is sufficient to go through the `codes/` folder and inspect one of the modeling notebooks. The ensembling code is also provided in this [Kaggle notebook](https://www.kaggle.com/kozodoi/47th-place-solution-bms-ensembling).

More details are provided in the documentation within the scripts & notebooks.
