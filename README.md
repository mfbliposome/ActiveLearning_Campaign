# ActiveLearning_Campaign

This repo is for archive AL rounds for image analysis and model construction

## Folder structure
In this repo, it have 2 subfolders: 'code', 'data'.

### folder 'code'
- In this folder, it has 3 subfolders: 'src', 'notebooks', 'Templates', which store the module function, working notebooks, and templates.

### folder 'data'
In this folder, it has stored the data files for each active learning rounds including:
-  input data for model (e.g.'df_input_20240429.csv'),
-  input data with original scale (e.g. 'df_input_ori_20240429.csv'),
-  updated datapool for current round ('df_input_update_20240511.csv'),
-  next plan for the experiment (e.g. 'dispense_df_20240416.xlsx'),
-  constructed model for current datapool,
-  turbidity data,
-  cvc data
-  Original microscopy image and vesicles detection results. These data files can be downloaded from [[Zenodo link](https://doi.org/10.5281/zenodo.12522610)]

Note, in this repository, we have used the term 'monocaprin' as a shorthand for glycerol monodecanoate (GMD). However, in the manuscript, we have opted to use the formal name, glycerol monodecanoate (GMD), to align with standard terminology in the scientific community and ensure clarity for the readership.

## Requirements for libraries
- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- botorch
- opencv

## Citation

If you use this repository in your research, please cite it as follows: TBD

