# Code for [papername](https://example.com)

## Preliminary Steps
- Acquire the dataset from the [ETH Libdrive](https://www.research-collection.ethz.ch/handle/20.500.11850/707895) or as otherwise directed in the [DDD20 Website](https://sites.google.com/view/davis-driving-dataset-2020/home). We note that downloads from the libdrive can silently fail with unfinished downloads appearing as finished, and recommend downloading the dataset file-by-file (and keeping external track of which files have been received), rather than folder-by-folder, which we found to be more robust.
- The original DDD20 utilities that were utilized to event-frame the data work over Python 2.7.18 (confirmed to work by us), which is no longer trivial to install. We recommend [pyenv](https://github.com/pyenv/pyenv) to install the relevant Python version.

## How to Use
1. Create a Python2 virtual environment, and install the [requirements](ddd20-utils/requirements.txt). These are different than the requirements as presented in the original repository. The new list works around the depreciation issues faced, and we confirm it works. Optionally inspect the steps provided in the repository's [README](ddd20-utils/README.md) for further details.

2. Create a Python 3 virtual environment, and install the requirements as presented in [requirements.txt](requirements.txt).
3. Populate the following location options of [config.env](config.env) to match up to your installation of the dataset. In detail:
```
VENV_DIR=nameofyourpython2venvdirectory

IN_DIR=/location/of/DDD20andDDD17/
OUT_DIR=/where/you/want/Binned-Recordings/bin${BINSIZE_MS}ms
PROCESSED_DIR=/where/you/want/Preprocessed-Recordings

MODEL_SAVE_DIR=/path/to/save/models/${MODEL_TYPE}
IMAGES_DIR/path/to/save/images/${MODEL_TYPE}
```
4. (Optional) Change the data options in [config.env](config.env). Note that changing the export_aps, export_dvs, width and height are certain to break the code's ability to work out of the box (use_aps and use_dvs can be changed instead for training on only one of the data and width and height are hardcoded into the neural networks and other places), while the binsize is alterable, and if desired should be changed before the event framing process.
5. Bin the data using [my_utils/prepare-data.sh](my_utils/prepare-data.sh). This will take a while. It utilizes the ddd20 utils. We found that it cut off some recordings abruptly, and worked best with a 50ms framing window. We prune these dead-air sections in our preprocessing. In case the abrupt terminations are not deterministic, the user can utilize the commented out lines at the end of [evaluate_models.ipynb](evaluate_models.ipynb) with no model to visualize the preprocessed recordings, and [note down the start and ending times](my_utils/csv_files/real-start-stop.csv). Populating this file will enable the dead-air to be pruned in the preprocessing step.
6. (Optional) Change the preprocessing options in [config.env](config.env). Changing the max and min steering will require altertions to the [steering angle scaling performed](src/data_utils.py).
7. Activate your python 3 virtual environment, and run preprocess_data.py. This will take a while. Note that we observed that DDD17 and DDD20 have recordings that are flipped and steering angle values that are mirrored relative to each other, and this is rectified here.
8. Change the training options in [config.env](config.env) to choose your model and hyperparameters of choice. More details on the available models can be found [here](src/model_utils.py).
9. (Optional) Change animate_loader to 1 if you'd like a visualization of the data loaders, showing the DVS, APS and combined frames in addition to the steering angle.
10. Since the model was trained in a distributed environment, you should run the training on a single machine via ```torchrun train_model.py```.