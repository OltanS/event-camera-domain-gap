## The project for processing DDD20 dataset using RNNs for performance improvement

Flow: 0. Clone repo, set up venvs for ddd20-utils (python 2) and dvs processing (python 3) based on their respective requirements.txt files.

1. Set project variables by changing config.env
2. Preprocess the data running prepare_data.sh

### TODO

- Create venv in Odachi, install requirements +
- Create function for continuing from checkpoint when training
- Figure out how to mount onedrive to Odachi +
- Create python 2 venv for event-framing the dataset using ddd20 utils
- Write script for event framing everything on the onedrive using utils (and run it)
- Create a lazy loading version of dataset
- Change logic to lazy load the dataset

### Future

- Investigate transformers for processing the data (self attention?)
- Investigate GRU
- Investigate RNN
- Try with/without APS data
- Try with/without preprocessing
- Consider transfer learning from ImageNet for the non-RNN part of the network
