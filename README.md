## The project for processing DDD20 dataset using RNNs for performance improvement
What to do below:

### TODO
- Create venv in Katana, install requirements
- Create function for continuing from checkpoint when training
- Figure out how to mount onedrive to Katana
- Create python 2 venv for event-framing the dataset using ddd20 utils
- Write script for event framing everything on the onedrive using utils (and run it)
- Create a lazy loading version of dataset
- Lazy preprocess the dataset
- Change logic to lazy load the dataset

### Future
- Investigate transformers for processing the data (self attention?)
- Investigate GRU
- Investigate RNN
- Try with/without APS data
- Try with/without preprocessing
- Consider transfer learning from ImageNet for the non-RNN part of the network
