## The project for processing the DDD20 Dataset

Setup: 
0. Clone repo, set up venvs for ddd20-utils (python 2) and dvs processing (python 3) based on their respective requirements.txt files (after sourcing venv pip install -r requirements.txt). Versions for things isn't very important at all for python3, adjust as needed  
1. Set project variables by changing config.env  
2. run my_utils/prepare-data.sh
3. run preprocess_data.py (after sourcing dvs processing venv, python 3)
4. torchrun train_mode.py (running with python won't work, it requires torchrun to populate some environment variables)
5. (Optionally) inspect performance with evaluate_models.ipynb

### TODO:
- Add Attention to 3DCNN
- Figure out why Swin doesn't learn