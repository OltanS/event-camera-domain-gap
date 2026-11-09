# this assumes it gets called from the root
import os
from dotenv import load_dotenv
import numpy as np

def setup_config_variables():
    '''
    Loads the project configuration from config.env from root. Assumes this function is called from root.
    returns a dictionary of config variables.  
    Has the following derived variables in addition to the ones outlined in config.env:  
    config["MODEL_DIMS"] = config["USE_DVS"] + config["USE_APS"]  
    config["DTYPE"] = np.float16 if config["PRECISION"] == 16 else np.float32  

    from torchrun:
    config["LOCAL_RANK"] = int(os.getenv("LOCAL_RANK", 0))  
    config["GLOBAL_RANK"] = int(os.getenv("RANK", 0))

    returns: dict of config variables
    '''

    load_dotenv("config.env")
    
    config = {}
    # --- Data Visualisation Options ---
    config["ANIMATE_LOADER"] = bool(int(os.getenv("ANIMATE_LOADER")))
    
    # --- Training parameters ---
    config["MODEL_TYPE"] = os.getenv("MODEL_TYPE")
    config["USE_DVS"] = int(os.getenv("USE_DVS"))
    config["USE_APS"] = int(os.getenv("USE_APS"))
    config["MODEL_DIMS"] = config["USE_DVS"] + config["USE_APS"]
    if config["MODEL_DIMS"] == 0:
        raise Exception("MODEL_DIMS must be non-zero")
    
    config["WARM_START"] = bool(int(os.getenv("WARM_START")))
    config["EPOCHS"] = int(os.getenv("EPOCHS"))
    config["BATCH_SIZE"] = int(os.getenv("BATCH_SIZE"))
    config["LEARNING_RATE"] = float(os.getenv("LEARNING_RATE"))
    config["WEIGHT_DECAY"] = float(os.getenv("WEIGHT_DECAY"))
    
    # --- Preprocessing parameters ---
    config["TRAINING_SPLIT"] = float(os.getenv("TRAINING_SPLIT"))
    config["PORTION_OF_DATASET"] = float(os.getenv("PORTION_OF_DATASET", 1.0))
    config["WINDOW_SIZE"] = int(os.getenv("WINDOW_SIZE"))
    config["RANDOM_SEED"] = int(os.getenv("RANDOM_SEED"))
    config["PERFORM_PRUNING"] = bool(int(os.getenv("PERFORM_PRUNING")))
    config["PRUNE_SPEED"] = bool(int(os.getenv("PRUNE_SPEED")))
    config["PRUNE_STEERING"] = bool(int(os.getenv("PRUNE_STEERING")))
    config["SPEED_THRESHOLD"] = float(os.getenv("SPEED_THRESHOLD"))
    config["STEERING_THRESHOLD"] = float(os.getenv("STEERING_THRESHOLD"))
    config["STEERING_DISCARD_CHANCE"] = float(os.getenv("STEERING_DISCARD_CHANCE"))
    config["MIN_STEERING"] = int(os.getenv("MIN_STEERING"))
    config["MAX_STEERING"] = int(os.getenv("MAX_STEERING"))

    # --- Dataset parameters, for Event Framing with scripts/prepare-dataset.sh ---
    config["EXPORT_DVS"] = os.getenv("EXPORT_DVS")
    config["EXPORT_APS"] = os.getenv("EXPORT_APS")
    config["BINSIZE_MS"] = int(os.getenv("BINSIZE_MS"))
    config["HEIGHT"] = int(os.getenv("HEIGHT"))
    config["WIDTH"] = int(os.getenv("WIDTH"))
    config["APS_FRAME_NAME"] = os.getenv("APS_FRAME_NAME")
    config["DVS_FRAME_NAME"] = os.getenv("DVS_FRAME_NAME")
    config["PRECISION"] = int(os.getenv("PRECISION"))
    config["DTYPE"] = np.float16 if config["PRECISION"] == 16 else np.float32
    config["AUGMENT_DATA"] = bool(int(os.getenv("AUGMENT_DATA")))
    
    # --- Location parameters ---
    config["OUT_DIR"] = os.getenv("OUT_DIR")
    config["DATA_DIR"] = os.getenv("DATA_DIR")
    config["MODEL_SAVE_DIR"] = os.getenv("MODEL_SAVE_DIR")
    config["PROCESSED_DIR"] = os.getenv("PROCESSED_DIR")
    config["TRAINING_DIR"] = os.getenv("TRAINING_DIR")
    config["TEST_DIR"] = os.getenv("TEST_DIR")
    config["IMAGES_DIR"] = os.getenv("IMAGES_DIR")
    config["CSV_FILE_MANUAL"] = os.getenv("CSV_FILE_MANUAL")

    # --- Multi-node training parameters ---
    config["LOCAL_RANK"] = int(os.getenv("LOCAL_RANK", 0))
    config["GLOBAL_RANK"] = int(os.getenv("RANK", 0))
    


    for key, value in config.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"Couldn't initialise {key} from the config file!")

    return config