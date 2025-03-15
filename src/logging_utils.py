import datetime
from torch.utils.tensorboard import SummaryWriter
import os

def log_loss(epoch, train_loss, test_loss, log_dir):
    '''Logs the training and test loss to a CSV file
    Args:
        epoch (int): The current epoch
        train_loss (float): The training loss
        test_loss (float): The test loss
        log_dir (str): The directory to save the log file
    '''
    log_file = f"{log_dir}/loss_log.csv"
    with open(log_file, "a") as f:
        f.write(f"{epoch},{train_loss},{test_loss}\n")

def get_timestamp():
    return datetime.datetime.now().strftime("[%H:%M:%S]")

def report_progress(global_rank, message, epoch=None, total_epochs=None):
    epoch_str = f"[E:{epoch + 1}/{total_epochs}]" if epoch is not None and total_epochs is not None else ""
    print(f"{get_timestamp()}[GPU{global_rank}]{epoch_str} {message}")

def setup_tensorboard(log_dir, global_rank):
    log_dir = os.path.join(log_dir, "tensorboard")
    if global_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir)
    
    return None

def close_tensorboard(writer):
    if writer is not None:
        writer.close()