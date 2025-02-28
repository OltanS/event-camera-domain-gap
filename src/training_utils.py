import gc

import torch
import contextlib
from src.logging_utils import report_progress

def run_epoch(model, loader, optimizer, scaler, criterion, mode, portion_of_dataset, local_rank, global_rank, model_type):
        '''
        Run a single epoch of training or testing.
        args:
            model: nn.Module, the model to train
            loader: DataLoader, the data loader
            optimizer: torch.optim, the optimizer
            scaler: torch.amp.GradScaler, the gradient scaler, for mixed precision training
            criterion: torch.nn, the loss function
            mode: str, "TRAIN" or "TEST", determines if the model is training or testing
            portion_of_dataset: float, the portion of the dataset to train on, between 0 and 1
            local_rank: int, the local rank of the process
            global_rank: int, the global rank of the process
            model_type: str, the type of model, see src/model_utils.py for options
        returns:
            float, the average loss for the epoch
        '''
        epoch_loss = 0
        context = torch.no_grad() if mode == "TEST" else contextlib.nullcontext()

        # report every 25% progress
        total_batches = len(loader)
        training_batches = portion_of_dataset * total_batches
        progress_thresholds = {int(training_batches * 0.25), int(training_batches * 0.50), int(training_batches * 0.75)}
        with context:
            for i, (combined_images, steering_angle) in enumerate(loader):
                if i >= training_batches:
                    break
                if model_type == "Resnet":
                    b, t, c, w, h = combined_images.shape
                    combined_images = combined_images.view(b * t, c, w, h)
                elif model_type != "ResnetRNN":
                    combined_images = combined_images.permute(0, 2, 1, 3, 4)
                if mode != "TEST":
                    optimizer.zero_grad()
                combined_images = combined_images.to(local_rank)
                steering_angle = steering_angle.to(local_rank)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(combined_images).squeeze(1)
                    if model_type == "Resnet":
                        loss = criterion(outputs, steering_angle.view(-1))
                    else:
                        last_angles = steering_angle[:, -1]
                        loss = criterion(outputs, last_angles)
                if mode != "TEST":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                epoch_loss += loss.item()
                # Report progress if we have hit one of the thresholds
                if (i + 1) in progress_thresholds:
                    progress = round(100 * (i + 1) / total_batches) 
                    report_progress(global_rank, f"{mode} Epoch progress: {progress}% completed.")
                
                del combined_images, steering_angle, outputs
                torch.cuda.empty_cache()
                gc.collect()

            # Average training loss for the epoch
            epoch_loss /= len(loader)
            return epoch_loss