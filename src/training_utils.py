import gc

from tqdm import tqdm

import torch
import contextlib

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

        total_batches = len(loader)
        training_batches = portion_of_dataset * total_batches
        with context:
            pbar = tqdm(loader, total=training_batches, desc=f"[GPU{global_rank}] {mode}", disable=local_rank != 0)
            for i, (combined_images, steering_angle) in enumerate(pbar):
                if i >= training_batches:
                    break
                if model_type == "Resnet":
                    b, t, c, h, w = combined_images.shape
                    combined_images = combined_images.view(b * t, c, h, w)
                elif model_type != "ResnetRNN":
                    # permute the data to (batch, channels, seq_len, width, height)
                    combined_images = combined_images.permute(0, 2, 1, 3, 4)
                if mode != "TEST":
                    optimizer.zero_grad()
                combined_images = combined_images.to(local_rank)
                steering_angle = steering_angle.to(local_rank)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(combined_images).squeeze(1)
                    if model_type == "Resnet":
                        target_angles = steering_angle.view(-1)
                    else:
                        target_angles = steering_angle[:, -1]
                
                    loss = criterion(outputs, target_angles)
                    
                if mode != "TEST":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                epoch_loss += loss.item()

                # Show current loss in the progress bar
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                del combined_images, steering_angle, outputs
                torch.cuda.empty_cache()
                gc.collect()

            # Average training loss for the epoch
            epoch_loss /= len(loader)
            return epoch_loss