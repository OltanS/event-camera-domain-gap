import gc

from tqdm import tqdm

import torch
import contextlib
     
def run_epoch(model, loader, optimizer, scaler, criterion, mode, portion_of_dataset, local_rank, global_rank, model_type, writer=None, epoch=0):
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
            for i, (combined_images, steering_angles) in enumerate(pbar):
                if i >= training_batches:
                    break
                if mode != "TEST":
                    optimizer.zero_grad()
                
                combined_images = combined_images.to(local_rank)
                steering_angles = steering_angles.to(local_rank)
                b, c, h, w = combined_images.shape
                outputs = model(combined_images).squeeze(1)
                loss = criterion(outputs, steering_angles)

                if mode != "TEST":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
            
                    scaler.step(optimizer)
                    scaler.update()
                
                epoch_loss += loss.item()

                # Show current loss in the progress bar
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                del combined_images, steering_angles, outputs
                torch.cuda.empty_cache()
                gc.collect()

            # Average training loss for the epoch
            epoch_loss /= len(loader)

            return epoch_loss
