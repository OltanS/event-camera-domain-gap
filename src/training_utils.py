from tqdm import tqdm
import torch
     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(model, loader, optimizer, criterion, mode):
    '''
    Run a single epoch of training or testing.
    args:
        model: nn.Module, the model to train
        loader: DataLoader, the data loader
        optimizer: torch.optim, the optimizer
        criterion: torch.nn, the loss function
        mode: str, "TRAIN" or "TEST", determines if the model is training or testing
    returns:
        float, the average loss for the epoch
    '''
    model.train() if mode == "TRAIN" else model.eval()
    epoch_loss = 0
    
    with torch.no_grad() if mode == "TEST" else torch.enable_grad():
        pbar = tqdm(loader, total=len(loader))
        for combined_images, steering_angles in pbar:
            combined_images = combined_images.to(device)
            steering_angles = steering_angles.to(device)
            
            if mode == "TRAIN":
                optimizer.zero_grad()
            
            outputs = model(combined_images).squeeze(1)
            loss = criterion(outputs, steering_angles)
            
            if mode == "TRAIN":
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(loader)
