import gc

from tqdm import tqdm

import torch
import contextlib

def _inspect_gradients(model):
    '''
    Utility for monitoring exploding gradients
    '''
    print("\n=== Parameter Gradient Analysis ===")
    
    # Track the largest gradients
    param_dict = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.detach().norm(2).item()
            param_dict[name] = grad_norm
            
    # Sort by gradient norm (highest first)
    sorted_grads = sorted(param_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 100 largest gradients:")
    for name, norm in sorted_grads[:100]:
        param_size = param_dict = ' x '.join(str(s) for s in model.get_parameter(name).shape)
        print(f"{name:40} | shape: {param_size:15} | grad norm: {norm:.2f}")
    
    return sorted_grads

def _classify_parameter(name):
    """Classify a parameter name into a meaningful layer type."""
    parts = name.split('.')
    
    # More robust prefix handling
    start_idx = 0
    if "module" in parts:
        start_idx += 1
    if "model" in parts:
        start_idx += 1
        
    # Safety check
    if len(parts) <= start_idx:
        return "other"
        
    # More robust classification with defensive programming
    try:
        if 'patch_embed' in name:
            return 'patch_embed'
        elif 'head' in name:
            return 'head'
        elif 'features' in name and len(parts) > start_idx + 1:
            block_num = parts[start_idx + 1]
            
            # Check for sufficient parts before indexing
            if len(parts) > start_idx + 3:
                component = parts[start_idx + 3]
                if component == 'attn' and len(parts) > start_idx + 4:
                    return f"block{block_num}.attn.{parts[start_idx + 4]}"
                elif component == 'attn':
                    return f"block{block_num}.attn"
                elif component == 'mlp':
                    return f"block{block_num}.mlp"
                elif component == 'reduction':
                    return f"block{block_num}.reduction"
                else:
                    return f"block{block_num}.{component}"
            else:
                return f"block{block_num}"
        else:
            return parts[start_idx]
    except IndexError:
        return "unknown"  # Fallback for unexpected formats

def _track_gradient_stats(model):
    """Track statistical measures of gradients by component type"""
    grad_norms = []
    grad_stats_by_layer_type = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.detach().norm(2).item()
            grad_norms.append(grad_norm)
            
            # Use the helper function to classify the parameter
            layer_type = _classify_parameter(name)
            
            if layer_type not in grad_stats_by_layer_type:
                grad_stats_by_layer_type[layer_type] = []
            grad_stats_by_layer_type[layer_type].append((name, grad_norm))
    
    stats = {}
    if grad_norms:
        # Calculate median properly
        sorted_norms = sorted(grad_norms)
        n = len(sorted_norms)
        if n % 2 == 0:  # Even length
            median = (sorted_norms[n//2 - 1] + sorted_norms[n//2]) / 2
        else:  # Odd length
            median = sorted_norms[n//2]
        
        stats['overall'] = {
            'mean': sum(grad_norms) / len(grad_norms),
            'median': median,
            'max': max(grad_norms),
            'min': min(grad_norms)
        }
        
        # Calculate stats for each layer type
        for layer_type, name_norm_pairs in grad_stats_by_layer_type.items():
            norms = [norm for _, norm in name_norm_pairs]
            stats[layer_type] = {
                'mean': sum(norms) / len(norms),
                'max': max(norms),
                'count': len(norms)
            }
            
            # Add top contributor for this layer type
            top_contributor = max(name_norm_pairs, key=lambda x: x[1])
            stats[layer_type]['top_param'] = top_contributor[0]
            stats[layer_type]['top_norm'] = top_contributor[1]
    
    return stats

def _log_gradient_stats(writer, grad_stats, mode, loader, epoch, batch_num):
    step = epoch * len(loader) + batch_num
    
    # Log overall statistics
    for stat_name, value in grad_stats['overall'].items():
        writer.add_scalar(f'gradients/{mode}/overall_{stat_name}', 
                         value, global_step=step)

    # Log per-component statistics and highlight top contributors
    for layer_type, layer_stats in grad_stats.items():
        if layer_type != 'overall':
            for stat_name, value in layer_stats.items():
                if stat_name not in ['top_param', 'top_norm']:  # Skip these for scalar logging
                    writer.add_scalar(f'gradients/{mode}/{layer_type}_{stat_name}', 
                                     value, global_step=step)
    
    # Optionally log top 5 largest gradient norms as text
    if writer.add_text and 'overall' in grad_stats:
        # Collect all (param_name, norm) pairs
        all_params = []
        for layer_type, stats in grad_stats.items():
            if layer_type != 'overall' and 'top_param' in stats:
                all_params.append((stats['top_param'], stats['top_norm']))
        
        # Sort and get top 5
        top5 = sorted(all_params, key=lambda x: x[1], reverse=True)[:5]
        text = "\n".join([f"{name}: {norm:.2f}" for name, norm in top5])
        writer.add_text(f'gradients/{mode}/top5', text, global_step=step)
                    
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
        no_time_dim_models = ["Resnet", "ResnetCBAM", "ConvNeXt", "DRFuser"]
        bcthw_models = ["Simple", "Swin", "MViTV2", "MViTV2Parallel","R3d", "R3dParallel", "R2Plus1d", "R2Plus1dCBAM", "R2Plus1dtanh"]
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
                b, t, c, h, w = combined_images.shape
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # for transformer models, we need to reshuffle the dimensions to their expected
                    # to b, c, t, h, w from the original b, t, c, h ,w
                    if model_type in bcthw_models:
                        combined_images = combined_images.permute(0, 2, 1, 3, 4)
                    # for non-time models, we need to reshape the input to b * t, c, h, w
                    # furthermore, we must have the output compare all the time slots instead of just the last one
                    elif model_type in no_time_dim_models:
                        combined_images = combined_images.reshape(b * t, c, h, w)
                        # after this line steering_angle[:, -1] will be all the time slots
                        steering_angles = steering_angles.reshape(b * t, 1)
                    
                    outputs = model(combined_images).squeeze(1)
                    # only consider the output for the last time slot
                    # take the rest of the input as "context"

                    target_angles = steering_angles[:, -1]
                    loss = criterion(outputs, target_angles)
                    
                if mode != "TEST":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # inspect_gradients(model)
                    if writer is not None and global_rank ==0 and i % 20 == 0:
                        grad_stats = _track_gradient_stats(model)
                        _log_gradient_stats(writer, grad_stats, mode, loader, epoch, i)
                    
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
