import matplotlib.pyplot as plt
from matplotlib import animation
import os

import torch
import torch.nn.functional as F

def get_default_3d_conv_kernel():
    # Create 3D-ResNet18-style kernel (typically 3×7×7 or 7×7×7)
    # Using 3×7×7 with temporal gradient detection for 2 input channels
    kernel_3d = torch.zeros(3, 7, 7, dtype=torch.float32)
    
    # First temporal slice - edge detection
    sobel_xy = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
    kernel_3d[0, 2:5, 2:5] = sobel_xy * 0.5
    
    # Middle temporal slice - enhance features
    kernel_3d[1, 2:5, 2:5] = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=torch.float32)
    
    # Last temporal slice - opposite of first for temporal differences
    kernel_3d[2, 2:5, 2:5] = sobel_xy * -0.5
    
    # Create second channel kernel - slightly different
    kernel_3d_2 = kernel_3d.clone()
    kernel_3d_2[1, 2:5, 2:5] = torch.tensor([[1, 0, 1], [0, 4, 0], [1, 0, 1]], dtype=torch.float32)

    # Stack for proper 3D-ResNet shape [out_channels, in_channels, t, h, w]
    # Using just 1 output channel for visualization
    return torch.stack([kernel_3d, kernel_3d_2], dim=0).unsqueeze(0)

def get_default_2d_conv_kernel():
    # Create ResNet18-style kernel (7x7, first layer)
    # Using Sobel-like filters for visualization with 2 input channels (APS, DVS)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # Create 7x7 kernel with Sobel-like pattern in the middle
    kernel_2d = torch.zeros(7, 7, dtype=torch.float32)
    kernel_2d[2:5, 2:5] = sobel_x
    
    # Second channel kernel - slightly different pattern
    kernel_2d_2 = torch.zeros(7, 7, dtype=torch.float32)
    kernel_2d_2[2:5, 2:5] = sobel_y

    # Combine into proper ResNet18 first layer shape [out_channels, in_channels, h, w]
    # Using just 1 output channel for visualization
    return torch.stack([kernel_2d, kernel_2d_2], dim=0).unsqueeze(0)

def apply_default_2d_conv_kernel(combined_image, conv_kernel_2d):
     # 2D Convolution - ResNet18-style
    pad_size_2d = conv_kernel_2d.shape[-1] // 2
    padded_input_2d = F.pad(combined_image, (pad_size_2d, pad_size_2d, pad_size_2d, pad_size_2d))
    conv_result_2d = F.conv2d(padded_input_2d, conv_kernel_2d)
    
    # Normalize 2D result for visualization
    return (conv_result_2d - conv_result_2d.min()) / (conv_result_2d.max() - conv_result_2d.min() + 1e-8)
    
def apply_default_3d_conv_kernel(temporal_buffer, conv_kernel_3d):
    """Apply 3D convolution to the temporal buffer"""
    # Stack temporal frames [1, 2, T, H, W]
    temporal_stack = torch.cat(temporal_buffer, dim=0).unsqueeze(0)
    temporal_stack = temporal_stack.permute(0, 2, 1, 3, 4)  # Reorder to [1, C, T, H, W]
    
    # Apply 3D convolution
    pad_size_3d = conv_kernel_3d.shape[-1] // 2
    padded_input_3d = F.pad(temporal_stack, 
                          (pad_size_3d, pad_size_3d, pad_size_3d, pad_size_3d, 0, 0))
    conv_result_3d = F.conv3d(padded_input_3d, conv_kernel_3d)
    
    # Take middle frame for visualization
    conv_result_3d = conv_result_3d[0, 0]  # Shape: [H, W]
    
    # Normalize 3D result
    r = (conv_result_3d - conv_result_3d.min()) / (conv_result_3d.max() - conv_result_3d.min() + 1e-8)
    return r.squeeze(0)

def animate_loader(data_loader, save_path, global_rank, num_batches=10, time_per_frame=200):
    '''
    Animates the first num_batches of the data_loader.
    args:
        data_loader (torch.utils.data.DataLoader): The data loader to animate
        save_path (str): The path to save the animation as animation.gif
        global_rank (int): The rank of the current process
        num_batches (int): The number of batches to animate
        time_per_frame (int): The time in ms per frame
    '''
    save_path += f"/gpu{global_rank}_animation.gif"

    frames = []  # for storing the generated images
    # Create 2 subplots (one for each channel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].set_title("APS Images")
    axes[0, 1].set_title("DVS Images")
    axes[1, 0].set_title("2D Convolution")
    axes[1, 1].set_title("3D Convolution")

    conv_kernel_2d = get_default_2d_conv_kernel()
    conv_kernel_3d = get_default_3d_conv_kernel()

    print(f"Saving an animation of the first {num_batches} batches at {time_per_frame} ms per frame, to {save_path}")
    # Loop over the first 100 batches

    temporal_buffer = []
    buffer_size = conv_kernel_3d.shape[2]
    current_3d_result = None

    for i, (combined_images, steering_angles) in enumerate(data_loader):
        if i >= num_batches:
            break
        b, t, c, h, w = combined_images.shape
        combined_images = combined_images.view(-1, c, h, w)
        # Extract the images for each channel
        aps_images = combined_images[:, 0, :, :].cpu()
        dvs_images = combined_images[:, 1, :, :].cpu()
        steering_angles = steering_angles.flatten().cpu().numpy()

        for j in range(dvs_images.shape[0]):
            current_aps = aps_images[j:j+1] # sliced like this to maintain dimensions
            current_dvs = dvs_images[j:j+1]
            current_combined = torch.stack([current_aps, current_dvs], dim=1)

            # Apply 2D Convolution
            conv_result_2d = apply_default_2d_conv_kernel(current_combined, conv_kernel_2d)
            # Apply 3D Convolution if the buffer is ready
            temporal_buffer.append(current_combined)
            if len(temporal_buffer) == buffer_size:
                current_3d_result = apply_default_3d_conv_kernel(temporal_buffer, conv_kernel_3d)
                temporal_buffer = []

            if current_3d_result is not None:
                displayed_3d_result = current_3d_result
                buffer_text = f"Next 3D conv in: {buffer_size-len(temporal_buffer)}/{buffer_size} frames"
            else:
                displayed_3d_result = torch.zeros((h, w), dtype=torch.float32)
                buffer_text = f"Collecting frames: {len(temporal_buffer)}/{buffer_size}"

            im0 = axes[0, 0].imshow(aps_images[j].numpy(), cmap="gray", vmin=0, vmax=1.0, animated=True, origin="lower")
            im1 = axes[0, 1].imshow(dvs_images[j].numpy(), cmap="gray", vmin=0, vmax=1.0, animated=True, origin="lower")                
            im2 = axes[1, 0].imshow(conv_result_2d[0, 0].numpy(), cmap="inferno", animated=True, origin="lower")
            im3 = axes[1, 1].imshow(displayed_3d_result.numpy(), cmap="viridis", animated=True, origin="lower") # same until buffer fills
            buffer_info = axes[1, 1].text(0.5, 0.95, buffer_text, ha='center', va='top', transform=axes[1, 1].transAxes, color='white', fontsize=9)

            # Add steering angle as text on the image
            angle_text = fig.text(0.5, 0.02, f"Steering Angle: {180 * steering_angles[j]:.2f}", size=12, ha="center", color="red")
            # Append the artists as the current frame.
            frames.append([im0, im1, im2, im3, buffer_info, angle_text])
    
    print(f"Created {len(frames)} frames. Attempting to save...")
    ani = animation.ArtistAnimation(fig, frames, interval=time_per_frame, blit=True, repeat_delay=1000)
    ani.save(save_path, dpi=80, writer='pillow')
    if os.path.exists(save_path):
        print(f"Success! Animation saved to {save_path} ({os.path.getsize(save_path)/1024:.2f} KB)")
    else:
        print(f"Warning: Animation file was not created at {save_path}")