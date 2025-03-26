import matplotlib.pyplot as plt
from matplotlib import animation
import os
import numpy as np

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

    print(f"Saving an animation of the first {num_batches} batches at {time_per_frame} ms per frame, to {save_path}")
    frames = []  # for storing the generated images
    # Create 2 subplots (one for each channel)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    axes[0].set_title("APS Images")
    axes[1].set_title("DVS Images")
    axes[2].set_title("Stacked APS + DVS Images")
    for i, (combined_images, steering_angles) in enumerate(data_loader):
        if i >= num_batches:
            break
        b, t, c, h, w = combined_images.shape
        combined_images = combined_images.view(-1, c, h, w)
        # Extract the images for each channel
        aps_images = combined_images[:, 0, :, :].cpu().numpy()
        dvs_images = combined_images[:, 1, :, :].cpu().numpy()
        # denormalise values, these are calculated at show_data_histogram.ipynb
        aps_mean, aps_std = 0.4827, 0.3458
        dvs_mean, dvs_std = 0.4672, 0.1243
        aps_images = np.clip((aps_images * aps_std) + aps_mean, 0.0, 1.0)
        dvs_images = np.clip((dvs_images * dvs_std) + dvs_mean, 0.0, 1.0)

        # b * t, h, w, 3
        faux_rgb = np.zeros((aps_images.shape[0], aps_images.shape[1], aps_images.shape[2], 3))
        faux_rgb[:, :, :, 0] = aps_images
        faux_rgb[:, :, :, 1] = dvs_images
        # blue channel remains zeros
        steering_angles = steering_angles.flatten().cpu().numpy()

        for j in range(aps_images.shape[0]):
            im0 = axes[0].imshow(aps_images[j], cmap="Reds", animated=True, origin="lower")
            im1 = axes[1].imshow(dvs_images[j], cmap="Greens", animated=True, origin="lower")                
            im2 = axes[2].imshow(faux_rgb[j], animated=True, origin="lower")
            # Position text centered (x=0.5) and below (y=-0.15) the image
            angle_text = axes[1].text(0.5, -0.15, f"Steering Angle: {180 * steering_angles[j]:.2f}°", 
                                    size=14, ha="center", va="center", 
                                    transform=axes[1].transAxes)
            frames.append([im0, im1, im2, angle_text])
            
    plt.tight_layout()
    print(f"Created {len(frames)} frames. Attempting to save...")
    ani = animation.ArtistAnimation(fig, frames, interval=time_per_frame, blit=True, repeat_delay=1000)
    ani.save(save_path, dpi=160, writer='pillow')
    if os.path.exists(save_path):
        print(f"Success! Animation saved to {save_path} ({os.path.getsize(save_path)/1024:.2f} KB)")
    else:
        print(f"Warning: Animation file was not created at {save_path}")