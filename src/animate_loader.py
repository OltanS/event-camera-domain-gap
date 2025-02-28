import matplotlib.pyplot as plt
from matplotlib import animation
import os

def animate_loader(data_loader, save_path, global_rank, num_batches=10, time_per_frame=200):
    save_path += f"/gpu{global_rank}_animation.gif"

    frames = []  # for storing the generated images
    # Create 2 subplots (one for each channel)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("APS Images")
    axes[1].set_title("DVS Images")
    
    print(f"Saving an animation of the first {num_batches} batches at {time_per_frame} ms per frame, to {save_path}")
    # Loop over the first 100 batches
    for i, (combined_images, steering_angles) in enumerate(data_loader):
        if i >= num_batches:
            break
        b, t, c, w, h = combined_images.shape
        combined_images = combined_images.view(-1, c, w, h)
        # Extract the images for each channel
        aps_images = combined_images[:, 0, :, :].cpu().numpy()  # converting tensor to numpy if needed
        dvs_images = combined_images[:, 1, :, :].cpu().numpy()
        steering_angles = steering_angles.flatten().cpu().numpy()
        # Display images on the subplots (set animated=True for animation)
        for j in range(dvs_images.shape[0]):
            im0 = axes[0].imshow(dvs_images[j], cmap="gray", vmin=0, vmax=1.0, animated=True, origin="lower")
            im1 = axes[1].imshow(aps_images[j], cmap="gray", vmin=0, vmax=1.0, animated=True, origin="lower")                
            # Add steering angle as text on the image
            angle_text = axes[0].text(0.5, -0.1, f"Steering Angle: {180 * steering_angles[j]:.2f}", 
                                    size=12, ha="center", transform=axes[0].transAxes, color="red")
            # Append the artists as the current frame.
            frames.append([im0, im1, angle_text])
    
    print(f"Created {len(frames)} frames. Attempting to save...")
    ani = animation.ArtistAnimation(fig, frames, interval=time_per_frame, blit=True, repeat_delay=1000)
    ani.save(save_path, dpi=80, writer='pillow')
    if os.path.exists(save_path):
        print(f"Success! Animation saved to {save_path} ({os.path.getsize(save_path)/1024:.2f} KB)")
    else:
        print(f"Warning: Animation file was not created at {save_path}")