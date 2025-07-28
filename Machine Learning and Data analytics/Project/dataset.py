from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np


class ImagesDataset2D(Dataset):

    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.image_files = [os.path.join(data_root, f) for f in os.listdir(data_root)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        image = Image.open(img_path).convert("1")
        if self.transform:
            image = self.transform(image)
        return image


class NoisyImagesDataset2D(Dataset):
    def __init__(
        self,
        data_root,
        noise_type="salt_pepper",
        flip_probability=0.05,
        grid_freq=10,
        grid_amplitude=0.5,
        grid_rotation=0,
        freq_mask_type="rings",
        freq_mask_intensity=0.8,
        freq_mask_width=0.1,
        transform=None,
    ):
        self.data_root = data_root
        self.transform = transform
        self.flip_probability = flip_probability
        self.noise_type = noise_type

        # Grid noise parameters
        self.grid_freq = grid_freq
        self.grid_amplitude = grid_amplitude
        self.grid_rotation = grid_rotation

        # Frequency domain noise parameters
        self.freq_mask_type = freq_mask_type  # "rings", "wedges", "blocks", "random"
        self.freq_mask_intensity = (
            freq_mask_intensity  # 0-1, how much to suppress frequencies
        )
        self.freq_mask_width = freq_mask_width  # Width of the frequency structures

        self.image_files = [os.path.join(data_root, f) for f in os.listdir(data_root)]

    def __len__(self):
        return len(self.image_files)

    def add_binary_noise(self, tensor_image):
        """Add salt and pepper noise by flipping bits with probability flip_probability"""
        flip_mask = torch.bernoulli(
            torch.ones_like(tensor_image) * self.flip_probability
        ).bool()
        noisy_image = tensor_image.clone()
        noisy_image[flip_mask] = 1 - noisy_image[flip_mask]
        return noisy_image

    def add_grid_noise(self, tensor_image):
        """Add structured grid noise to the image"""
        # Get image dimensions
        _, h, w = tensor_image.shape

        # Create coordinate grids
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        # Apply rotation if specified
        if self.grid_rotation != 0:
            # Center coordinates
            grid_x_centered = grid_x - 0.5
            grid_y_centered = grid_y - 0.5

            # Convert rotation to radians
            theta = np.radians(self.grid_rotation)

            # Rotation matrix application
            grid_x_rot = (
                grid_x_centered * np.cos(theta) - grid_y_centered * np.sin(theta) + 0.5
            )
            grid_y_rot = (
                grid_x_centered * np.sin(theta) + grid_y_centered * np.cos(theta) + 0.5
            )

            grid_x, grid_y = grid_x_rot, grid_y_rot

        # Generate grid pattern
        pattern = torch.sin(2 * np.pi * self.grid_freq * grid_x) * torch.sin(
            2 * np.pi * self.grid_freq * grid_y
        )

        # Reshape to match tensor_image
        pattern = pattern.unsqueeze(0) * self.grid_amplitude

        # Apply noise
        noisy_image = tensor_image.clone()

        # For binary images (0,1), we need to handle the noise differently
        if "binary" in self.noise_type:
            # Add pattern then threshold back to binary (0,1)
            temp = noisy_image + pattern
            noisy_image = (temp > 0.5).float()
        else:
            # For non-binary case, add noise and clip to valid range [0,1]
            noisy_image = torch.clamp(noisy_image + pattern, 0, 1)

        return noisy_image

    def add_frequency_noise(self, tensor_image):
        """Add noise in the frequency domain using FFT"""
        # Get dimensions
        _, h, w = tensor_image.shape

        # Convert to numpy for FFT (could use torch.fft but this is clearer)
        image_np = tensor_image.squeeze().numpy()

        # Apply FFT
        fft_image = np.fft.fft2(image_np)
        fft_shifted = np.fft.fftshift(fft_image)  # Shift zero frequency to center

        # Create frequency domain mask based on mask_type
        mask = np.ones((h, w), dtype=np.float32)
        center_y, center_x = h // 2, w // 2

        # Calculate distance from center for each point
        y_grid, x_grid = np.ogrid[:h, :w]
        distances = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

        # Normalize distances to 0-1 range
        max_distance = np.sqrt(center_x**2 + center_y**2)
        distances_norm = distances / max_distance

        if self.freq_mask_type == "rings":
            # Create ring pattern
            num_rings = 5
            for i in range(num_rings):
                ring_center = (i + 1) / (num_rings + 1)  # Position rings evenly
                ring_mask = np.abs(distances_norm - ring_center) < self.freq_mask_width
                mask[ring_mask] = 1 - self.freq_mask_intensity

        elif self.freq_mask_type == "wedges":
            # Create angular wedges
            angles = (
                np.arctan2(y_grid - center_y, x_grid - center_x) + np.pi
            )  # 0 to 2pi
            angles_norm = angles / (2 * np.pi)

            num_wedges = 8
            for i in range(num_wedges):
                wedge_center = i / num_wedges
                wedge_width = 1 / (num_wedges * 2)

                # Handle wraparound at 2pi
                distance = np.minimum(
                    np.abs(angles_norm - wedge_center),
                    np.abs(angles_norm - (wedge_center + 1)),
                )
                wedge_mask = distance < wedge_width
                mask[wedge_mask] = 1 - self.freq_mask_intensity

        elif self.freq_mask_type == "blocks":
            # Create block pattern in frequency domain
            block_size = int(w * self.freq_mask_width)
            for i in range(0, h, block_size * 2):
                for j in range(0, w, block_size * 2):
                    if i + block_size < h and j + block_size < w:
                        mask[i : i + block_size, j : j + block_size] = (
                            1 - self.freq_mask_intensity
                        )

        elif self.freq_mask_type == "random":
            # Random frequency masking
            random_mask = np.random.rand(h, w) < self.freq_mask_width
            mask[random_mask] = 1 - self.freq_mask_intensity

        elif self.freq_mask_type == "highpass":
            # High-pass filter (remove low frequencies)
            high_pass_radius = self.freq_mask_width * max_distance
            mask[distances < high_pass_radius] = 1 - self.freq_mask_intensity

        elif self.freq_mask_type == "lowpass":
            # Low-pass filter (remove high frequencies)
            low_pass_radius = (1 - self.freq_mask_width) * max_distance
            mask[distances > low_pass_radius] = 1 - self.freq_mask_intensity

        # Apply mask to frequency domain
        fft_filtered = fft_shifted * mask

        # Inverse FFT
        ifft_shifted = np.fft.ifftshift(fft_filtered)
        image_filtered = np.fft.ifft2(ifft_shifted)
        image_filtered = np.real(image_filtered)

        # Normalize back to 0-1 range
        image_filtered = np.clip(image_filtered, 0, 1)

        # Convert back to tensor
        result = torch.from_numpy(image_filtered).float().unsqueeze(0)

        # Handle binary images
        if "binary" in self.noise_type:
            result = (result > 0.5).float()

        return result

    def __getitem__(self, index):
        img_path = self.image_files[index]
        image = Image.open(img_path).convert("1")  # Load as binary image

        if self.transform:
            image = self.transform(image)

        clean_image = image.clone()

        # Apply noise based on the specified type
        if self.noise_type == "salt_pepper":
            image = self.add_binary_noise(image)
        elif "grid" in self.noise_type:
            image = self.add_grid_noise(image)
        elif "frequency" in self.noise_type:
            image = self.add_frequency_noise(image)

        return image, clean_image


if __name__ == "__main__":
    import numpy as np

    dataset = ImagesDataset2D(data_root="2D_square/train_images")
    val_dataset = ImagesDataset2D(data_root="2D_square/val_images")
    print(len(dataset), len(val_dataset))
