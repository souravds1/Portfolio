import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from models import VAE
from dataset import ImagesDataset2D, NoisyImagesDataset2D
from tqdm import tqdm
import csv


def vae_loss_function(reconstructed_x, clean_x, mu, log_var, beta=1.0):
    batch_size = clean_x.size(0)
    # reconstructed_x = torch.clamp(reconstructed_x, 0.0, 1.0)
    BCE = F.binary_cross_entropy(
        reconstructed_x.view(batch_size, -1),
        clean_x.view(batch_size, -1),
        reduction="sum",
    )
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = (BCE + beta * KLD) / batch_size
    return total_loss


def train_vae_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    num_epochs,
    beta_kld=1.0,
):
    pbar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        unit="batch",
    )
    model.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(pbar):
        x, clean_x = data
        x, clean_x = x.to(device), clean_x.to(device)
        optimizer.zero_grad()
        reconstructed_x, mu, log_var = model(x)
        loss = vae_loss_function(reconstructed_x, clean_x, mu, log_var, beta=beta_kld)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        current_running_avg_loss = train_loss / (batch_idx + 1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", avg_loss=f"{current_running_avg_loss:.4f}"
        )

    avg_epoch_loss = train_loss / len(train_loader)
    return avg_epoch_loss


def evaluate_vae_epoch(model, val_loader, device, beta_kld=1.0):
    model.eval()
    val_loss = 0.0

    batch = next(iter(val_loader))
    fixed_samples = {"noisy": batch[0][:4].cpu(), "original": batch[1][:4].cpu()}

    with torch.no_grad():
        for data in val_loader:
            x, clean_x = data
            x, clean_x = x.to(device), clean_x.to(device)
            reconstructed_x, mu, log_var = model(x)
            loss = vae_loss_function(
                reconstructed_x, clean_x, mu, log_var, beta=beta_kld
            )
            val_loss += loss.item()

        fixed_reconstructed, _, _ = model(fixed_samples["noisy"].to(device))

    return val_loss / len(val_loader), {
        "noisy": fixed_samples["noisy"],
        "original": fixed_samples["original"],
        "reconstructed": fixed_reconstructed.cpu(),
    }


def visualize_reconstructions(samples, epoch, save_path="./results"):
    os.makedirs(save_path, exist_ok=True)

    num_samples = len(samples["noisy"])
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

    row_data = [
        ("Noisy Input", "noisy"),
        ("Clean Target", "original"),
        ("Reconstructed", "reconstructed"),
    ]

    for row, (label, image_type) in enumerate(row_data):
        for col in range(num_samples):
            ax = axes[row, col]
            ax.imshow(samples[image_type][col].squeeze(), cmap="gray")
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(label)

    plt.tight_layout()
    plt.savefig(f"{save_path}/denoising_progress_epoch_{epoch}.png")
    plt.close("all")


@torch.no_grad()
def generate_samples_from_vae(
    vae_model, latent_dim, num_samples, device, generation_seed=None
):
    if generation_seed is not None:
        torch.manual_seed(generation_seed)
    vae_model.eval()
    generated_images_tensors = []
    for _ in range(num_samples):
        z = torch.randn(1, latent_dim).to(device)
        generated_image_tensor = vae_model.decoder(z)
        processed_tensor = generated_image_tensor.squeeze(0).squeeze(0).cpu()
        generated_images_tensors.append(processed_tensor)
    return generated_images_tensors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flip",
        type=float,
        default=0.05,
        help="A float with default value (default: 0.05)",
    )
    args = parser.parse_args()
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    LATENT_DIM = 128
    BETA_KLD = 1.0
    LOG_INTERVAL = 20
    FLIP_PROBABILITY = args.flip
    MODEL_PATH = f"high_freq_noise_vae_ld{LATENT_DIM}_beta{BETA_KLD}.pth"

    BASE_OUTPUT_DIR = "VAE_freq_noise_study"
    model_identifier = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    output_directory = os.path.join(BASE_OUTPUT_DIR, model_identifier)

    os.makedirs(output_directory, exist_ok=True)

    csv_log_path = os.path.join(output_directory, "training_log_simple.csv")
    csv_header = ["Epoch", "Train_Loss", "Val_Loss"]

    with open(csv_log_path, "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(csv_header)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_model = VAE(latent_dim=LATENT_DIM).to(device)

    optimizer = optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)

    train_image_transforms = transforms.Compose(
        [
            transforms.Resize((101, 101)),
            transforms.ToTensor(),
            transforms.RandomRotation((-180, 180)),
        ]
    )
    val_image_transforms = transforms.Compose(
        [
            transforms.Resize((101, 101)),
            transforms.ToTensor(),
        ]
    )

    train_data_path = "2D_square/train_images/"
    val_data_path = "2D_square/val_images/"

    # train_dataset = NoisyImagesDataset2D(
    #     data_root=train_data_path,
    #     noise_type="grid_binary",
    #     grid_freq=7,
    #     grid_amplitude=1,
    #     grid_rotation=45,
    #     transform=transforms.ToTensor(),
    # )
    # val_dataset = NoisyImagesDataset2D(
    #     data_root=val_data_path,
    #     noise_type="grid_binary",
    #     grid_freq=7,
    #     grid_amplitude=1,
    #     grid_rotation=45,
    #     transform=transforms.ToTensor(),
    # )

    train_dataset = NoisyImagesDataset2D(
        data_root=train_data_path,
        noise_type="frequency_binary",  # Add "_binary" for binary images
        freq_mask_type="rings",  # Options: "rings", "wedges", "blocks", "random", "highpass", "lowpass"
        freq_mask_intensity=1.0,  # How strongly to modify frequencies (0-1)
        freq_mask_width=0.1,  # Width of the frequency structures
        transform=transforms.ToTensor(),
    )

    val_dataset = NoisyImagesDataset2D(
        data_root=val_data_path,
        noise_type="frequency_binary",  # Add "_binary" for binary images
        freq_mask_type="rings",  # Options: "rings", "wedges", "blocks", "random", "highpass", "lowpass"
        freq_mask_intensity=1.0,  # How strongly to modify frequencies (0-1)
        freq_mask_width=0.1,  # Width of the frequency structures
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_vae_epoch(
            vae_model,
            train_loader,
            optimizer,
            device,
            epoch,
            NUM_EPOCHS,
            beta_kld=BETA_KLD,
        )
        val_loss, recon_images = evaluate_vae_epoch(
            vae_model, val_loader, device, beta_kld=BETA_KLD
        )

        if epoch % 10 == 0:
            visualize_reconstructions(recon_images, epoch, save_path=output_directory)

        if val_loss < best_val_loss:
            torch.save(
                vae_model.state_dict(), os.path.join(output_directory, MODEL_PATH)
            )
            print(
                f"Epoch {epoch}: Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f})"
            )
            best_val_loss = val_loss

        log_data_row = [
            epoch,
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
        ]

        with open(csv_log_path, "a", newline="") as f_csv:
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(log_data_row)

    print(
        f"Best validation loss: {best_val_loss:.4f}. Best model saved to {MODEL_PATH}"
    )

    NUM_SAMPLES_TO_GENERATE = 20
    GENERATION_THRESHOLD = 0.5
    GENERATION_SEED = 42

    new_microstructure_tensors = generate_samples_from_vae(
        vae_model,
        LATENT_DIM,
        NUM_SAMPLES_TO_GENERATE,
        device,
        generation_seed=GENERATION_SEED,
    )

    for i, img_tensor in enumerate(new_microstructure_tensors):
        binarized_tensor = (img_tensor > GENERATION_THRESHOLD).float()
        num_digits = len(str(NUM_SAMPLES_TO_GENERATE))
        filename = f"sample_{str(i+1).zfill(num_digits)}_binary.png"
        save_path = os.path.join(output_directory, filename)
        pil_image = transforms.ToPILImage()(binarized_tensor.unsqueeze(0))
        pil_image = pil_image.convert("1")
        pil_image.save(save_path)

    print(
        f"All {len(new_microstructure_tensors)} binarized samples saved to {output_directory}"
    )
