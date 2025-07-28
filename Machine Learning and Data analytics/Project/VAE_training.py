import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os
from models import VAE
from dataset import ImagesDataset2D
from tqdm import tqdm
import csv


def vae_loss_function(reconstructed_x, x, mu, log_var, beta=1.0):
    batch_size = x.size(0)
    BCE = F.binary_cross_entropy(
        reconstructed_x.view(batch_size, -1), x.view(batch_size, -1), reduction="sum"
    )
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = (BCE + beta * KLD) / batch_size
    return total_loss


def vae_validation_loss_function(reconstructed_x, x, mu, log_var, beta=1.0):
    batch_size = x.size(0)
    BCE = F.binary_cross_entropy(
        reconstructed_x.view(batch_size, -1), x.view(batch_size, -1), reduction="sum"
    )
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = (BCE + beta * KLD) / batch_size
    return total_loss, BCE / batch_size


def train_vae_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    num_epochs,
    beta_kld=1.0,
    log_interval=10,
):
    pbar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        unit="batch",  # Describes what each iteration step means
    )
    model.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(pbar):
        images = data
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed_x, mu, log_var = model(images)
        loss = vae_loss_function(reconstructed_x, images, mu, log_var, beta=beta_kld)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        current_running_avg_loss = train_loss / (batch_idx + 1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", avg_loss=f"{current_running_avg_loss:.4f}"
        )

    avg_epoch_loss = train_loss / len(train_loader)
    return avg_epoch_loss


def evaluate_vae_epoch(model, val_loader, device, epoch, num_epochs, beta_kld=1.0):
    model.eval()
    val_loss = 0.0
    recon_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            images = data
            images = images.to(device)
            reconstructed_x, mu, log_var = model(images)
            loss, loss_recon = vae_validation_loss_function(
                reconstructed_x, images, mu, log_var, beta=beta_kld
            )
            val_loss += loss.item()
            recon_loss += loss_recon.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_recon_loss = recon_loss / len(val_loader)
    return avg_val_loss, avg_recon_loss


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
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    LATENT_DIM = 128
    BETA_KLD = 1.0
    LOG_INTERVAL = 20
    MODEL_PATH = f"vae_ld{LATENT_DIM}_beta{BETA_KLD}.pth"

    BASE_OUTPUT_DIR = "VAE_hyperparameter_study"
    model_identifier = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    output_directory = os.path.join(BASE_OUTPUT_DIR, model_identifier)

    os.makedirs(output_directory, exist_ok=True)

    csv_log_path = os.path.join(output_directory, "training_log_simple.csv")
    csv_header = ["Epoch", "Train_Loss", "Val_Loss", "Recon_Loss"]

    with open(csv_log_path, "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(csv_header)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_model = VAE(latent_dim=LATENT_DIM).to(device)

    optimizer = optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)

    image_transforms = transforms.Compose(
        [
            transforms.Resize((101, 101)),
            transforms.ToTensor(),
            transforms.RandomRotation((-180, 180)),
        ]
    )

    train_data_path = "2D_square/train_images/"
    val_data_path = "2D_square/val_images/"

    train_dataset = ImagesDataset2D(
        data_root=train_data_path, transform=image_transforms
    )
    val_dataset = ImagesDataset2D(data_root=val_data_path, transform=image_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
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
            log_interval=LOG_INTERVAL,
        )
        val_loss, recon_loss = evaluate_vae_epoch(
            vae_model, val_loader, device, epoch, NUM_EPOCHS, beta_kld=BETA_KLD
        )

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
            f"{recon_loss:.4f}",
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
