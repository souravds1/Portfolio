import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os
from models import VAE, LDM
from dataset import ImagesDataset2D
from tqdm import tqdm
import csv


def train_ldm_epoch(ldm_model, epoch, num_epochs, train_loader, optimizer_ldm, device):
    pbar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        unit="batch",
    )
    ldm_model.unet.train()
    total_loss = 0.0

    for batch_idx, data_batch in enumerate(pbar):
        images = data_batch
        images = images.to(device)
        optimizer_ldm.zero_grad()
        loss = ldm_model.get_loss(images)
        loss.backward()
        optimizer_ldm.step()
        total_loss += loss.item()
        current_running_avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", avg_loss=f"{current_running_avg_loss:.4f}"
        )

    return total_loss / len(train_loader)


def evaluate_ldm_epoch(ldm_model, val_loader, device, val_t_step_for_recon=None):
    ldm_model.eval()

    total_noise_loss = 0.0
    total_recon_loss = 0.0

    num_batches = len(val_loader)

    for data_batch in val_loader:
        images = data_batch.to(device)

        noise_loss = ldm_model.get_loss(images)
        total_noise_loss += noise_loss.item()

        recon_loss = ldm_model.get_val_reconstruction_metrics(
            images, val_t_step=val_t_step_for_recon
        )
        total_recon_loss += recon_loss.item()

    avg_noise_loss = total_noise_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches

    return avg_noise_loss, avg_recon_loss


@torch.no_grad()
def generate_samples_from_ldm(ldm_model, num_samples, device, generation_seed=None):
    if generation_seed is not None:
        torch.manual_seed(generation_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(generation_seed)

    ldm_model.eval()

    generated_images_batch = ldm_model.sample(batch_size=num_samples, device=device)

    processed_tensors = []
    for i in range(generated_images_batch.size(0)):
        img_tensor = generated_images_batch[i]
        processed_tensor = img_tensor.cpu()
        processed_tensors.append(processed_tensor)
    return processed_tensors


if __name__ == "__main__":
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 32
    LATENT_DIM = 128
    BETA_KLD = 1.0
    LOG_INTERVAL = 20

    MODEL_PATH = f"ldm_ld{LATENT_DIM}_beta{BETA_KLD}.pth"

    BASE_OUTPUT_DIR = "LDM_hyperparameter_study"

    model_identifier = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    output_directory = os.path.join(BASE_OUTPUT_DIR, model_identifier)
    LDM_UNET_SAVE_PATH = os.path.join(
        output_directory, f"{model_identifier}_best_unet.pth"
    )

    os.makedirs(output_directory, exist_ok=True)

    csv_log_path = os.path.join(output_directory, "training_log_simple.csv")
    csv_header = ["Epoch", "Train_Loss", "Val_Loss", "Recon_Loss"]

    with open(csv_log_path, "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(csv_header)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LDM_UNET_LATENT_RESHAPE_CHANNELS = 8
    LDM_UNET_TIME_EMB_DIM = 256
    LDM_UNET_INIT_CHANNELS = 64
    NUM_TIMESTEPS = 1000
    BETA_START = 0.0001
    BETA_END = 0.02

    VAE_MODEL_PATH = "VAE_hyperparameter_study/vae_ld128_beta1.0/vae_ld128_beta1.0.pth"

    loaded_vae_model = VAE(latent_dim=LATENT_DIM).to(device)
    loaded_vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))

    loaded_vae_model.eval()

    ldm_model = LDM(
        vae_model=loaded_vae_model,
        latent_dim=LATENT_DIM,
        ldm_unet_latent_reshape_channels=LDM_UNET_LATENT_RESHAPE_CHANNELS,
        ldm_unet_time_emb_dim=LDM_UNET_TIME_EMB_DIM,
        ldm_unet_init_channels=LDM_UNET_INIT_CHANNELS,
        num_timesteps=NUM_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
    ).to(device)

    optimizer_ldm = optim.Adam(ldm_model.unet.parameters(), lr=LEARNING_RATE)

    image_transforms_ldm = transforms.Compose(
        [
            transforms.Resize((101, 101)),
            transforms.ToTensor(),
            transforms.RandomRotation((-180, 180)),
        ]
    )

    train_data_path_ldm = "2D_square/train_images/"
    val_data_path_ldm = "2D_square/val_images/"
    train_dataset_ldm = ImagesDataset2D(
        data_root=train_data_path_ldm,
        transform=image_transforms_ldm,
    )
    val_dataset_ldm = ImagesDataset2D(
        data_root=val_data_path_ldm,
        transform=image_transforms_ldm,
    )
    train_loader_ldm = DataLoader(
        train_dataset_ldm,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=(True if len(train_dataset_ldm) > BATCH_SIZE else False),
    )
    val_loader_ldm = DataLoader(
        val_dataset_ldm,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True if len(val_dataset_ldm) > BATCH_SIZE else False,
    )

    best_val_loss_ldm = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss_ldm = train_ldm_epoch(
            ldm_model, epoch, NUM_EPOCHS, train_loader_ldm, optimizer_ldm, device
        )
        val_loss_ldm, recon_loss = evaluate_ldm_epoch(ldm_model, val_loader_ldm, device)
        print(
            f"LDM Epoch [{epoch}/{NUM_EPOCHS}], Train Loss: {train_loss_ldm:.4f}, Val Loss: {val_loss_ldm:.4f}"
        )
        log_data_row = [
            epoch,
            f"{train_loss_ldm:.4f}",
            f"{val_loss_ldm:.4f}",
            f"{recon_loss:.4f}",
        ]
        with open(csv_log_path, "a", newline="") as f_csv:
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(log_data_row)

        if val_loss_ldm < best_val_loss_ldm:
            best_val_loss_ldm = val_loss_ldm
            torch.save(ldm_model.unet.state_dict(), LDM_UNET_SAVE_PATH)

    NUM_SAMPLES_TO_GENERATE = 20
    SAMPLING_OUTPUT_DIRECTORY = os.path.join(output_directory, "generated_samples")
    os.makedirs(SAMPLING_OUTPUT_DIRECTORY, exist_ok=True)
    BINARIZATION_THRESHOLD = 0.5
    SAMPLING_GENERATION_SEED = 42

    ldm_model.unet.load_state_dict(torch.load(LDM_UNET_SAVE_PATH, map_location=device))

    ldm_model.eval()

    generated_image_tensors = generate_samples_from_ldm(
        ldm_model,
        NUM_SAMPLES_TO_GENERATE,
        device,
        generation_seed=SAMPLING_GENERATION_SEED,
    )

    for i, img_tensor_chw in enumerate(generated_image_tensors):
        binarized_tensor_chw = (img_tensor_chw > BINARIZATION_THRESHOLD).float()

        pil_image = transforms.ToPILImage()(binarized_tensor_chw.cpu())
        pil_image = pil_image.convert("1")

        num_digits = len(str(NUM_SAMPLES_TO_GENERATE))
        filename = f"ldm_sample_{str(i+1).zfill(num_digits)}_binary.png"
        save_path = os.path.join(SAMPLING_OUTPUT_DIRECTORY, filename)
        pil_image.save(save_path)
