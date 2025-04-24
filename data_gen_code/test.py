import torch
import os
from PIL import Image
from data_loading import get_dataloaders
from cyclegan_resnet import ResnetGenerator, PatchGANDiscriminator, ImageBuffer

# Define paths
ddi_data_dir = "../data/ddi_cropped"
ham_data_dir = "../data/HAM10000"
scin_data_dir = "../data/dark_scin"
test_image_path = "../data/darkHAM"  # Path to save generated images
model_path = "../results/best_cyclegan_model.pth"  # Path to your saved model
os.makedirs(test_image_path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set batch size and number of workers
batch_size = 1
num_workers = 2
seed = 42

# Load test data using get_dataloaders
ddi_loader_train, ddi_loader_val, ham_loader_train, ham_loader_val, ham_loader_test, scin_loader_train, scin_loader_val = get_dataloaders(
    ddi_data_dir,
    ham_data_dir,
    scin_data_dir,
    batch_size=batch_size,
    num_workers=num_workers,
    seed=seed
)

# Load the best model state
checkpoint = torch.load(model_path)

generate_XtoY = ResnetGenerator().to(device)
generate_YtoX = ResnetGenerator().to(device)
Discriminator_X = PatchGANDiscriminator().to(device)
Discriminator_Y = PatchGANDiscriminator().to(device)

# Load the model weights, ignoring mismatched keys
generate_XtoY.load_state_dict(checkpoint['Generator_XtoY'], strict=False)
generate_YtoX.load_state_dict(checkpoint['Generator_YtoX'], strict=False)
Discriminator_X.load_state_dict(checkpoint['Discriminator_X'], strict=False)
Discriminator_Y.load_state_dict(checkpoint['Discriminator_Y'], strict=False)

generate_XtoY.eval()
generate_YtoX.eval()
Discriminator_X.eval()
Discriminator_Y.eval()

def denormalize_image(image_tensor, dataset_type='LIGHT'):
    """
    Denormalizes the image tensor based on the dataset (HAM or SCIN).
    """
    # Convert tensor to numpy array
    image = image_tensor.cpu().detach().numpy()

    if dataset_type == 'LIGHT':
        mean = np.array([0.7660, 0.5462, 0.5714])  # Mean for HAM eval transforms (light skin)
        std = np.array([0.0872, 0.1171, 0.1318])   # Std for HAM eval transforms (light skin)
    elif dataset_type == 'DARK':
        mean = np.array([0.4543, 0.3487, 0.2936])  # Mean for SCIN dataset (dark skin)
        std = np.array([0.1920, 0.1547, 0.1364])   # Std for SCIN dataset (dark skin)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Denormalize the image (reverse normalization)
    image = image * std[:, None, None] + mean[:, None, None]

    # Ensure the pixel values are between 0 and 1
    image = np.clip(image, 0, 1)
    
    return image

# Function to save generated images
def save_generated_images(ham_loader_test, generate_XtoY, generate_YtoX, new_data_path, device):
    """
    Generate and save images from test data (HAM test set).
    """
    os.makedirs(new_data_path, exist_ok=True)

    with torch.no_grad():
        for i, (real_X, real_X_id) in enumerate(ham_loader_test):
            real_X = real_X.float().to(device)
            real_X_id = real_X_id[0]  # Assuming only one image per batch in test loader

            # Generate Fake Y and Cycle X images
            fake_Y = generate_XtoY(real_X)
            cycle_X = generate_YtoX(fake_Y)

            # Denormalize and save the images
            fake_Y_image = denormalize_image(fake_Y, 'DARK')
            cycle_X_image = denormalize_image(cycle_X, 'LIGHT')

            fake_Y_image = np.transpose(fake_Y_image, (1, 2, 0))
            cycle_X_image = np.transpose(cycle_X_image, (1, 2, 0))

            fake_Y_image = fake_Y_image.squeeze(0)
            cycle_X_image = cycle_X_image.squeeze(0)

            # Save the images as PNG
            fake_Y_pil = Image.fromarray((fake_Y_image * 255).astype(np.uint8))
            cycle_X_pil = Image.fromarray((cycle_X_image * 255).astype(np.uint8))

            fake_Y_pil.save(f"{new_data_path}/{real_X_id}_fake_Y.png")
            cycle_X_pil.save(f"{new_data_path}/{real_X_id}_cycle_X.png")

# Run the test part and save the generated images
save_generated_images(ham_loader_test, generate_XtoY, generate_YtoX, test_image_path, device)

