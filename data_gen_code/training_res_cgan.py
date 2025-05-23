import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from data_loading import get_dataloaders
from torch.utils.data import DataLoader
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from cyclegan_resnet import ResnetGenerator, PatchGANDiscriminator, ImageBuffer, lsgan_loss, generator_adversarial_loss, discriminator_adversarial_loss, cycle_loss, identity_loss
import itertools

# Argparse Details
parser = argparse.ArgumentParser(description='CycleGAN Training Script')
parser.add_argument('--gen_images_path', type=str, default='../generated_images')
parser.add_argument('--results_path', type=str, default='../results')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--scheduler', type=str, choices=['lambda', 'cosine', 'none'], default='none',
                    help='Learning rate scheduler: lambda (linear decay), cosine (CosineAnnealingLR), or none')
parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw'], default='adam',
                    help='Choose optimizer: adam or adamw')



args = parser.parse_args()

# Override hyperparameters
epochs = args.num_epochs
learning_rate = args.learning_rate

# Hyperparameters
batch_size = 1
num_workers = 2
seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
ddi_data_dir = "../data/ddi_cropped"
ham_data_dir = "../data/HAM10000"
scin_data_dir = "../data/dark_scin"
ddi_loader_train, ddi_loader_val, ham_loader_train, ham_loader_val, ham_loader_test, scin_loader_train, scin_loader_val = get_dataloaders(ddi_data_dir,
                                                                                                                        ham_data_dir, scin_data_dir,
                                                                                                                        batch_size=batch_size,
                                                                                                                        num_workers=num_workers,
                                                                                                                        seed=seed)

#I combined the dataset for ddi & scin together as a darkskin_dataset 
darkskin_loader_train = torch.utils.data.ConcatDataset([ddi_loader_train.dataset, scin_loader_train.dataset])
darkskin_loader_val = torch.utils.data.ConcatDataset([ddi_loader_val.dataset, scin_loader_val.dataset])
#darkskin_loader_test = torch.utils.data.ConcatDataset([ddi_loader_test.dataset, scin_loader_test.dataset])

darkskin_loader_train = DataLoader(darkskin_loader_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
darkskin_loader_val = DataLoader(darkskin_loader_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
#darkskin_loader_test = DataLoader(darkskin_loader_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)



'''
combined_train_dataset = torch.utils.data.ConcatDataset([ham_loader_train.dataset, ddi_loader_train.dataset])
train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

combined_val_dataset = torch.utils.data.ConcatDataset([ham_loader_val.dataset, ddi_loader_val.dataset])
val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

combined_test_dataset = torch.utils.data.ConcatDataset([ham_loader_test.dataset, ddi_loader_test.dataset])
test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
'''

# Set random seed for reproducibility
torch.manual_seed(seed)

# Initialize the model
'''
X to Y: upsampling (encoder)
Y to X: downsampling (decoder)
'''
generate_XtoY = ResnetGenerator().to(device) #upsampling (encoder)
generate_YtoX = ResnetGenerator().to(device) #downsampling (decoder)
Discriminator_X = PatchGANDiscriminator().to(device) #discriminator for X 
Discriminator_Y = PatchGANDiscriminator().to(device) #discriminator for Y

if args.optimizer == 'adam':
    optimizer_for_generators = optim.Adam(
        itertools.chain(generate_XtoY.parameters(), generate_YtoX.parameters()), 
        lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizer_discriminator_X = optim.Adam(Discriminator_X.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_discriminator_Y = optim.Adam(Discriminator_Y.parameters(), lr=learning_rate, betas=(0.5, 0.999))

elif args.optimizer == 'adamw':
    optimizer_for_generators = optim.AdamW(
        itertools.chain(generate_XtoY.parameters(), generate_YtoX.parameters()), 
        lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-4
    )
    optimizer_discriminator_X = optim.AdamW(Discriminator_X.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_discriminator_Y = optim.AdamW(Discriminator_Y.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)


# learning rate decay
if args.scheduler == 'lambda':
    def lambda_rule(epoch):
        if epoch < 100:
            return 1.0
        else:
            return 1.0 - (epoch - 100) / float(epochs - 100 + 1)

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_for_generators, lr_lambda=lambda_rule)
    scheduler_D_X = optim.lr_scheduler.LambdaLR(optimizer_discriminator_X, lr_lambda=lambda_rule)
    scheduler_D_Y = optim.lr_scheduler.LambdaLR(optimizer_discriminator_Y, lr_lambda=lambda_rule)

elif args.scheduler == 'cosine':
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_for_generators, T_max=epochs)
    scheduler_D_X = optim.lr_scheduler.CosineAnnealingLR(optimizer_discriminator_X, T_max=epochs)
    scheduler_D_Y = optim.lr_scheduler.CosineAnnealingLR(optimizer_discriminator_Y, T_max=epochs)

else:
    scheduler_G = None
    scheduler_D_X = None
    scheduler_D_Y = None


# image buffers
fake_X_buffer = ImageBuffer()
fake_Y_buffer = ImageBuffer()


# Training
def train_cyclegan(ham_loader_train, darkskin_loader_train, generate_XtoY, generate_YtoX, Discriminator_X, Discriminator_Y, device):
    # model.train() # Model in training mode
    generate_XtoY.train()
    generate_YtoX.train()
    Discriminator_X.train()
    Discriminator_Y.train()

    total_generator_loss, total_discriminator_X_loss, total_discriminator_Y_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    total_identity_X_loss, total_identity_Y_loss = 0, 0
    n = 0 # batch

    loop = tqdm(zip(ham_loader_train, darkskin_loader_train), total=min(len(ham_loader_train), len(darkskin_loader_train)))
    for (real_X, real_X_id), darkskin_batch in loop:
        real_X = real_X.float().to(device) # reference HAM
        real_X_id = real_X_id[0]
        real_Y = darkskin_batch.float().to(device) # reference darkskin images
        #real_Y = real_Y.unsqueeze(0)
        n += 1
        
        # ==========================GENERATOR==========================

        # Train generators
        optimizer_for_generators.zero_grad()

        #forward pass
        fake_Y = generate_XtoY(real_X) # light skinned DDI
        fake_X = generate_YtoX(real_Y) # dark skinned HAM
        cycle_X = generate_YtoX(fake_Y) # Reconstruct X from fake Y
        cycle_Y = generate_XtoY(fake_X) # Reconstruct Y from fake X

        # Adversarial loss
        loss_G_XtoY = generator_adversarial_loss(Discriminator_Y, fake_Y)
        loss_G_YtoX = generator_adversarial_loss(Discriminator_X, fake_X)

        # Cycle consistency loss 
        loss_cycle_X = cycle_loss(real_X, cycle_X)
        loss_cycle_Y = cycle_loss(real_Y, cycle_Y)

        # Identity loss 
        identity_X = generate_YtoX(real_X)
        identity_Y = generate_XtoY(real_Y)
        loss_identity_X = identity_loss(real_X, identity_X)
        loss_identity_Y = identity_loss(real_Y, identity_Y)

        # total generator loss
        loss_generator = (
            loss_G_XtoY + loss_G_YtoX + 10 * (loss_cycle_X + loss_cycle_Y) + 5 * (loss_identity_X + loss_identity_Y)
        )

        loss_generator.backward()
        optimizer_for_generators.step()
        total_generator_loss += loss_generator.item()

        # ==========================DISCRIMINATOR==========================

        # Train discriminator X
        optimizer_discriminator_X.zero_grad()

        fake_X_for_disc = fake_X_buffer.push_and_pop(fake_X)    # add buffer
        loss_discriminator_X = discriminator_adversarial_loss(Discriminator_X, real_X, fake_X_for_disc)
        
        loss_discriminator_X.backward()
        optimizer_discriminator_X.step()

        # Discriminator Y loss
        optimizer_discriminator_Y.zero_grad()
        
        fake_Y_for_disc = fake_Y_buffer.push_and_pop(fake_Y)    # add buffer
        loss_discriminator_Y = discriminator_adversarial_loss(Discriminator_Y, real_Y, fake_Y_for_disc)
        
        loss_discriminator_Y.backward()
        optimizer_discriminator_Y.step()

        # Accumulate losses
        total_discriminator_X_loss += loss_discriminator_X.item()
        total_discriminator_Y_loss += loss_discriminator_Y.item()
        total_identity_X_loss += loss_identity_X
        total_identity_Y_loss += loss_identity_Y

    # Average Losses
    average_generator_loss = total_generator_loss / n
    average_discriminator_X_loss = total_discriminator_X_loss / n
    average_discriminator_Y_loss = total_discriminator_Y_loss / n
    average_identity_X_loss = total_identity_X_loss / n
    average_identity_Y_loss = total_identity_Y_loss / n
        
    return average_generator_loss, average_discriminator_X_loss, average_discriminator_Y_loss, average_identity_X_loss, average_identity_Y_loss

#validation
def validate_cyclegan(ham_loader_val, darkskin_loader_val, generate_XtoY, generate_YtoX, Discriminator_X, Discriminator_Y, device, epoch, image_path):
    generate_XtoY.eval()
    generate_YtoX.eval()
    Discriminator_X.eval()
    Discriminator_Y.eval()

    total_generator_loss, total_discriminator_X_loss, total_discriminator_Y_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    total_identity_X_loss, total_identity_Y_loss = 0, 0
    n = 0

    with torch.no_grad():
        loop = tqdm(zip(ham_loader_val, darkskin_loader_val), total=min(len(ham_loader_val), len(darkskin_loader_val)))
        for (real_X, real_X_id), darkskin_batch in loop:
            real_X = real_X.float().to(device)
            real_X_id = real_X_id[0]
            real_Y = darkskin_batch.float().to(device)  # reference darkskin images
            #real_Y = real_Y.unsqueeze(0)
            n += 1

            # ==========================GENERATOR==========================

            fake_Y = generate_XtoY(real_X)
            fake_X = generate_YtoX(real_Y)
            cycle_X = generate_YtoX(fake_Y)
            cycle_Y = generate_XtoY(fake_X)

            # Adversarial loss
            loss_G_XtoY = generator_adversarial_loss(Discriminator_Y, fake_Y)
            loss_G_YtoX = generator_adversarial_loss(Discriminator_X, fake_X)

            # Cycle consistency loss 
            loss_cycle_X = cycle_loss(real_X, cycle_X)
            loss_cycle_Y = cycle_loss(real_Y, cycle_Y)

            # Identity loss 
            identity_X = generate_YtoX(real_X)
            identity_Y = generate_XtoY(real_Y)
            loss_identity_X = identity_loss(real_X, identity_X)
            loss_identity_Y = identity_loss(real_Y, identity_Y)

            # Total generator loss 
            loss_generator = (
                loss_G_XtoY + loss_G_YtoX + 
                10 * (loss_cycle_X + loss_cycle_Y) +
                5 * (loss_identity_X + loss_identity_Y)
            )

            # ==========================DISCRIMINATOR==========================

            loss_discriminator_X = discriminator_adversarial_loss(Discriminator_X, real_X, fake_X)
            loss_discriminator_Y = discriminator_adversarial_loss(Discriminator_Y, real_Y, fake_Y)
            
            # Accumulate losses
            total_generator_loss += loss_generator.item()
            total_discriminator_X_loss += loss_discriminator_X.item()
            total_discriminator_Y_loss += loss_discriminator_Y.item()
            total_identity_X_loss += loss_identity_X
            total_identity_Y_loss += loss_identity_Y

    # Average Losses
    average_generator_loss = total_generator_loss / n
    average_discriminator_X_loss = total_discriminator_X_loss / n
    average_discriminator_Y_loss = total_discriminator_Y_loss / n
    average_identity_X_loss = total_identity_X_loss / n
    average_identity_Y_loss = total_identity_Y_loss / n

    # Display images
    display_images(image_path, real_X, fake_Y, cycle_X, identity_X, epoch, real_X_id)

    return average_generator_loss, average_discriminator_X_loss, average_discriminator_Y_loss, average_identity_X_loss, average_identity_Y_loss

def visualize_metrics(train_generator_losses, val_generator_losses,
                      train_discriminator_X_losses, val_discriminator_X_losses,
                      train_discriminator_Y_losses, val_discriminator_Y_losses,
                      train_identity_X_losses, val_identity_X_losses,
                      train_identity_Y_losses, val_identity_Y_losses,
                      results_path):

    # Plot for generator loss
    plt.figure()
    plt.plot(train_generator_losses, label='Train Generator Loss')
    plt.plot(val_generator_losses, label='Validation Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator loss over Epochs')
    plt.savefig(f'{results_path}/generator_loss.png')
    plt.show()

    # Plot for discriminator X loss
    plt.figure()
    plt.plot(train_discriminator_X_losses, label='Train Discriminator X Loss')
    plt.plot(val_discriminator_X_losses, label='Validation Discriminator X Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Discriminator X loss over Epochs')
    plt.savefig(f'{results_path}/discriminator_X_loss.png')
    plt.show()

    # Plot for discriminator Y loss
    plt.figure()
    plt.plot(train_discriminator_Y_losses, label='Train Discriminator Y Loss')
    plt.plot(val_discriminator_Y_losses, label='Validation Discriminator Y Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')      
    plt.legend()
    plt.title('Discriminator Y loss over Epochs')
    plt.savefig(f'{results_path}/discriminator_Y_loss.png')
    plt.show()

    # Plot for identity X loss
    plt.figure()
    plt.plot(train_identity_X_losses, label='Train Identity X Loss')
    plt.plot(val_identity_X_losses, label='Validation Identity X Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')      
    plt.legend()
    plt.title('Identity X loss over Epochs')
    plt.savefig(f'{results_path}/identity_X_loss.png')
    plt.show()

    # Plot for identity Y loss
    plt.figure()
    plt.plot(train_identity_Y_losses, label='Train Identity Y Loss')
    plt.plot(val_identity_Y_losses, label='Validation Identity Y Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')      
    plt.legend()
    plt.title('Identity Y loss over Epochs')
    plt.savefig(f'{results_path}/identity_Y_loss.png')
    plt.show()


def denormalize_image(image_tensor, dataset_type='Light'):
    """
    Denormalizes the image tensor based on the dataset (LIGHT or DARK).
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



# display images
def display_images(image_path, real_X, fake_Y, cycle_X, identity_X, epoch, real_X_id, num_images=1):
    """
    Displays real and generated images for reference and comparison.
    """

    # Convert tensors to numpy arrays for displaying with matplotlib
    real_X = real_X.cpu().detach() # HAM
    fake_Y = fake_Y.cpu().detach() # dark skinned HAM
    cycle_X = cycle_X.cpu().detach()
    identity_X = identity_X.cpu().detach()

    # squeeze the batch dimension
    real_X = np.squeeze(real_X, axis=0)
    fake_Y = np.squeeze(fake_Y, axis=0)
    cycle_X = np.squeeze(cycle_X, axis=0)
    identity_X = np.squeeze(identity_X, axis=0)
    
    real_X = denormalize_image(real_X, 'LIGHT')  # Light skin
    fake_Y = denormalize_image(fake_Y, 'DARK')  # Dark skin (generated)
    cycle_X = denormalize_image(cycle_X, 'LIGHT')  # Light skin
    identity_X = denormalize_image(identity_X, 'LIGHT')  # Light skin
    
    # Transpose the image from (C, H, W) to (H, W, C) for visualization
    real_X = np.transpose(real_X, (1, 2, 0))
    fake_Y = np.transpose(fake_Y, (1, 2, 0))
    cycle_X = np.transpose(cycle_X, (1, 2, 0))
    identity_X = np.transpose(identity_X, (1, 2, 0))
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Display real images
    axes[0].imshow(real_X)
    axes[0].set_title(f"Real X")
    axes[0].axis('off')

    # Display generated images
    axes[1].imshow(fake_Y)
    axes[1].set_title(f"Fake Y")
    axes[1].axis('off')

    # Display cycle images
    axes[2].imshow(cycle_X)
    axes[2].set_title(f"Cycle X")
    axes[2].axis('off')

    # Display identity images
    axes[3].imshow(identity_X)
    axes[3].set_title(f"Identity X")
    axes[3].axis('off')
        
    plt.savefig(f'{image_path}/epoch{epoch+1}_{real_X_id}_real_fake_cycle_identity.png')


# ================== MAIN TESTING FUNCTION ==================

def save_generated_images(ham_loader_test, generate_XtoY, generate_YtoX, new_data_path, device):
    """
    Generate and save images from test data (HAM test set).
    """
    generate_XtoY.eval()
    generate_YtoX.eval()

    # Create folder to store generated images
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

            # Check shape of the generated images
            #print(f"fake_Y_image shape: {fake_Y_image.shape}")
            #print(f"cycle_X_image shape: {cycle_X_image.shape}")

            # Ensure the image is 3D (C, H, W) before transposing
            if fake_Y_image.ndim == 3:
                fake_Y_image = np.transpose(fake_Y_image, (1, 2, 0))
                cycle_X_image = np.transpose(cycle_X_image, (1, 2, 0))
            elif fake_Y_image.ndim == 4:
                fake_Y_image = fake_Y_image.squeeze(0)  # Remove the batch dimension
                fake_Y_image = np.transpose(fake_Y_image, (1, 2, 0))
                cycle_X_image = cycle_X_image.squeeze(0)  # Remove the batch dimension
                cycle_X_image = np.transpose(cycle_X_image, (1, 2, 0))

            # Save the images as PNG
            fake_Y_pil = Image.fromarray((fake_Y_image * 255).astype(np.uint8))
            cycle_X_pil = Image.fromarray((cycle_X_image * 255).astype(np.uint8))

            fake_Y_pil.save(f"{new_data_path}/{real_X_id}_fake_Y.png")
            cycle_X_pil.save(f"{new_data_path}/{real_X_id}_cycle_X.png")



def main():
    best_val_generator_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    # tracking loss lists
    train_generator_losses, val_generator_losses = [], []
    train_discriminator_X_losses, val_discriminator_X_losses = [], []
    train_discriminator_Y_losses, val_discriminator_Y_losses = [], []
    train_identity_X_losses, val_identity_X_losses = [], []
    train_identity_Y_losses, val_identity_Y_losses = [], []

    # logging
    os.makedirs(args.results_path, exist_ok=True)

    with open(f"{args.results_path}/cyclegan_training_log.txt", "w") as f:
        f.write("Epoch, Train Generator Loss, Val Generator Loss, Train Discriminator X Loss, Val Discriminator X Loss, Train Discriminator Y Loss, Val Discriminator Y Loss\n")

        # Main training loop
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            
            (train_generator_loss,
             train_discriminator_X_loss, train_discriminator_Y_loss,
             train_identity_X_loss, train_identity_Y_loss) = train_cyclegan(
                ham_loader_train = ham_loader_train,
                darkskin_loader_train = darkskin_loader_train,
                generate_XtoY = generate_XtoY,
                generate_YtoX = generate_YtoX,
                Discriminator_X = Discriminator_X,
                Discriminator_Y = Discriminator_Y,
                device = device
            )

            validation_img_path = f'{args.gen_images_path}' # path to save validation generated images
            os.makedirs(validation_img_path, exist_ok=True)

            (validate_generator_loss,
             validate_discriminator_X_loss, validate_discriminator_Y_loss,
             validate_identity_X_loss, validate_identity_Y_loss) = validate_cyclegan(
                ham_loader_val = ham_loader_val,
                darkskin_loader_val = darkskin_loader_val,
                generate_XtoY = generate_XtoY,
                generate_YtoX = generate_YtoX,
                Discriminator_X = Discriminator_X,
                Discriminator_Y = Discriminator_Y,
                device = device,
                epoch = epoch,
                image_path = validation_img_path
            )

            # log losses
            train_generator_losses.append(train_generator_loss)
            val_generator_losses.append(validate_generator_loss)
            train_discriminator_X_losses.append(train_discriminator_X_loss)
            val_discriminator_X_losses.append(validate_discriminator_X_loss)
            train_discriminator_Y_losses.append(train_discriminator_Y_loss)
            val_discriminator_Y_losses.append(validate_discriminator_Y_loss)
            train_identity_X_losses.append(train_identity_X_loss)
            val_identity_X_losses.append(validate_identity_X_loss)
            train_identity_Y_losses.append(train_identity_Y_loss)
            val_identity_Y_losses.append(validate_identity_Y_loss)

            f.write(f"{epoch + 1}, {train_generator_loss:.4f}, {validate_generator_loss:.4f},"
                     f"{train_discriminator_X_loss:.4f}, {validate_discriminator_X_loss:.4f},"
                     f"{train_discriminator_Y_loss:.4f}, {validate_discriminator_Y_loss:.4f},"
                     f"{train_identity_X_loss:.4f}, {validate_identity_X_loss:.4f},"
                     f"{train_identity_Y_loss:.4f}, {validate_identity_Y_loss:.4f}\n")
            
            #print(f"[Epoch {epoch + 1}]")
            print(f"Generator loss: Train = {train_generator_loss:.4f}, Validation = {validate_generator_loss:.4f}")
            print(f"Discriminator X loss: Train = {train_discriminator_X_loss:.4f}, Validation = {validate_discriminator_X_loss:.4f}")
            print(f"Discriminator Y loss: Train = {train_discriminator_Y_loss:.4f}, Validation = {validate_discriminator_Y_loss:.4f}")
            print(f"Identity X loss: Train = {train_identity_X_loss:.4f}, Validation = {validate_identity_X_loss:.4f}")
            print(f"Identity Y loss: Train = {train_identity_Y_loss:.4f}, Validation = {validate_identity_Y_loss:.4f}")
           
            # save best model
            if validate_generator_loss < best_val_generator_loss:
                best_val_generator_loss = validate_generator_loss
                best_epoch = epoch + 1
                best_model_state = {
                   'Generator_XtoY': generate_XtoY.state_dict(),
                   'Generator_YtoX': generate_YtoX.state_dict(),
                   'Discriminator_X': Discriminator_X.state_dict(),
                   'Discriminator_Y': Discriminator_Y.state_dict(),
                }
                print(f"Best model saved at epoch {best_epoch} with validation generator loss {best_val_generator_loss:.4f}")
                f.write(f"Best model saved at epoch {best_epoch} with validation generator loss {best_val_generator_loss:.4f}")

            # learning rate decay
            if scheduler_G:
                scheduler_G.step()
                scheduler_D_X.step()
                scheduler_D_Y.step()

                current_lr = scheduler_G.get_last_lr()[0]
                print(f"Current Learning Rate: {current_lr:.6f}")



    if best_model_state is not None:
        torch.save(best_model_state, f'{args.results_path}/best_cyclegan_model.pth')
        print(f"====Final best model saved at epoch {best_epoch} with validation generator loss {best_val_generator_loss:.4f}====")

    train_generator_losses = torch.tensor(train_generator_losses)
    val_generator_losses = torch.tensor(val_generator_losses)
    train_discriminator_X_losses = torch.tensor(train_discriminator_X_losses)
    val_discriminator_X_losses = torch.tensor(val_discriminator_X_losses)
    train_discriminator_Y_losses = torch.tensor(train_discriminator_Y_losses)
    val_discriminator_Y_losses = torch.tensor(val_discriminator_Y_losses)
    train_identity_X_losses = torch.tensor(train_identity_X_losses)
    val_identity_X_losses = torch.tensor(val_identity_X_losses)
    train_identity_Y_losses = torch.tensor(train_identity_Y_losses)
    val_identity_Y_losses = torch.tensor(val_identity_Y_losses)

    # visualise loss curves
    results_path = results_path = f'{args.results_path}'
    os.makedirs(results_path, exist_ok=True)
    visualize_metrics(train_generator_losses, val_generator_losses, 
                      train_discriminator_X_losses, val_discriminator_X_losses,
                      train_discriminator_Y_losses, val_discriminator_Y_losses,
                      train_identity_X_losses, val_identity_X_losses,
                      train_identity_Y_losses, val_identity_Y_losses,
                      results_path)


    # TESTING
    # load best model
    if best_model_state is not None:
        generate_XtoY.load_state_dict(best_model_state['Generator_XtoY'])
        generate_YtoX.load_state_dict(best_model_state['Generator_YtoX'])
        Discriminator_X.load_state_dict(best_model_state['Discriminator_X'])
        Discriminator_Y.load_state_dict(best_model_state['Discriminator_Y'])
        print(f"Best model loaded from epoch {best_epoch}")

        # generate images using the HAM test set
        print('Starting data generation on HAM test dataset..')
        test_image_path = '../data/darkHAM'
        os.makedirs(test_image_path, exist_ok=True)

        save_generated_images(ham_loader_test, generate_XtoY, generate_YtoX, test_image_path, device)
        print('Generation Complete!')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Main training loop
    main()
