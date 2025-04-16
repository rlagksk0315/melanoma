import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loading import load_data # data_loading.py
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cyclegan import Generator, Discriminator, adversarial_loss, cycle_loss, identity_loss
import itertools

#Load dataset
batch_size = 1
train_loader, validation_loader, test_loader, classes, batch_size = load_data(batch_size)

# Hyperparameters
learning_rate = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# version = 'v1'

# Initialize the model
'''
X to Y: upsampling (encoder)
Y to X: downsampling (decoder)
'''
generate_XtoY = Generator().to(device) #upsampling (encoder)
generate_YtoX = Generator().to(device) #downsampling (decoder)
Discriminator_X = Discriminator().to(device) #discriminator for X 
Discriminator_Y = Discriminator().to(device) #discriminator for Y

# model = cyclegan_model(version=version).to(device)

#call the loss functions
adversarial_loss_func = adversarial_loss()  # Adversarial loss
cycle_loss_func = cycle_loss()   # Cycle consistency loss
identity_loss_func = identity_loss()   # Identity loss
optimizer_for_generators = optim.Adam(
    itertools.chain(generate_XtoY.parameters(), generate_YtoX.parameters()), lr=learning_rate, betas=(0.5, 0.999)
)
optimizer_discriminator_X = optim.Adam(Discriminator_X.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_discriminator_Y = optim.Adam(Discriminator_Y.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training
def train_cyclegan(train_loader, generate_XtoY, generate_YtoX, Discriminator_X, Discriminator_Y, device):
    # model.train() # Model in training mode
    generate_XtoY.train()
    generate_YtoX.train()
    Discriminator_X.train()
    Discriminator_Y.train()

    total_loss_generator = 0.0
    total_loss_discriminator_X = 0.0
    total_loss_discriminator_Y = 0.0

    for batch_idx, (real_X, real_Y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        real_X, real_Y = real_X.to(device), real_Y.to(device)
        # Train generators
        optimizer_for_generators.zero_grad()

        #forward pass
        fake_Y = generate_XtoY(real_X)
        fake_X = generate_YtoX(real_Y)
        cycle_X = generate_YtoX(fake_Y)
        cycle_Y = generate_XtoY(fake_X)

        # Adversarial loss
        loss_G_XtoY = adversarial_loss(Discriminator_Y(fake_Y), torch.ones_like(Discriminator_Y(fake_Y)))
        loss_G_YtoX = adversarial_loss(Discriminator_X(fake_X), torch.ones_like(Discriminator_X(fake_X)))

        # Cycle consistency loss 
        loss_cycle_X = cycle_loss_func(real_X, cycle_X)
        loss_cycle_Y = cycle_loss_func(real_Y, cycle_Y)

        # Identity loss 
        identity_X = generate_YtoX(real_X)
        identity_Y = generate_XtoY(real_Y)
        loss_identity_X = identity_loss_func(real_X, identity_X)
        loss_identity_Y = identity_loss_func(real_Y, identity_Y)

        #total generator loss
        total_loss_generator = (
            loss_G_XtoY + loss_G_YtoX + 10 * (loss_cycle_X + loss_cycle_Y) + 5 * (loss_identity_X + loss_identity_Y)
        )

        # Train discriminators
        optimizer_discriminator_X.zero_grad()
        optimizer_discriminator_Y.zero_grad()

        # Discriminator X loss
        loss_discriminator_X_real = adversarial_loss_func(Discriminator_X(real_X), torch.ones_like(Discriminator_X(real_X)))
        loss_discriminator_X_fake = adversarial_loss_func(Discriminator_X(fake_X.detach()), torch.zeros_like(Discriminator_X(fake_X)))
        loss_discriminator_X = (loss_discriminator_X_real + loss_discriminator_X_fake) / 2
        loss_discriminator_X.backward()
        optimizer_discriminator_X

        #Discriminator Y loss
        loss_discriminator_Y_real = adversarial_loss_func(Discriminator_Y(real_Y), torch.ones_like(Discriminator_Y(real_Y)))
        loss_discriminator_Y_fake = adversarial_loss_func(Discriminator_Y(fake_Y.detach()), torch.zeros_like(Discriminator_Y(fake_Y)))
        loss_discriminator_Y = (loss_discriminator_Y_real + loss_discriminator_Y_fake) / 2
        loss_discriminator_Y.backward()
        optimizer_discriminator_Y.step()
    
    return total_loss_generator, loss_discriminator_X, loss_discriminator_Y

#validation
def validate_cyclegan(generate_XtoY, generate_YtoX, Discriminator_X, Discriminator_Y, validation_loader, device):
    generate_XtoY.eval()
    generate_YtoX.eval()
    Discriminator_X.eval()
    Discriminator_Y.eval()

    total_generator_loss = 0.0
    total_discriminator_X_loss = 0.0
    total_discriminator_Y_loss = 0.0

    with torch.no_grad():
        for real_X, real_Y in tqdm(validation_loader, desc='Validation'):
            real_X, real_Y = real_X.to(device), real_Y.to(device)

            # Generator Forward Pass
            fake_Y = generate_XtoY(real_X)
            fake_X = generate_YtoX(real_Y)
            cycle_X = generate_YtoX(fake_Y)
            cycle_Y = generate_XtoY(fake_X)

            # Adversarial Loss 
            loss_G_XtoY = adversarial_loss(Discriminator_Y(fake_Y), torch.ones_like(Discriminator_Y(fake_Y))) #how well generator fools discriminator
            loss_G_YtoX = adversarial_loss(Discriminator_X(fake_X), torch.ones_like(Discriminator_X(fake_X)))

            # Cycle consistency loss
            loss_cycle_X = cycle_loss_func(real_X, cycle_X)
            loss_cycle_Y = cycle_loss_func(real_Y, cycle_Y)

            # Identity loss
            identity_X = generate_YtoX(real_X)
            identity_Y = generate_XtoY(real_Y)
            loss_identity_X = identity_loss_func(real_X, identity_X)
            loss_identity_Y = identity_loss_func(real_Y, identity_Y)

            # Total generator loss 
            loss_generator = (
                loss_G_XtoY + loss_G_YtoX + 
                10 * (loss_cycle_X + loss_cycle_Y) +
                5 * (loss_identity_X + loss_identity_Y)
            )

            # Discrimnator loss
            loss_discriminator_X_real = adversarial_loss_func(Discriminator_X(real_X), torch.ones_like(Discriminator_X(real_X)))
            loss_discriminator_X_fake = adversarial_loss_func(Discriminator_X(fake_X), torch.zeros_like(Discriminator_X(fake_X)))
            loss_discriminator_X = (loss_discriminator_X_real + loss_discriminator_X_fake) / 2  

            loss_discriminator_Y_real = adversarial_loss_func(Discriminator_Y(real_Y), torch.ones_like(Discriminator_Y(real_Y)))
            loss_discriminator_Y_fake = adversarial_loss_func(Discriminator_Y(fake_Y), torch.zeros_like(Discriminator_Y(fake_Y)))
            loss_discriminator_Y = (loss_discriminator_Y_real + loss_discriminator_Y_fake) / 2

            # accumulate losses
            total_generator_loss += loss_generator.item()
            total_discriminator_X_loss += loss_discriminator_X.item()
            total_discriminator_Y_loss += loss_discriminator_Y.item()   
    average_generator_loss = total_generator_loss / len(validation_loader)
    average_discriminator_X_loss = total_discriminator_X_loss / len(validation_loader)
    average_discriminator_Y_loss = total_discriminator_Y_loss / len(validation_loader)

    return average_generator_loss, average_discriminator_X_loss, average_discriminator_Y_loss

def visualize_metrics(train_generator_losses, val_generator_losses, 
                      train_discriminator_X_losses, val_discriminator_X_losses,
                      train_discriminator_Y_losses, val_discriminator_Y_losses):
    
    # Plot for generator loss
    plt.figure()
    plt.plot(train_generator_losses, label='Train Generator Loss')
    plt.plot(val_generator_losses, label='Validation Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator loss over Epochs')
    plt.savefig('generator_loss.png')
    plt.show()

    # Plot for discriminator X loss
    plt.figure()
    plt.plot(train_discriminator_X_losses, label='Train Discriminator X Loss')
    plt.plot(val_discriminator_X_losses, label='Validation Discriminator X Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Discriminator X loss over Epochs')
    plt.savefig('discriminator_X_loss.png')
    plt.show()

    # Plot for discriminator Y loss
    plt.figure()
    plt.plot(train_discriminator_Y_losses, label='Train Discriminator Y Loss')
    plt.plot(val_discriminator_Y_losses, label='Validation Discriminator Y Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')      
    plt.legend()
    plt.title('Discriminator Y loss over Epochs')
    plt.savefig('discriminator_Y_loss.png')
    plt.show()


def main():
    best_val_generator_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    # tracking loss lists
    train_generator_losses, val_generator_losses = [], []
    train_discriminator_X_losses, val_discriminator_X_losses = [], []
    train_discriminator_Y_losses, val_discriminator_Y_losses = [], []

    with open("cyclegan_training_log.txt", "w") as f:
        f.write("Epoch, Train Generator Loss, Val Generator Loss, Train Discriminator X Loss, Val Discriminator X Loss, Train Discriminator Y Loss, Val Discriminator Y Loss\n")

        # Main training loop
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            
            train_generator_loss, train_discriminator_X_loss, train_discriminator_Y_loss = train_cyclegan(
                train_loader = train_loader,
                generate_XtoY = generate_XtoY,
                generate_YtoX = generate_YtoX,
                Discriminator_X = Discriminator_X,
                Discriminator_Y = Discriminator_Y,
                device = device
            )

            validate_generator_loss, validate_discriminator_X_loss, validate_discriminator_Y_loss = validate_cyclegan(
                generate_XtoY = generate_XtoY,
                generate_YtoX = generate_YtoX,
                Discriminator_X = Discriminator_X,
                Discriminator_Y = Discriminator_Y,
                validation_loader = validation_loader,
                device = device
            )

            # log losses
            train_generator_losses.append(train_generator_loss)
            val_generator_losses.append(validate_generator_loss)
            train_discriminator_X_losses.append(train_discriminator_X_loss)
            val_discriminator_X_losses.append(validate_discriminator_X_loss)
            train_discriminator_Y_losses.append(train_discriminator_Y_loss)
            val_discriminator_Y_losses.append(validate_discriminator_Y_loss)

            f.write(f"{epoch + 1}, {train_generator_loss:.4f}, {validate_generator_loss:.4f},"
                     f"{train_discriminator_X_loss:.4f}, {validate_discriminator_X_loss:.4f},"
                     f"{train_discriminator_Y_loss:.4f}, {validate_discriminator_Y_loss:.4f}\n")
            
            print(f"[Epoch {epoch + 1}]")
            print(f"Generator loss: Train = {train_generator_loss:.4f}, Validation = {validate_generator_loss:.4f}")
            print(f"Discriminator X loss: Train = {train_discriminator_X_loss:.4f}, Validation = {validate_discriminator_X_loss:.4f}")
            print(f"Discriminator Y loss: Train = {train_discriminator_Y_loss:.4f}, Validation = {validate_discriminator_Y_loss:.4f}")
           
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

    if best_model_state is not None:
        torch.save(best_model_state, 'best_cyclegan_model.pth')
        print(f"Best model saved at epoch {best_epoch} with validation generator loss {best_val_generator_loss:.4f}")


    # visualise loss curves
    visualize_metrics(train_generator_losses, val_generator_losses, 
                      train_discriminator_X_losses, val_discriminator_X_losses,
                      train_discriminator_Y_losses, val_discriminator_Y_losses)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Main training loop
    main()

