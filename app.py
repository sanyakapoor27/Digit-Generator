import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import os

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Define the Generator model (lightweight for easy deployment)
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Small but effective architecture
        self.model = nn.Sequential(
            # Input: latent_dim + 10 (for digit label)
            nn.Linear(latent_dim + 10, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 784),  # 28*28 = 784
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        device = noise.device
        # One-hot encode labels
        labels_onehot = torch.zeros(labels.size(0), 10, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Concatenate noise and labels
        gen_input = torch.cat((noise, labels_onehot), 1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(784 + 10, 512),  # 784 + 10 for label
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        device = img.device
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        # One-hot encode labels
        labels_onehot = torch.zeros(labels.size(0), 10, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Concatenate image and labels
        d_input = torch.cat((img_flat, labels_onehot), 1)
        validity = self.model(d_input)
        return validity

# Training function
def train_model():
    """Train the GAN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    latent_dim = 100
    lr = 0.0002
    batch_size = 128
    epochs = 50  # Reduced for faster training
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Training loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)
            
            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss on real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)
            
            # Loss on fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
            gen_imgs = generator(z, gen_labels)
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Update progress
            if i % 100 == 0:
                progress = (epoch * len(dataloader) + i) / (epochs * len(dataloader))
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch}/{epochs}, Batch {i}/{len(dataloader)}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    # Save the trained generator
    torch.save(generator.state_dict(), 'generator.pth')
    return generator

# Function to generate images
def generate_digit_images(generator, digit, num_images=5):
    """Generate images for a specific digit"""
    device = next(generator.parameters()).device
    generator.eval()
    
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(num_images, generator.latent_dim, device=device)
        
        # Create labels for the desired digit
        labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
        
        # Generate images
        fake_imgs = generator(z, labels)
        
        # Convert to numpy and denormalize
        fake_imgs = fake_imgs.cpu().numpy()
        fake_imgs = (fake_imgs + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        
    return fake_imgs

# Function to convert numpy array to PIL Image
def numpy_to_pil(img_array):
    """Convert numpy array to PIL Image"""
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array.squeeze(), mode='L')

# Load or train model
@st.cache_resource
def load_model():
    """Load the trained model or train a new one"""
    generator = Generator()
    
    if os.path.exists('generator.pth'):
        try:
            generator.load_state_dict(torch.load('generator.pth', map_location='cpu'))
            return generator
        except:
            st.warning("Could not load existing model. Training a new one...")
    
    # If no model exists, train a new one
    st.info("No trained model found. Training a new model... This may take a few minutes.")
    generator = train_model()
    return generator

# Main Streamlit app
def main():
    st.title("🔢 Handwritten Digit Image Generator")
    st.markdown("Generate synthetic MNIST-style handwritten digits using a trained GAN model!")
    
    # Sidebar for controls
    st.sidebar.header("Generation Controls")
    
    # Model status
    with st.spinner("Loading model..."):
        generator = load_model()
    
    st.sidebar.success("Model loaded successfully!")
    
    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Select digit to generate (0-9):",
        options=list(range(10)),
        index=2
    )
    
    # Number of images to generate
    num_images = st.sidebar.slider(
        "Number of images to generate:",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Generate button
    if st.sidebar.button("🎲 Generate Images", type="primary"):
        with st.spinner(f"Generating {num_images} images of digit {selected_digit}..."):
            # Generate images
            generated_images = generate_digit_images(generator, selected_digit, num_images)
            
            # Display results
            st.subheader(f"Generated Images of Digit {selected_digit}")
            
            # Create columns for image display
            cols = st.columns(min(num_images, 5))
            
            for i in range(num_images):
                col_idx = i % 5
                with cols[col_idx]:
                    # Convert to PIL Image
                    pil_img = numpy_to_pil(generated_images[i])
                    
                    # Resize for better display
                    pil_img = pil_img.resize((128, 128), Image.NEAREST)
                    
                    # Display image
                    st.image(pil_img, caption=f"Generated {selected_digit} #{i+1}")
                    
                    # Add download button
                    img_buffer = io.BytesIO()
                    pil_img.save(img_buffer, format='PNG')
                    st.download_button(
                        label=f"📥 Download #{i+1}",
                        data=img_buffer.getvalue(),
                        file_name=f"digit_{selected_digit}_{i+1}.png",
                        mime="image/png",
                        key=f"download_{i}"
                    )
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About This App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Architecture")
        st.markdown("""
        - **Generator**: Conditional GAN with ~0.5M parameters
        - **Training**: MNIST dataset (28x28 grayscale)
        - **Framework**: PyTorch
        - **Deployment**: Streamlit
        """)
    
    with col2:
        st.subheader("🎯 Features")
        st.markdown("""
        - Generate any digit (0-9)
        - Multiple images per generation
        - Download generated images
        - Lightweight model for fast inference
        - Responsive web interface
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Model Specifications:**
        - Generator: 4-layer MLP with BatchNorm and ReLU
        - Input: 100-dim noise + 10-dim one-hot label
        - Output: 28×28 grayscale image
        - Total parameters: ~500K (lightweight for deployment)
        
        **Training Details:**
        - Dataset: MNIST (60,000 training images)
        - Epochs: 50 (optimized for balance of quality/speed)
        - Optimizer: Adam (lr=0.0002, betas=(0.5, 0.999))
        - Loss: Binary Cross Entropy
        
        **Deployment:**
        - Framework: Streamlit
        - Model size: ~2MB
        - CPU inference: <100ms per batch
        """)

if __name__ == "__main__":
    main()