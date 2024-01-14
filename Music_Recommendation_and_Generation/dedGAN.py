import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm
from transformers import GPT2Model, GPT2Config


# Conditional Generator with emotional states
class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, emotion_dim, music_dim, hidden_dims, transformer_config):
        super(ConditionalGenerator, self).__init__()
        self.z_dim = z_dim
        self.emotion_dim = emotion_dim
        self.music_dim = music_dim
        input_dim = z_dim + emotion_dim
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.LeakyReLU(0.2)]
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                ]
            )
        layers.append(nn.Linear(hidden_dims[-1], music_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z, emotional_state):
        z_emotion = torch.cat([z, emotional_state], dim=1)
        return self.model(z_emotion)


# Discriminator with spectral normalization
class SpectralNormDiscriminator(nn.Module):
    def __init__(self, music_dim, hidden_dims):
        super(SpectralNormDiscriminator, self).__init__()
        layers = [spectral_norm(nn.Linear(music_dim, hidden_dims[0])), nn.LeakyReLU(0.2)]
        for i in range(len(hidden_dims) - 1):
            # Correction: Apply spectral normalization properly.
            layers.extend(
                [
                    spectral_norm(nn.Linear(hidden_dims[i], hidden_dims[i + 1])),  # Apply spectral norm here
                    nn.LeakyReLU(0.2)
                ]
            )
        layers.append(spectral_norm(nn.Linear(hidden_dims[-1], 1)))  # And here
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, music_motif):
        return self.model(music_motif)


# Initialize models
z_dim = 100  # Dimensionality of the noise vector
music_dim = 200  # Dimensionality of the music motif vector
emotion_dim = 50
g_hidden_dims = [128, 256, 512]
d_hidden_dims = [512, 256, 128]

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer configuration
transformer_config = GPT2Config(vocab_size=1, n_positions=music_dim)

# Initialize models and send them to the device
G = ConditionalGenerator(z_dim, emotion_dim, music_dim, g_hidden_dims, transformer_config).to(device)
D = SpectralNormDiscriminator(music_dim, d_hidden_dims).to(device)

# Loss functions and optimizers
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Training procedure
def train_conditional_gan(
        dataset, G, D, criterion, d_optimizer, g_optimizer, z_dim, emotion_dim, epochs=1000, batch_size=64
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for emotional_state, music_motif in dataloader:
            # Ensure data and models are on the same device
            emotional_state, music_motif = emotional_state.to(device), music_motif.to(device)

            # Make sure the batch dimension is first for both emotional_state and music_motif
            emotional_state = emotional_state.view(batch_size, -1)
            music_motif = music_motif.view(batch_size, -1)

            # Correction: Train Discriminator with real data
            D.zero_grad()
            real_data = music_motif
            real_decision = D(real_data)
            real_error = criterion(real_decision, torch.ones_like(real_decision))  # ones = true
            real_error.backward()

            # Correction: Generate fake data with both z and emotional state
            z = torch.randn(batch_size, z_dim)
            # The emotional_state needs to be included as an input to the generator
            fake_data = G(z, emotional_state).detach()  # Detach to avoid training G on these labels
            fake_decision = D(fake_data)
            fake_error = criterion(fake_decision, torch.zeros_like(fake_decision))  # zeros = fake
            fake_error.backward()

            d_optimizer.step()

            # Correction: Train Generator with z and emotional state
            G.zero_grad()
            z = torch.randn(batch_size, z_dim)
            # Again, include emotional_state as an input to the generator
            fake_data = G(z, emotional_state)
            fake_decision = D(fake_data)
            g_error = criterion(fake_decision, torch.ones_like(fake_decision))  # we want to fool D
            g_error.backward()

            g_optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs} - D Loss: {real_error + fake_error:.4f} - G Loss: {g_error:.4f}")


# Add the dataset (reocrd using EMOTIV EPOC+ and the Audio Recorder Script)
# emotional_states = ...
# music_motifs = ...

dataset = TensorDataset(torch.Tensor(emotional_states), torch.Tensor(music_motifs))
# Rename the function to match the definition
train_conditional_gan(
    dataset=dataset, G=G, D=D, criterion=criterion, d_optimizer=d_optimizer, g_optimizer=g_optimizer, z_dim=z_dim,
    emotion_dim=emotion_dim
    )
