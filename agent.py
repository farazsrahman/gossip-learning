import torch
from torch import nn
from typing import Callable
import copy

class Agent(nn.Module):
    def __init__(
            self, 
            encoder: nn.Module, 
            decoder: nn.Module,
            encoder_cumulator: str = 'embedding_avg'
    ):
        super().__init__()  # Call parent class initialization first
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_cumulator = encoder_cumulator
        self.update_parameters = True  # Flag to control parameter updates

        self.optimizer = torch.optim.Adam(
            self.parameters(), # create optimizer for only the instances parameters
            lr=0.00001
        )

        self.neighbors: list[Agent] = []

    def forward(self, x: torch.Tensor, exclude_self: bool = False) -> torch.Tensor:
        if self.encoder_cumulator == 'embedding_avg':
            # Original behavior: average embeddings
            embeddings = [self.encoder(x)] if not exclude_self else []
            for neighbor in self.neighbors:
                embeddings.append(neighbor.encoder(x))
            embedding = torch.mean(torch.stack(embeddings), dim=0)
        else:  # encoder_avg
            # Create a new encoder with averaged parameters
            avg_encoder = copy.deepcopy(self.encoder)
            if not exclude_self:
                # Initialize with self's parameters
                for param, self_param in zip(avg_encoder.parameters(), self.encoder.parameters()):
                    param.data.copy_(self_param.data)
            else:
                # Initialize with zeros
                for param in avg_encoder.parameters():
                    param.data.zero_()
            
            # Average parameters with neighbors
            n_models = 1 if not exclude_self else 0
            for neighbor in self.neighbors:
                n_models += 1
                for param, neighbor_param in zip(avg_encoder.parameters(), neighbor.encoder.parameters()):
                    param.data.add_(neighbor_param.data)
            
            # Divide by number of models to get average
            for param in avg_encoder.parameters():
                param.data.div_(n_models)
            
            # Update the current encoder's parameters in place if updates are enabled
            if self.update_parameters:
                for self_param, avg_param in zip(self.encoder.parameters(), avg_encoder.parameters()):
                    self_param.data.copy_(avg_param.data)
            
            # Use the averaged encoder to compute embedding
            embedding = avg_encoder(x)

        return self.decoder(embedding)

    def eval(self):
        super().eval()
        self.update_parameters = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.update_parameters = mode

def create_agents(
        topology: list[list[int]],
        encoder_cumulator: str = 'embedding_avg'
    ) -> list[Agent]:
    assert len(topology) == len(topology[0]), "Adjacency matrix must be square"
    
    n_agents = len(topology)
    agents = []
    for i in range(n_agents):
        # NOTE (faraz): hard coding the encoder / decoder for now
        encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 256)  # Assuming 28x28 MNIST images
        )
        decoder = nn.Sequential(
            nn.Linear(256, 32),
        )
        agents.append(Agent(
            encoder=encoder,
            decoder=decoder,
            encoder_cumulator=encoder_cumulator
        ))

    for i in range(n_agents):
        for j in range(n_agents):
            if topology[i][j] == 1:
                agents[i].neighbors.append(agents[j])

    return agents
