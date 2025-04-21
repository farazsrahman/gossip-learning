import torch
from torch import nn
from typing import Callable

class Agent(nn.Module):
    def __init__(
            self, 
            encoder: nn.Module, 
            decoder: nn.Module, 
    ):
        super().__init__()  # Call parent class initialization first
        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = torch.optim.Adam(
            self.parameters(), # create optimizer for only the instances parameters
            lr=0.00001
        )

        self.neighbors: list[Agent] = []

    def forward(self, x: torch.Tensor, exclude_self: bool = False) -> torch.Tensor:
        # NOTE (faraz): it would be interesting (maybe non-trivial) 
        # to see if there would be a difference if we averaged 
        # across parameters instead of embeddings.
        embeddings = [self.encoder(x)] if not exclude_self else []
        for neighbor in self.neighbors:
            embeddings.append(neighbor.encoder(x))
        embedding = torch.mean(torch.stack(embeddings), dim=0)

        return self.decoder(embedding)

def create_agents(
        topology: list[list[int]],
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
        ))

    for i in range(n_agents):
        for j in range(n_agents):
            if topology[i][j] == 1:
                agents[i].neighbors.append(agents[j])

    return agents
