import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy
import time
import os

from agent import Agent, create_agents
from dataset import load_mnist_data
from config import TrainingConfig
from utils import visualize_agents

def train(config: TrainingConfig):
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Visualize the agent topology and digit assignments
    visualize_agents(config)
    
    # Fix random seed for reproducibility
    torch.manual_seed(42)

    agents = create_agents(config.topology)
    print(f"Created {len(agents)} agents")

    data_loaders = []
    for idx, agent in enumerate(agents):
        train_loader, val_loader = load_mnist_data(
            digits=config.digits_partition[idx],
            batch_size=config.batch_size
        )
        data_loaders.append({
            "train": train_loader,
            "val": val_loader,
        })
    print(f"Created {len(data_loaders)} data loaders")

    criterion = nn.CrossEntropyLoss()

    update_count = 0
    running_loss = []
    
    pbar = tqdm(range(config.n_updates), desc="Training Progress")
    
    for _ in pbar:
        for agent_idx, agent in enumerate(agents):
            # Check if we need to refresh the data loader
            if not data_loaders[agent_idx]["train"]:
                train_loader, _ = load_mnist_data(
                    digits=config.digits_partition[agent_idx],
                    batch_size=config.batch_size
                )
                data_loaders[agent_idx]["train"] = train_loader
            
            # Get a batch of data
            try:
                batch_x, batch_y = next(iter(data_loaders[agent_idx]["train"]))
            except StopIteration:
                # Refresh the data loader if it's exhausted
                train_loader, _ = load_mnist_data(
                    digits=config.digits_partition[agent_idx],
                    batch_size=config.batch_size
                )
                data_loaders[agent_idx]["train"] = train_loader
                batch_x, batch_y = next(iter(data_loaders[agent_idx]["train"]))
            
            agent.optimizer.zero_grad()
            
            outputs = agent(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            agent.optimizer.step()
            
            running_loss.append(loss.item())
            running_avg = np.mean(running_loss[-50:])
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'running_loss': f'{running_avg:.4f}'
            })
        
        update_count += 1
        
        if update_count % 100 == 0:  # Perform validation every 100 updates
            metrics = do_validation(agents, data_loaders, criterion)
            
            print(f"Update {update_count}/{config.n_updates}, Running Loss: {running_avg:.4f}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
    
    pbar.close()

def do_validation(
        agents: list[Agent], 
        data_loaders: list[DataLoader], 
        criterion: nn.Module
    ) -> dict[str, float]:
    metrics = {}
    with torch.no_grad():
        for idx, agent in enumerate(agents):
            agent.eval()
            agent_val_loss = 0.0
            agent_correct = 0
            agent_total = 0
            
            for val_x, val_y in data_loaders[idx]["val"]:
                val_outputs = agent(val_x)
                agent_val_loss += criterion(val_outputs, val_y).item()
                _, predicted = torch.max(val_outputs.data, 1)
                agent_total += val_y.size(0)
                agent_correct += (predicted == val_y).sum().item()

            metrics[f"agent_{idx}_val_loss"] = agent_val_loss / len(data_loaders[idx]["val"])
            metrics[f"agent_{idx}_accuracy"] = 100 * agent_correct / agent_total


    # oracle agent validation
    oracle_encoder = lambda x: torch.mean(torch.stack([agent.encoder(x) for agent in agents]), dim=0)
    oracle_decoder = nn.Sequential(
        nn.Linear(256, 32),
    )
    oracle_decoder.train()
    
    oracle_train_loader, oracle_val_loader = load_mnist_data(
        digits=list(range(10)),
        batch_size=config.batch_size
    )

    oracle_optimizer = torch.optim.Adam(oracle_decoder.parameters(), lr=0.001)

    # Train oracle decoder for 100 updates
    for train_x, train_y in tqdm(oracle_train_loader, desc="Training oracle decoder"):
        oracle_optimizer.zero_grad()
        with torch.no_grad():
            embedding = oracle_encoder(train_x)
        output = oracle_decoder(embedding)
        loss = criterion(output, train_y)
        loss.backward()
        oracle_optimizer.step()

    # Evaluate oracle
    oracle_decoder.eval()
    oracle_val_loss = 0.0
    oracle_correct = 0
    oracle_total = 0
    with torch.no_grad():
        for val_x, val_y in oracle_val_loader:
            embedding = oracle_encoder(val_x)
            output = oracle_decoder(embedding)
            oracle_val_loss += criterion(output, val_y).item()
            _, predicted = torch.max(output.data, 1)
            oracle_total += val_y.size(0)
            oracle_correct += (predicted == val_y).sum().item()

    metrics["oracle_val_loss"] = oracle_val_loss / len(oracle_val_loader)
    metrics["oracle_accuracy"] = 100 * oracle_correct / oracle_total

    for agent in agents:
        agent.train()

    return metrics

if __name__ == "__main__":
    config = TrainingConfig.from_args()
    train(config)