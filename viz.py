import networkx as nx
import matplotlib.pyplot as plt
from config import TrainingConfig

def visualize_agents(config: TrainingConfig):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with their assigned digits as labels
    for i, digits in enumerate(config.digits_partition):
        G.add_node(i, label=f"Agent {i}\nDigits: {digits}")
    
    # Add edges based on topology
    for i in range(len(config.topology)):
        for j in range(len(config.topology[i])):
            if config.topology[i][j] == 1:
                G.add_edge(i, j)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes with labels
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
    nx.draw_networkx_labels(G, pos, nx.get_node_attributes(G, 'label'), font_size=10)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    
    # Save the plot
    plt.axis('off')
    plt.savefig('plots/agent_graph.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_metrics(metrics: dict, config: TrainingConfig):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot validation loss
    ax1.set_title('Validation Loss Over Time')
    for i in range(len(config.digits_partition)):
        agent_id = f'agent_{i}'
        ax1.plot(metrics[agent_id]['val_loss'], 
                label=f'Agent {i} (digits {config.digits_partition[i]})')
    ax1.plot(metrics['oracle']['val_loss'], label='Oracle', linestyle='--', color='black')
    ax1.set_xlabel('Validation Step (x100 updates)')
    ax1.set_ylabel('Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.set_title('Accuracy Over Time')
    for i in range(len(config.digits_partition)):
        agent_id = f'agent_{i}'
        ax2.plot(metrics[agent_id]['accuracy'], 
                label=f'Agent {i} (digits {config.digits_partition[i]})')
    ax2.plot(metrics['oracle']['accuracy'], label='Oracle', linestyle='--', color='black')
    ax2.set_xlabel('Validation Step (x100 updates)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('plots/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()