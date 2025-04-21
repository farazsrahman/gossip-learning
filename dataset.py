import torch
from torchvision import datasets, transforms

def load_mnist_data(
        digits: list[int] = list(range(10)),
        batch_size: int = 128,
    ):
    """
    Returns train and val only containing specified digits.
    Uses a deterministic hashing function to set aside 10% of the dataset for validation.
    """
    import hashlib

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    def hash_index(index):
        return hashlib.md5(str(index).encode()).hexdigest()

    # Filter for specified digits and split into train/val
    train_indices = []
    val_indices = []
    for i, (_, label) in enumerate(full_dataset):
        if label in digits:
            if int(hash_index(i), 16) % 10 == 0:  # Use 10% for validation
                val_indices.append(i)
            else:
                train_indices.append(i)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader