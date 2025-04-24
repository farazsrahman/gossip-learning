from dataclasses import dataclass

@dataclass
class TrainingConfig:
    topology: list[list[int]]
    digits_partition: list[list[int]]
    n_updates: int
    batch_size: int
    learning_rate: float
    encoder_cumulator: str  # 'embedding_avg' or 'encoder_avg'

    @classmethod
    def from_args(cls):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--topology', type=str, required=True, help='Name of the topology to use')
        parser.add_argument('--digits-partition', type=str, required=True, help='Name of the digits partition to use')
        parser.add_argument('--n_updates', type=int, default=1000, help='Number of updates to train')
        parser.add_argument('--val_freq', type=int, default=100, help='Number of updates between validation')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--encoder-cumulator', type=str, default='embedding_avg', 
                          choices=['embedding_avg', 'encoder_avg'],
                          help='Method to cumulate encoder outputs: embedding_avg (default) or encoder_avg')
        
        args = parser.parse_args()

        topology = get_topology(args.topology)
        digits_partition = get_digits_partition(args.digits_partition)

        assert len(topology) == len(digits_partition), "Length of topology adjacency matrix and list of partitions must be the same"

        return cls(
            topology=topology,
            digits_partition=digits_partition,
            n_updates=args.n_updates,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            encoder_cumulator=args.encoder_cumulator
        )

def get_topology(topology_name: str):
    if topology_name == "N1_FC":
        return N1_FC_TOPOLOGY
    elif topology_name == "N2_FC":
        return N2_FC_TOPOLOGY
    elif topology_name == "N4_RING":
        return N4_RING_TOPOLOGY
    elif topology_name == "N4_DISCONNECTED":
        return N4_DISCONNECTED_TOPOLOGY
    elif topology_name == "N4_FC":
        return N4_FC_TOPOLOGY
    else:
        raise ValueError(f"Invalid topology: {topology_name}")

def get_digits_partition(partition_name: str):
    if partition_name == "N1_ALL":
        return N1_ALL_DIGITS
    elif partition_name == "N2_ALL":
        return N2_ALL_DIGITS
    elif partition_name == "N2_SPLIT":
        return N2_SPLIT_DIGITS
    elif partition_name == "N4_SPLIT_DIGITS":
        return N4_SPLIT_DIGITS
    else:
        raise ValueError(f"Invalid digits partition: {partition_name}")


# N1
N1_FC_TOPOLOGY = [[0]]
N1_ALL_DIGITS = [list(range(10))]

# N2
N2_DISCONNECTED_TOPOLOGY = [[0, 0],
                            [0, 0]]
N2_FC_TOPOLOGY =  [[0, 1], 
                   [1, 0]]

N2_ALL_DIGITS = [list(range(10)), list(range(10))]

N2_SPLIT_DIGITS = [list(range(5)), list(range(5, 10))]


# N4
N4_DISCONNECTED_TOPOLOGY = [[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]

N4_FC_TOPOLOGY = [[0, 1, 1, 1],
                  [1, 0, 1, 1],
                  [1, 1, 0, 1],
                  [1, 1, 1, 0]]

N4_LIN_TOPOLOGY = [[0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0]]

N4_RING_TOPOLOGY = [[0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0]]

N4_ALL_DIGITS = [list(range(10)), list(range(10)), list(range(10)), list(range(10))]

N4_SPLIT_DIGITS = [list(range(0,2)), list(range(2,5)), list(range(5,7)), list(range(7,10))]

