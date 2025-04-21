# Gossip Learning

Since reading [AI 2027](https://ai-2027.com/), I have been thinking a lot about multi-agent collaboration and specifically *learning*. This project considers a network of learning agents who may want to share information about their representations of the world with neighbors (hence "gossip"). This sort of learning paradigm is interesting for two reasons 
1. It relaxes the need to centralize all data and compute while training an agent
2. The optimization dynamics for any single agent will depend on the toplogy and tasks of the entire network and may be worth studying on its own.

The experiments in this repo formulate this by giving each agent encoder-decoder architecture where decoders don't just recieve the embedding from the encoder, but recieve some cumulative embedding from all of it's neighbors (e.g. avg. embedding, embedding from the parameter avg. encoder, concat embedding, etc.)

## Work Log:

TODO:
- See oracle performance over time on a couple simple topologies
- See how parameter averaging compares to embedding averaging 
- Validate agent encoder itself on far away agent's task to see if they pick up the representation
- See training speed without recieving neighbor updates vs. with receiving neighbor updates.

April 20th 
- Create agent class with encoder, decoder architecture. (DONE)
- Test with two agents, on a single-task toy data (DONE)
- Create Oracle validation loop (DONE)

## Example Usage
The following command launches the training loop using two learners with one seeing digits 0-4 of MNIST and another seeing 5-9.
```python train.py --topology N2_FC --digits-partition N2_SPLIT --n_updates 1000 --batch-size 32 --lr 0.001```
