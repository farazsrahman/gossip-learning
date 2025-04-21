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

## Description

A distributed learning framework where multiple agents train on different subsets of MNIST digits. Each agent has an encoder and decoder, and can share embeddings with neighboring agents based on a specified topology. We use an "Oracle" agent with access to the encoders of all agents to measure performance of the group.

## Example Usage
The following command launches the training loop using two learners with one seeing digits 0-4 of MNIST and another seeing 5-9.
```python train.py --topology N2_FC --digits-partition N2_SPLIT --n_updates 1000 --batch-size 32 --lr 0.001```
