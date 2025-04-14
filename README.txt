This code utilizes a Monte Carlo Tree Search algorithm combined with a CNN to evaluate the policy and value of a state. We initialize multiple games of Self Play 
in parallel which allows us to perform the forward pass inside the MCTS in parallel for different states.

The CNN is composed of: 
a startblock,
a backbone,
a value head,
a policy head.
There are num_hidden(channel size) times 3x3 kernels. The input goes through the start block and the backbone which is formed of ResBlocks that makes sure each input is converted to be channel size after applying batchnorm and convolutional layers. The output from the backbone is fed into the value head and the policy head to obtain policy and value logits.

The MCTS is initialized with a set number of parallel games or trees.
We develop each MCTS, by picking the child node with the highest PUCT value until we reach either a leaf node or and end state. If it is a leaf node instead of performing a roll out, we backpropagate the value obtained from the CNN. All forward passes through the CNN are done in parallel batches. If it is a terminal state we backpropagate the end game value.
The MCTS simulation is ran a fixed number of times and the true policy of the position is based on  the final distribution of number of visits for each child node at the end of the simulation. This entire process constitutes one round of self play.  

After every round of self play the true policy is recorded in each games memory. At the end of each self play game the true values of the positions are associated with the states of the gameplay and appended to the replay buffer.

After playing a fixed amount of parallel games we train by reshuffling the replay buffer and adjust the weights of the CNN based on the loss function.
We train on the same replay buffer a an epoch number of times and save the model weights based on the save frequency for future training.

