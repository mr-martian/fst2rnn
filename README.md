# fst2rnn

This is an experiment in converting finite-state transducers (FSTs) into recurrent neural networks (RNNs) by using the FST structure directly in the training process rather than by training on a random sample of the string pairs present in the FST.

## Usage

To try out the code, run `./learn_fst.py -h` for the script arguments or see [`test.sh`](test.sh) for an example.

The code has been tested the following package versions:
- Python 3.8.10
- numpy 1.22.0
- matplotlib 3.1.2
Python must be at least 3.6, but otherwise most versions of the libraries are probably fine.

## Architecture

The model is trained as a simple feedforward network with two inputs and two outputs. The training data consists of each transition of the FST, where an encoding of the source state and input symbol are the inputs and an encoding of the target state and the output symbol are the desired outputs.

The two input vectors are concatenated and passed through two linear layers with tanh activations. Then the output vector is split, with half forming the output state and the other half being passed through a third linear layer with softmax to generate the output character prediction.

To avoid oracle issues, for each transition, transitions leading to its source state are sampled and their predicted output state is used in place of the correct source state for some training iterations.

At prediction time, the representation of the state number becomes the hidden state of the RNN, which then takes the symbols as its only input and output.

## Results and Future Work

In experiments thus far, the loss typically reduces to about half its original value before stabilzing. At this point the output is still typically repetitions of a single character regardless of the input.

A likely source of this difficulty is that the FSTs used in testing so far have fewer transitions than the total number of parameters, probably leading to sparsity problems. This issue could be mitigated by using an FST where the number of transitions is larger than the number of parameters in the trained model, though this situation may prove difficult to construct, since the prediction code currently does not have a way of accounting for epsilon transitions on the source side.

Additionally, other representations of state numbers could be attempted. Currently, the vectors are binary encodings of the state numbers in the training data. They could instead be learned embeddings of one-hot representations of the state numbers, with samples from adjacent transitions potentially used to help ensure that input states and output states had the same representations.
