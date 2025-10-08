# Evolutive-neural-network

Abastract
Inspired by a research paper describing a method to automate the discovery of neural network models[https://arxiv.org/abs/1703.01041], the DynamicNetwork module enables the user to 
creat a DynamicNetwork object, which, at initialization, is a simple linear network without convolution, and to evolve it via the evolve() method. A random mutation(adding or
removal of convolutive or linear layers, of neurone, the changing of the learning rate...) is then picked and applied to the object. Via the compete function from the same module,
the user can make two different DynamicNetwork object compete with each other in order to keep the better one, make an infant copy of it, and kill the worst one. The infant 
inherits the structure and weights of its parent and then goes through one mutation. The aim of this scheme is to have a population of many DynamicNetwork objects
and make them compete with each other in the hope of discovering a model which is efficent on a given dataset
An automated evolutive neural network inspired by a research team's article on the same subject[https://arxiv.org/abs/1703.01041]
