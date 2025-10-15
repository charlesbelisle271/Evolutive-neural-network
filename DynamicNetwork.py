from torch import nn
from torch.optim import SGD
import torch
import random as rd
from torch.nn.functional import relu
import copy
import numpy as np


class DynamicNetwork(nn.Module):

    
    
    def __init__(self, dummy, nb_of_labels):
        super().__init__()
        self.dummy=dummy
        self.nb_of_labels=nb_of_labels
        lin_input=self.dummy.shape[0]*self.dummy.shape[1]*self.dummy.shape[2]
        self.conv_skeleton=nn.ModuleList([])
        self.linear_skeleton=nn.ModuleList([
            nn.Flatten(),
            nn.Linear(lin_input, 1024),
            nn.Linear(1024,512),
            nn.Linear(512,nb_of_labels)
        ])
        
        self.score=0.0 
        self.learning_rate=3e-2
        self.loss_fn=nn.CrossEntropyLoss()
        
        
        
    def evolve(self):
        #Here we chose a random mutation indexed by a number from 1 to 6. We keep the value of the mutation taking place, along with
        #other parameters like the location of the mutation in the network, a bool last representing if the mutation took place on the
        #last layer of a particular network...
        evolution=rd.randint(1,6)
        if evolution==1:
            mutation=self.add_linear_layer()
            print("Mutation #1 took places")

        elif evolution==2:
            mutation=self.remove_linear_layer()
            print("Mutation #2 took places")
            
        elif evolution==3:
            mutation=self.change_nb_of_neurones()
            print("Mutation #3 took places")
            
        elif evolution==4: 
            mutation=self.change_learning_rate()
            print("Mutation #4 took places")

        elif evolution==5: 
            mutation=self.add_conv_layer()
            print("Mutation #5 took places")

        elif evolution==6: 
            mutation=self.remove_conv_layer()
            print("Mutation #6 took places")
    
        return mutation
       
    def forward(self, x): 
        
        for conv_layer in self.conv_skeleton:
            x=relu(conv_layer(x))
            
        for lin_layer in self.linear_skeleton[0:-1]:
            x = relu(lin_layer(x))
           
        x = self.linear_skeleton[-1](x)
     
        return x
    
    def get_conv_output_features(self): 
        #This method gets the number of features of our data after leaving the convolutive part of our model. This is done
        #by feeding a dummy tensor taken from our dataset and looking at its shape when leaving the convolutive skeleton
        
        im=self.dummy
        for layer in self.conv_skeleton:
            im=layer(im)
            
        return im.shape[0]*im.shape[1]*im.shape[2]

    def conv_layer_is_valid(self, kernel_size, stride): 
        #This method checks that, given a pair (kernel_size, stride), a convolutive layer with these parameters fits in our network.

        im=self.dummy #We'll use this dummy to get the output of the convolutive skeleton before our new conv layer is added
        for layer in self.conv_skeleton:
            im=layer(im)
        dim=im.shape[1]#We use square kernels, so the height dimension is always the same as the width dimension
        
        
        new_dim=np.floor((dim-kernel_size+stride)/stride)
        if new_dim>0: return True
        else: return False
        
    def get_kernel_size_and_stride(self):

        #This method returns a random pair of integer (kernel_size, stride) which will be used to create a new convolutive layer. The 
        #method conv_layer_is_valid checks that the pair (kernel_size, stride) fits with the shape of the input images

        stride=rd.randint(1,5)
        kernel_size=rd.randint(1,7)
        while not self.conv_layer_is_valid(kernel_size, stride):
            stride=rd.randint(1,5)
            kernel_size=rd.randint(1,7)
        
        return kernel_size, stride

    def exist_valid_kernel_size(self):
        #This method checks if there exists a valid kernel_size for an hypothetic convolutive layer to be added to our 
        #conv_skeleton. This is done by looking at the output features of our actual convolutive network: if it is bigger than one,
        #then a convolutive layer with a kernel_size of two(the smallest possible) would fit.

        im=self.dummy#This will be used as a dummy to get the shape of the output tensor of our conv_skeleton
        for layer in self.conv_skeleton:
            im=layer(im)

        if im.shape[1]>1:return True
        else: return False

    def add_conv_layer(self):
        #Here we add a convolutive layer to our conv_skeleton, if such a layer that's compatible with the dimension of our data and our 
        #actual conv_skeleton exists. If no such convolutive layer exists, we ask for another mutation.
        #added will keep in memory if a layer was added or not to the layer.
        added=True
        if len(self.conv_skeleton)==0:
            
            kernel_size=rd.randint(1,5)
            stride=rd.randint(1,3)
            new_layer_in_channels=self.dummy.shape[0] #The number of channels of the input is that of the data
            new_layer_out_channels=rd.randint(1,32)
            new_layer=nn.Conv2d(new_layer_in_channels, new_layer_out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=1)
            self.conv_skeleton.append(new_layer)
            
        elif self.exist_valid_kernel_size():
            kernel_size, stride = self.get_kernel_size_and_stride()
            nb_of_out_channels=rd.randint(1,32)
            nb_of_in_channels=self.conv_skeleton[-1].out_channels
            new_layer=nn.Conv2d(nb_of_in_channels, nb_of_out_channels, kernel_size=kernel_size, stride=stride)
            self.conv_skeleton.append(new_layer)

        else:
        
            added=False #Nothing was added

        
        if added:
            #We need to update the number of features accepted by our first linear layer
            nb_of_in_features=self.get_conv_output_features()
            nb_of_out_features=self.linear_skeleton[1].out_features
            first_linear_layer=nn.Linear(nb_of_in_features, nb_of_out_features)
            self.linear_skeleton[1]=first_linear_layer
            return 5, added
        else:return self.evolve()
        
    def remove_conv_layer(self):
        #Each time we remove a layer that's in between two other layers, we need to connect the output to the input of 
        #the layers that are now to be following each other. If the removed layer is at the end, then there is nothing to do
        #If there aren't any convolutive layers, we ask for another mutation
        if len(self.conv_skeleton)==0:return self.evolve()

        else:
            location=rd.randint(0,len(self.conv_skeleton)-1)
            last=location==(len(self.conv_skeleton)-1)
            

            del self.conv_skeleton[location]

            if location==0:
                if len(self.conv_skeleton)==0:pass
                else:
                    first_layer_stride=self.conv_skeleton[0].stride 
                    first_layer_kernel_size=self.conv_skeleton[0].kernel_size
                    first_layer_out_channels=self.conv_skeleton[0].out_channels
                    first_layer_in_channels=self.dummy.shape[0]
                    first_layer=nn.Conv2d(first_layer_in_channels, first_layer_out_channels, 
                                      kernel_size=first_layer_kernel_size, stride=first_layer_stride)
                    self.conv_skeleton[0]=first_layer
            elif last:pass
                
            else:


                
                location_layer_in_channels=self.conv_skeleton[location-1].out_channels
                location_layer_out_channels=self.conv_skeleton[location].out_channels
                location_layer_stride=self.conv_skeleton[location].stride 
                location_layer_kernel_size=self.conv_skeleton[location].kernel_size
                location_layer=nn.Conv2d(location_layer_in_channels, location_layer_out_channels,
                                          kernel_size = location_layer_kernel_size ,stride = location_layer_stride)
                self.conv_skeleton[location]=location_layer

            #We now need to update the number of input features of the first linear layer
            #to make it compatible with our new convolutive skeleton
            nb_of_input_features=self.get_conv_output_features()
            nb_of_output_features=self.linear_skeleton[1].out_features
            first_linear_layer=nn.Linear(nb_of_input_features, nb_of_output_features)
            self.linear_skeleton[1]=first_linear_layer
            return 6, location, last

    def remove_linear_layer(self):
        
        depth = len(self.linear_skeleton) - 1 
        lin_input=self.get_conv_output_features()

        #If the depth is less than two, then there is only one linear layer and we cannot remove it. Thus we ask for another 
        #mutation
        if depth<2:return self.evolve()
        else:
            location=rd.randint(1,depth)
            last=location==depth
            del self.linear_skeleton[location]
            #If the last layer is deleted, we need to make sure that the network outputs a number of features 
            #equal to the number of labels of the classification task. If we delete the first layer, we 
            #need to make sure that the layer that replaces it has a number of input features equals to
            #the output of the flatten layer, which is here equal to the variable lin_input
            
            if last:
                #If after deletion our depth is equal to two, than we need to adress the case that skeleton[-2]
                #refers to the flatten layer, which doesn,t have the .out_features attribute
                if len(self.linear_skeleton)==2: last_layer_in_features=lin_input
                else: last_layer_in_features=self.linear_skeleton[-2].out_features
                last_layer_out_features=self.linear_skeleton[-1].out_features
                last_layer=nn.Linear(last_layer_in_features,self.nb_of_labels)
                self.linear_skeleton[-1]=last_layer
            else:
                #If we removed the first layer, we then need to make sure that the layer that replaces it
                #(the previously second layer) has the right number of input(lin_input). Otherwise we juste
                #need to connect together the previously precedding and following layers.
                if location==1:location_layer_in_features=lin_input
                else:location_layer_in_features=self.linear_skeleton[location-1].out_features
                location_layer_out_features=self.linear_skeleton[location].out_features
                location_layer=nn.Linear(location_layer_in_features, location_layer_out_features)
                self.linear_skeleton[location]=location_layer

            return 2, location, last

    def add_linear_layer(self):
        depth = len(self.linear_skeleton) - 1
        location=rd.randint(1,depth)
        last=location==depth
        lin_input=self.get_conv_output_features()
         
        #To make sure our new layers accepts the good amount of input features, we need to get
        #the number of ouput features of the layer that preceeds it
        #If the new layers is inserted at index one, it's input needs to be that of the output
        #of the flatten layer, lin_input.
        if location==1:new_layer_in_features=lin_input
        else :new_layer_in_features=self.linear_skeleton[location-1].out_features
        new_layer_out_features=rd.randint(1,1000)  
        new_layer=nn.Linear(new_layer_in_features, new_layer_out_features)
        
        #We also need to make sure that the layer that follows our new layer accepts as input the output 
        #of the layer to be added
        following_layer_in_features=new_layer_out_features
        following_layer_out_features=self.linear_skeleton[location].out_features
        following_layer=nn.Linear(following_layer_in_features,following_layer_out_features)

        self.linear_skeleton[location]=following_layer
        self.linear_skeleton.insert(location, new_layer)
        return 1, location, last
            
    def change_nb_of_neurones(self):
        #Here we change the number of neurones of a specific layer. We cannot change the number of neurones of 
        #the last layer since it needs to ouput 10 labels. If the network only has one layer(depth=1), we add 
        #a layer at the end of the network.

            depth=len(self.linear_skeleton)-1

            if depth>1:
                location=rd.randint(1,depth-1)
                while(location==depth):
                    location=rd.randint(1,depth-1)
                #Here we chose a random number of neurones in the interval 
                #[abs(nb of neurones of current layer-10), nb of neurones of current layer+1]
                #we use the absolute value to make sure the lower bound of the interval is positive
                
                current_layer_in_features=self.linear_skeleton[location].in_features
                current_layer_out_features=self.linear_skeleton[location].out_features
                
                new_layer_out_features=rd.randint(abs(current_layer_out_features-10),current_layer_out_features+10)
                new_layer_in_features=current_layer_in_features
                new_layer=nn.Linear(new_layer_in_features, new_layer_out_features)
                
                next_layer_in_features=new_layer_out_features
                next_layer_out_features=self.linear_skeleton[location+1].out_features
                next_layer=nn.Linear(next_layer_in_features, next_layer_out_features)
                
                self.linear_skeleton[location]=new_layer
                self.linear_skeleton[location+1]=next_layer
                return 3, location
            
            else:return self.evolve()
        
    def change_learning_rate(self):
        self.learning_rate=rd.uniform(1e-4,1)
        return [4]
    
def compete(player1, player2, nb_of_epochs, train_dataloader, test_dataloader):

    re_train=True
    if player1.score[0]>player2.score[0]:
        player2=copy.deepcopy(player1)
        mutation=player2.evolve()
        for param in player2.parameters():
            param.requires_grad=False

        if mutation[0]==1:#We added a linear layer
            
            if mutation[2]:
                #We added the layer at the end of the linear skeleton
                #We only need to train the weights of the last layer and we keep the weights of all the other layers
                for param in player2.linear_skeleton[-1].parameters():
                    param.requires_grad=True
            else:
                #The layer was added at location
                #We need to update the weights of the layers at location and location+1
                location=mutation[1]
                for parem in player2.linear_skeleton[location].parameters():
                    parem.requires_grad=True
                for parem in player2.linear_skeleton[location+1].parameters():
                    parem.requires_grad=True

        elif mutation[0]==2:#We removed a linear layer
            
            if mutation[2]:
                for param in player2.linear_skeleton[-1].parameters():
                    param.requires_grad=True
            else:
                location=mutation[1]
                for param in player2.linear_skeleton[location].parameters():
                    param.requires_grad=True

        elif mutation[0]==3:#We changed the number of neurones of a linear layer
            
            location=mutation[1]
            for param in player2.linear_skeleton[location].parameters():
                param.requires_grad=True
            for param in player2.linear_skeleton[location+1].parameters():
                param.requires_grad=True
        
        elif mutation[0]==4:#We changed the learning rate, so we don't need to retrain the parameters
            re_train=False
            
        elif mutation[0]==5:#We added a convolutive layer at the end of the conv_skeleton
            
            for param in player2.conv_skeleton[-1].parameters():
                param.requires_grad=True
            for param in player2.linear_skeleton[1].parameters():
                param.requires_grad=True

        elif mutation[0]==6:#We removed a convolutive layer at location or at the end
            
            empty=len(player2.conv_skeleton)==0

            if empty:pass
            elif not empty:
                if mutation[2]:
                    for param in player2.conv_skeleton[-1].parameters():
                        param.requires_grad=True
                else:
                    location=mutation[1]
                    for param in player2.conv_skeleton[location].parameters():
                        param.requires_grad=True
            for param in player2.linear_skeleton[1].parameters():
                param.requires_grad=True

        if re_train:
            #print(f"Mutation #{mutation[0]} took place")
            #print(f"The following parameters are to be trained")
            #for element in filter(lambda p:p.requires_grad, player2.parameters()):
                #print(element)
            print(f"player 2 looks like {player2}")
            optimizer=SGD(filter(lambda p: p.requires_grad, player2.parameters()),lr=player2.learning_rate)
            for epoch in range(nb_of_epochs):
                print(f"Epoch {epoch+1}\n-------------------------------")
                train(train_dataloader, player2, player2.loss_fn, optimizer)
                test(test_dataloader, player2, player2.loss_fn)
            player2.score=test(test_dataloader, player2, player2.loss_fn)

    elif player2.score[0]>player1.score[0]:

        player1=copy.deepcopy(player2)
        mutation=player1.evolve()
        print(f"debug mutation equals {mutation}")
        for param in player1.parameters():
            param.requires_grad=False

        if mutation[0]==1:#We added a linear layer
            
            if mutation[2]:#We added the layer at the end of the linear skeleton
                #We only need to train the weights of the last layer and we keep the weights of all the other layers
                for param in player1.linear_skeleton[-1].parameters():
                    param.requires_grad=True
            else:#The layer was added at location
                #We need to update the weights of the layers at location and location+1
                location=mutation[1]
                for param in player1.linear_skeleton[location].parameters():
                    param.requires_grad=True
                for param in player1.linear_skeleton[location+1].parameters():
                    param.requires_grad=True

        elif mutation[0]==2:#We removed a linear layer
                
            if mutation[2]:
                for param in player1.linear_skeleton[-1].parameters():
                    param.requires_grad=True
            else:
                location=mutation[1]
                for param in player1.linear_skeleton[location].parameters():
                    param.requires_grad=True

        elif mutation[0]==3:#We changed the number of neurones of a linear layer
            
            location=mutation[1]
            for param in player1.linear_skeleton[location].parameters():
                param.requires_grad=True
            for param in player1.linear_skeleton[location+1].parameters():
                param.requires_grad=True
        
        elif mutation[0]==4:#We changed the learning rate, so we don't need to retrain the parameters
            re_train=False
            

        elif mutation[0]==5:#We added a convolutive layer at the end of the conv_skeleton
            
            for param in player1.conv_skeleton[-1].parameters():
                param.requires_grad=True
            for param in player1.linear_skeleton[1].parameters():
                param.requires_grad=True

        elif mutation[0]==6:#We removed a convolutive layer at location or at the end
            
            empty=len(player1.conv_skeleton)==0
            if empty:pass
            elif not empty:

                if mutation[2]:
                    for param in player1.conv_skeleton[-1].parameters():
                        param.requires_grad=True
                else:
                    location=mutation[1]
                    for param in player1.conv_skeleton[location].parameters():
                        param.requires_grad=True
            for param in player1.linear_skeleton[1].parameters():
                param.requires_grad=True

        if re_train:
            #print(f"Mutation #{mutation[0]} took place")
            #print(f"The following parameters are to be trained")
            #for element in filter(lambda p:p.requires_grad, player1.parameters()):
                #print(element)
            print(f"player1 looks like {player1}")
            optimizer=SGD(filter(lambda p: p.requires_grad, player1.parameters()), lr=player1.learning_rate)
            for epoch in range(nb_of_epochs):
                print(f"Epoch {epoch+1}\n-------------------------------")
                train(train_dataloader, player1, player1.loss_fn, optimizer)
                test(test_dataloader, player1, player1.loss_fn)
            player1.score=test(test_dataloader, player1, player1.loss_fn)

    return player1, player2

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
       

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
             
def test(dataLoader, model, loss_fn):
    
    size=len(dataLoader.dataset)
    num_batches = len(dataLoader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataLoader:
            pred = model(X)
            test_loss+=loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        test_loss/=num_batches
        correct/=size
        
        print(f"Test Error: /n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}/n")
    return correct*100, test_loss
        



