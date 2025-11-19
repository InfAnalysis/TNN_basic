import torch.nn as nn
import torch
import numpy as np

from quadrature import *


# ********** Activation functions with Derivative **********
# Redefine activation functions and the corresponding local gradient
# sin(x)
class TNN_Sin(nn.Module):
    """TNN_Sin"""
    def forward(self,x):
        return torch.sin(x)

    def grad(self,x):
        return torch.cos(x)

    def grad_grad(self,x):
        return -torch.sin(x)




# ********** Network layers **********
# Linear layer for TNN
class TNN_Linear(nn.Module):
    """
    Applies a batch linear transformation to the incoming data:
        input data: x:[dim, n1, N]
        learnable parameters: W:[dim,n2,n1], b:[dim,n2,1]
        output data: y=Wx+b:[dim,n2,N]

    Parameters:
        dim: dimension of TNN
        out_features: n2
        in_features: n1
        bias: if bias needed or not (boolean)
    """
    def __init__(self, dim, out_features, in_features, bias):
        super(TNN_Linear, self).__init__()
        self.dim = dim
        self.out_features = out_features
        self.in_features = in_features

        self.weight = nn.Parameter(torch.empty((self.dim, self.out_features, self.in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((self.dim, self.out_features, 1)))
        else:
            self.bias = None

    def forward(self,x):
        if self.bias==None:
            if self.in_features==1:
                return self.weight*x
            else:
                return self.weight@x
        else:
            if self.in_features==1:
                return self.weight*x+self.bias
            else:
                return self.weight@x+self.bias

    def extra_repr(self):
        return 'weight.size={}, bias.size={}'.format(
            [self.dim, self.out_features, self.in_features], [self.dim, self.out_features, 1] if self.bias!=None else []
        )


# Scaling layer for TNN.
class TNN_Scaling(nn.Module):
    """
    Define the scaling parameters.

    size:
        [k,p] for Multi-TNN
        [p] for TNN
    """
    def __init__(self, size):
        super(TNN_Scaling, self).__init__()
        self.size = size
        self.alpha = nn.Parameter(torch.empty(self.size))

    def extra_repr(self):
        return 'size={}'.format(self.size)


# Define extra parameters
class TNN_Extra(nn.Module):
    """
    Define extra parameters.
    """
    def __init__(self, size):
        super(TNN_Extra, self).__init__()
        self.size = size
        self.beta = nn.Parameter(torch.empty(self.size))

    def extra_repr(self):
        return 'size={}'.format(self.size)


# ********** TNN architectures **********
# One simple TNN
class TNN(nn.Module):
    """
    Architectures of the simple tensor neural network.
    FNN on each demension has the same size,
    and the input integration points are same in different dinension.
    TNN values and gradient values at data points are provided.

    Parameters:
        dim: dimension of TNN, number of FNNs
        size: [1, n0, n1, ..., nl, p], size of each FNN
        activation: activation function used in hidden layers
        bd: extra function for boundary condition
        grad_bd: gradient of bd
        initializer: initial method for learnable parameters
    """
    def __init__(self, dim, size, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=True, extra_size=False, initializer=['default',None]):
        super(TNN, self).__init__()
        self.dim = dim
        self.size = size
        self.activation = activation()
        self.bd = bd
        self.grad_bd = grad_bd
        self.grad_grad_bd = grad_grad_bd
        self.scaling = scaling
        self.extra_size = extra_size

        self.p = abs(self.size[-1])

        self.init_type = initializer[0]
        self.init_data = initializer[1]

        self.ms = self.__init_modules()
        self.__initialize()

    # Register learnable parameters of TNN module.
    def __init_modules(self):
        modules = nn.ModuleDict()
        # Create two parallel subnetworks per dimension whose outputs will be multiplied
        for i in range(1, len(self.size)):
            bias = True if self.size[i] > 0 else False
            modules['TNN1_Linear{}'.format(i-1)] = TNN_Linear(self.dim,abs(self.size[i]),abs(self.size[i-1]),bias)
            modules['TNN2_Linear{}'.format(i-1)] = TNN_Linear(self.dim,abs(self.size[i]),abs(self.size[i-1]),bias)
        if self.scaling:
            modules['TNN_Scaling'] = TNN_Scaling([self.p])
        if self.extra_size:
            modules['TNN_Extra'] = TNN_Extra(self.extra_size)
        return modules

    # Initialize learnable parameters of TNN module.
    def __initialize(self):
        # default initialization.
        if self.init_type == 'default':
            for i in range(1, len(self.size)):
                for j in range(self.dim):
                    nn.init.orthogonal_(self.ms['TNN1_Linear{}'.format(i-1)].weight[j,:,:])
                    nn.init.orthogonal_(self.ms['TNN2_Linear{}'.format(i-1)].weight[j,:,:])
                if self.size[i] > 0:
                    nn.init.constant_(self.ms['TNN1_Linear{}'.format(i-1)].bias, 0.5)
                    nn.init.constant_(self.ms['TNN2_Linear{}'.format(i-1)].bias, 0.5)
            if self.scaling:
                nn.init.constant_(self.ms['TNN_Scaling'].alpha, 1)
            if self.extra_size:
                nn.init.constant_(self.ms['TNN_Extra'].beta, 1)


    # function to return scaling parameters
    def scaling_par(self):
        if self.scaling:
            return self.ms['TNN_Scaling'].alpha
        else:
            raise NameError('The TNN Module does not have Scaling Parameters')

    # function to return extra parameters
    def extra_par(self):
        if self.extra_size:
            return self.ms['TNN_Extra'].beta
        else:
            raise NameError('The TNN Module does not have Extra Parameters')


    def forward(self,w,x,need_grad=0,normed=True):
        """
        Parameters:
            w: quadrature weights [N]
            x: quadrature points [N]
            need_grad: if return gradient or not

        Returns:
            phi: values of each dimensional FNN [dim, p, N]
            grad_phi: gradient values of each dimensional FNN [dim, p, N]
        """
        # Compute values of each one-dimensional input FNN at each quadrature point.
        if need_grad==0:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Forward process for two subnetworks then multiply
            x1 = x
            x2 = x
            for i in range(1, len(self.size) - 1):
                x1 = self.ms['TNN1_Linear{}'.format(i-1)](x1)
                x1 = self.activation(x1)
                x2 = self.ms['TNN2_Linear{}'.format(i-1)](x2)
                x2 = self.activation(x2)
            out1 = self.ms['TNN1_Linear{}'.format(len(self.size) - 2)](x1)
            out2 = self.ms['TNN2_Linear{}'.format(len(self.size) - 2)](x2)
            if bd_value==None:
                phi = out1*out2
            else:
                phi = (out1*out2)*bd_value
            # normalization
            if normed:
                norm = torch.sqrt(torch.sum(w*phi**2, dim=2) + 1e-10).unsqueeze(dim=-1)
                return phi / norm
            else:
                return phi


        # Compute values and gradient values of each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad==1:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # Compute forward and backward process simutaneously for two subnetworks
            x1 = x
            x2 = x
            grad_x1 = self.ms['TNN1_Linear{}'.format(0)].weight
            grad_x2 = self.ms['TNN2_Linear{}'.format(0)].weight
            for i in range(1, len(self.size) - 1):
                x1 = self.ms['TNN1_Linear{}'.format(i-1)](x1)
                grad_x1 = self.activation.grad(x1)*grad_x1
                grad_x1 = self.ms['TNN1_Linear{}'.format(i)].weight@grad_x1
                x1 = self.activation(x1)

                x2 = self.ms['TNN2_Linear{}'.format(i-1)](x2)
                grad_x2 = self.activation.grad(x2)*grad_x2
                grad_x2 = self.ms['TNN2_Linear{}'.format(i)].weight@grad_x2
                x2 = self.activation(x2)
            out1 = self.ms['TNN1_Linear{}'.format(len(self.size) - 2)](x1)
            out2 = self.ms['TNN2_Linear{}'.format(len(self.size) - 2)](x2)
            # Product rule and boundary condition
            if self.bd==None:
                phi = out1*out2
                grad_phi = grad_x1*out2 + out1*grad_x2
            else:
                phi = (out1*out2)*bd_value
                grad_phi = (out1*out2)*grad_bd_value + (grad_x1*out2 + out1*grad_x2)*bd_value
            # normalization
            if normed:
                norm = torch.sqrt(torch.sum(w*phi**2, dim=2) + 1e-10).unsqueeze(dim=-1)
                return phi / norm, grad_phi / norm
            else:
                return phi, grad_phi


        # Compute values, first and second gradient values for each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad==2:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # get grad_grad_bd values
            if self.grad_grad_bd==None:
                grad_grad_bd_value = None
            else:
                grad_grad_bd_value = self.grad_grad_bd(x)

            # Compute forward and backward process simutaneously for two subnetworks
            x1 = x
            x2 = x
            grad_x1 = self.ms['TNN1_Linear{}'.format(0)].weight
            grad_x2 = self.ms['TNN2_Linear{}'.format(0)].weight
            grad_grad_x1 = torch.zeros_like(grad_x1)
            grad_grad_x2 = torch.zeros_like(grad_x2)
            for i in range(1, len(self.size) - 1):
                # subnetwork 1
                x1 = self.ms['TNN1_Linear{}'.format(i-1)](x1)
                grad_grad_x1 = self.activation.grad_grad(x1)*(grad_x1**2)+self.activation.grad(x1)*grad_grad_x1
                grad_grad_x1 = self.ms['TNN1_Linear{}'.format(i)].weight@grad_grad_x1
                grad_x1 = self.activation.grad(x1)*grad_x1
                grad_x1 = self.ms['TNN1_Linear{}'.format(i)].weight@grad_x1
                x1 = self.activation(x1)
                # subnetwork 2
                x2 = self.ms['TNN2_Linear{}'.format(i-1)](x2)
                grad_grad_x2 = self.activation.grad_grad(x2)*(grad_x2**2)+self.activation.grad(x2)*grad_grad_x2
                grad_grad_x2 = self.ms['TNN2_Linear{}'.format(i)].weight@grad_grad_x2
                grad_x2 = self.activation.grad(x2)*grad_x2
                grad_x2 = self.ms['TNN2_Linear{}'.format(i)].weight@grad_x2
                x2 = self.activation(x2)
            out1 = self.ms['TNN1_Linear{}'.format(len(self.size) - 2)](x1)
            out2 = self.ms['TNN2_Linear{}'.format(len(self.size) - 2)](x2)
            if self.bd==None:
                phi = out1
                grad_phi = grad_x1
                grad_grad_phi = grad_grad_x1
                # combine via product
                phi = out1*out2
                grad_phi = grad_x1*out2 + out1*grad_x2
                grad_grad_phi = grad_grad_x1*out2 + 2*grad_x1*grad_x2 + out1*grad_grad_x2
            else:
                base_phi = out1*out2
                base_grad = grad_x1*out2 + out1*grad_x2
                base_grad_grad = grad_grad_x1*out2 + 2*grad_x1*grad_x2 + out1*grad_grad_x2
                phi = base_phi*bd_value
                grad_phi = base_phi*grad_bd_value + base_grad*bd_value
                grad_grad_phi = base_phi*grad_grad_bd_value + 2*base_grad*grad_bd_value + base_grad_grad*bd_value
            # normalization
            if normed:
                norm = torch.sqrt(torch.sum(w*phi**2, dim=2) + 1e-10).unsqueeze(dim=-1)
                return phi / norm, grad_phi / norm, grad_grad_phi / norm
            else:
                return phi, grad_phi, grad_grad_phi

    def extra_repr(self):
        return '{}\n{}'.format('Architectures of one TNN(dim={},rank={}) which has {} FNNs:'.format(self.dim,self.p,self.dim),\
                'Each FNN has size: {}'.format(self.size))



def main():
    pass


if __name__ == '__main__':
    main()
