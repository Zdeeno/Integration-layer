
import torch
import math
from torch.autograd import Function
from torch.nn import Module, Parameter, init, CrossEntropyLoss, ModuleList, Linear


# Network
class Net(Module):

    def __init__(self, input_size, hidden_sizes, output_size, lr):
        super(Net, self).__init__()
        max_potential = 3.0
        self.layers = ModuleList()
        self.layers.append(Integrating(input_size, hidden_sizes[0], max_potential))

        for i in range(len(hidden_sizes) - 1):
            layer = Integrating(hidden_sizes[i], hidden_sizes[i+1], max_potential, self.layers[-1])
            self.layers.append(layer)

        # self.layers.append(Integrating(hidden_sizes[-1], output_size, max_potential, self.layers[-1]))
        self.layers.append(Linear(hidden_sizes[-1], output_size))
        self.loss = CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reset_potentials(self):
        for layer in self.layers:
            if type(layer).__name__ == "Integrating":
                layer.reset_potentials()


# Layer
class Integrating(Module):

    def __init__(self, in_features, out_features, p_thresh, prev_layer=None):
        super(Integrating, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features).float())
        self.potential = torch.Tensor(out_features).float()
        self.p_thresh = p_thresh
        self.reset_parameters()
        self.prev_layer = prev_layer
        self.has_potentials = True

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(10))
        init.constant_(self.potential, 1)

    def reset_potentials(self):
        init.constant_(self.potential, 1)

    def forward(self, input):
        return IntegrateFunction.apply(input, self.weight, self.potential, self.p_thresh)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.potential is not None
        )


# Function
class IntegrateFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, potential, p_thresh):
        out = input.mm(weight.t())

        ctx.save_for_backward(input, weight, potential)

        potential[torch.abs(potential) > p_thresh] = 0
        potential += out[0]

        ret = torch.zeros_like(potential)
        ret[torch.abs(potential) > p_thresh] = potential[torch.abs(potential) > p_thresh]

        ret = ret.view(1, list(ret.size())[0])  # flatten
        return ret

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, potential = ctx.saved_tensors
        assert not torch.isnan(weight).any()
        grad_input = grad_weight = None
        grad = grad_output  # * potential

        if ctx.needs_input_grad[0]:
            grad_input = grad.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad.t().mm(input)

        return grad_input, grad_weight, None, None, None