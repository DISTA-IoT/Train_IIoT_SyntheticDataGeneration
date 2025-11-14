import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        input_dim = kwargs.get('input_dim', 40)
        h_dim = kwargs.get('h_dim', 128)
        dropout = kwargs.get('dropout', 0.1)
        num_layers = kwargs.get('num_layers', 3)
        layer_norm = kwargs.get('layer_norm', False)
        output_dim = kwargs.get('output_dim',1)
        self.mode = kwargs.get('mode')
        
        main_stream = []

        if layer_norm:
            main_stream.append(nn.LayerNorm(input_dim))
        
        curr_output_dim = h_dim
        curr_main_input_dim = input_dim

        for _ in range(num_layers):
            main_stream.append(nn.Linear(curr_main_input_dim, curr_output_dim))
            main_stream.append(nn.ReLU())
            main_stream.append(nn.Dropout(dropout))
            curr_main_input_dim = curr_output_dim
            curr_output_dim = curr_output_dim // 2

        # manifold layer:
        curr_output_dim = 2
        self.manifold_layer = nn.Linear(curr_main_input_dim, curr_output_dim)
        curr_main_input_dim = curr_output_dim


        self.output_module = nn.Sequential(
            nn.Linear(curr_main_input_dim, output_dim),
            (nn.Sigmoid() if output_dim == 1 else nn.Softmax(1)))
        
        self.main_stream = nn.Sequential(*main_stream)
        print("Architecture of the MLP: ")
        print(self.parameters)
            
    def forward(self, x):
        
        main = self.main_stream(x)
        manifold = self.manifold_layer(main)
        return self.output_module(manifold), manifold.detach()
