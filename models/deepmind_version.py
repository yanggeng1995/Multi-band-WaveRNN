import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveRNN(nn.Module):
    def __init__(self, hidden_size=384, quantization=256):
        super(WaveRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.split_size = hidden_size // 2
        
        # The main matmul
        self.R = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        
        # Output fc layers
        self.O1 = nn.Linear(self.split_size, self.split_size)
        self.O3 = nn.Linear(self.split_size, self.split_size)

        #band one fc layers
        self.one_O2 = nn.Linear(self.split_size, quantization)
        self.one_O4 = nn.Linear(self.split_size, quantization)
        #band two fc layers
        self.two_O2 = nn.Linear(self.split_size, quantization)
        self.two_O4 = nn.Linear(self.split_size, quantization)
        #band three fc layers
        self.three_O2 = nn.Linear(self.split_size, quantization)
        self.three_O4 = nn.Linear(self.split_size, quantization)
        #band four fc layers
        self.four_O2 = nn.Linear(self.split_size, quantization)
        self.four_O4 = nn.Linear(self.split_size, quantization)
        
        # Input fc layers
        self.I_coarse = nn.Linear(2 * 4, 3 * self.split_size, bias=False)
        self.I_fine = nn.Linear(3 * 4, 3 * self.split_size, bias=False)

        # biases for the gates
        self.bias_u = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_r = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_e = nn.Parameter(torch.zeros(self.hidden_size))
        
        # display num params
        self.num_params()

        
    def forward(self, prev_y, prev_hidden, current_coarse):
        """
          prev_y: [B, 8]
          prev_hidden: [B, 384]
          current_coarse: [B, 4]
        """ 
        # Main matmul - the projection is split 3 ways
        R_hidden = self.R(prev_hidden)
        R_u, R_r, R_e, = torch.split(R_hidden, self.hidden_size, dim=1)
        
        # Project the prev input 
        coarse_input_proj = self.I_coarse(prev_y)
        I_coarse_u, I_coarse_r, I_coarse_e = \
            torch.split(coarse_input_proj, self.split_size, dim=1)
        
        # Project the prev input and current coarse sample
        fine_input = torch.cat([prev_y, current_coarse], dim=1)
        fine_input_proj = self.I_fine(fine_input)
        I_fine_u, I_fine_r, I_fine_e = \
            torch.split(fine_input_proj, self.split_size, dim=1)
        
        # concatenate for the gates
        I_u = torch.cat([I_coarse_u, I_fine_u], dim=1)
        I_r = torch.cat([I_coarse_r, I_fine_r], dim=1)
        I_e = torch.cat([I_coarse_e, I_fine_e], dim=1)
        
        # Compute all gates for coarse and fine 
        u = F.sigmoid(R_u + I_u + self.bias_u)
        r = F.sigmoid(R_r + I_r + self.bias_r)
        e = F.tanh(r * R_e + I_e + self.bias_e)
        hidden = u * prev_hidden + (1. - u) * e
        
        # Split the hidden state
        h_c, h_f = torch.split(hidden, self.split_size, dim=1)
        
        # Compute outputs 
        out_c = F.relu(self.O1(h_c))
        out_f = F.relu(self.O3(h_f))

        #band one
        one_c = self.one_O2(out_c)
        one_f = self.one_O4(out_f)
        #band two
        two_c = self.two_O2(out_c)
        two_f = self.two_O4(out_f)
        #band three
        three_c = self.three_O2(out_c)
        three_f = self.three_O4(out_f)
        #band four
        four_c = self.four_O2(out_c)
        four_f = self.four_O4(out_f)

        c = torch.cat([one_c, two_c, three_c, four_c], dim=0)
        f = torch.cat([one_f, two_f, three_f, four_f], dim=0)

        return c, f, hidden
        
    def get_initial_hidden(self, batch_size=1):
        device = next(self.parameters()).device  # use same device as parameters
        return torch.zeros(batch_size, self.hidden_size, device=device)
    
    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)
