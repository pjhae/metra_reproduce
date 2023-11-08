import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# For Metra
class Phi(nn.Module):
    def __init__(self, num_inputs, args):
        super(Phi, self).__init__()
        
        self.lr = args.lr
        self.skill_dim = args.skill_dim
        self.hidden_dim = args.hidden_size
        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        # phi architecture
        self.linear1 = nn.Linear(num_inputs, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.skill_dim)

        self.apply(weights_init_)

        self.optimizer = torch.optim.Adam(self.parameters(), lr= self.lr )

    def forward(self, state):
        
        state = torch.from_numpy(state).float().to(self.device)
        state = state[:, :-self.skill_dim]

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

    def forward_np(self, state):
        
        state = torch.from_numpy(state).float().to(self.device)
        state = state[:-self.skill_dim]

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1.detach().cpu().numpy()


    def update_parameters(self, memory, batch_size, lambda_value, epsilon=1e-3):
        state_batch, _,  _, next_state_batch, _ = memory.sample(batch_size=batch_size)
        
        z_batch = state_batch[:, -self.skill_dim:]

        phi_s = self.forward(state_batch)
        phi_next_s = self.forward(next_state_batch)
        
        loss = -(phi_next_s - phi_s).mul(torch.from_numpy(z_batch).to(self.device).detach()).sum(1) - lambda_value.detach() * torch.min(torch.tensor(epsilon).detach(), 1 - (phi_s - phi_next_s).pow(2).sum(1))

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean().item()

class Lambda():
    def __init__(self, args):
        self.lr = args.lr
        self.skill_dim = args.skill_dim
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.lambda_value = torch.tensor(30.0, requires_grad=True, device=self.device)
        self.optimizer = torch.optim.Adam([self.lambda_value], lr=self.lr)

    def update_parameters(self, memory, batch_size, phi_net, epsilon = 1e-3):
        state_batch, _,  _, next_state_batch, _ = memory.sample(batch_size=batch_size)

        phi_s = phi_net(state_batch).detach() 
        phi_next_s = phi_net(next_state_batch).detach() 

        loss = self.lambda_value * torch.min(torch.tensor(epsilon).detach(), 1 - ( phi_s - phi_next_s).pow(2).sum(1))

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean().item()

