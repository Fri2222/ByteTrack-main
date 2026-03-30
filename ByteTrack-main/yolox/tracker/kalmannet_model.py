import torch
import torch.nn as nn

KF_INPUT_DIM = 13
KF_STATE_DIM = 8
KF_OBS_DIM = 4
KF_HIDDEN_DIM = 80


class KalmanNetNN(nn.Module):
    def __init__(
        self,
        input_dim=KF_INPUT_DIM,
        state_dim=KF_STATE_DIM,
        obs_dim=KF_OBS_DIM,
        hidden_dim=KF_HIDDEN_DIM,
    ):
        super(KalmanNetNN, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.h1_dim = state_dim * state_dim
        self.h2_dim = state_dim * state_dim
        self.h3_dim = obs_dim * obs_dim

        self.fc_in1 = nn.Linear(input_dim, self.h1_dim)
        self.fc_in2 = nn.Linear(input_dim, self.h2_dim)
        self.fc_in3 = nn.Linear(input_dim, self.h3_dim)

        self.gru_q = nn.GRU(self.h1_dim, self.h1_dim, batch_first=True)
        self.gru_sigma = nn.GRU(self.h2_dim + self.h1_dim, self.h2_dim, batch_first=True)

        self.fc_sigma_to_s = nn.Linear(self.h2_dim, self.h3_dim)
        self.gru_s = nn.GRU(self.h3_dim + self.h3_dim, self.h3_dim, batch_first=True)

        self.fc_out = nn.Sequential(
            nn.Linear(self.h2_dim + self.h3_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, state_dim * obs_dim),
        )

        nn.init.zeros_(self.fc_out[-1].weight)
        nn.init.zeros_(self.fc_out[-1].bias)

    def forward(self, inputs, hidden_states=None):
        batch_size = inputs.size(0)

        if hidden_states is None:
            device = inputs.device
            h_q_0 = torch.zeros(1, batch_size, self.h1_dim, device=device)
            h_sigma_0 = torch.zeros(1, batch_size, self.h2_dim, device=device)
            h_s_0 = torch.zeros(1, batch_size, self.h3_dim, device=device)
        else:
            h_q_0, h_sigma_0, h_s_0 = hidden_states

        x1 = torch.tanh(self.fc_in1(inputs))
        out_q, h_q_n = self.gru_q(x1, h_q_0)

        x2 = torch.tanh(self.fc_in2(inputs))
        gru_sigma_input = torch.cat([x2, out_q], dim=-1)
        out_sigma, h_sigma_n = self.gru_sigma(gru_sigma_input, h_sigma_0)

        x3 = torch.tanh(self.fc_in3(inputs))
        sigma_mapped = self.fc_sigma_to_s(out_sigma)
        gru_s_input = torch.cat([x3, sigma_mapped], dim=-1)
        out_s, h_s_n = self.gru_s(gru_s_input, h_s_0)

        last_out_sigma = out_sigma[:, -1, :]
        last_out_s = out_s[:, -1, :]

        k_input = torch.cat([last_out_sigma, last_out_s], dim=-1)
        k_flat = self.fc_out(k_input)

        k_gain = k_flat.view(-1, self.state_dim, self.obs_dim)
        new_hidden_states = (h_q_n, h_sigma_n, h_s_n)
        return k_gain, new_hidden_states
