import torch
import torch.nn as nn

KF_STATE_DIM = 8
KF_OBS_DIM = 4
KF_HIDDEN_DIM = 80

FEATURE_SPECS = {
    "f2_conf": {
        "input_dim": 5,
        "f1_dim": 0,
        "main_input_dim": 5,
        "use_f1_branch": False,
    },
    "f2_f4_conf": {
        "input_dim": 13,
        "f1_dim": 0,
        "main_input_dim": 13,
        "use_f1_branch": False,
    },
    "f1_f2_f4_conf": {
        "input_dim": 17,
        "f1_dim": 4,
        "main_input_dim": 13,
        "use_f1_branch": True,
    },
}


def get_feature_spec(feature_mode):
    if feature_mode not in FEATURE_SPECS:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    return FEATURE_SPECS[feature_mode]


def infer_feature_mode_from_state_dict(state_dict):
    if any(key.startswith("f1_encoder.") for key in state_dict.keys()):
        return "f1_f2_f4_conf"

    fc_in1_weight = state_dict.get("fc_in1.weight")
    if fc_in1_weight is None:
        return "f1_f2_f4_conf"

    in_features = fc_in1_weight.shape[1]
    if in_features == 5:
        return "f2_conf"
    if in_features == 13:
        return "f2_f4_conf"
    return "f1_f2_f4_conf"


class KalmanNetNN(nn.Module):
    def __init__(
        self,
        feature_mode="f1_f2_f4_conf",
        state_dim=KF_STATE_DIM,
        obs_dim=KF_OBS_DIM,
        hidden_dim=KF_HIDDEN_DIM,
    ):
        super(KalmanNetNN, self).__init__()
        spec = get_feature_spec(feature_mode)

        self.feature_mode = feature_mode
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.f1_dim = spec["f1_dim"]
        self.main_input_dim = spec["main_input_dim"]
        self.use_f1_branch = spec["use_f1_branch"]
        self.input_dim = spec["input_dim"]

        self.h1_dim = state_dim * state_dim
        self.h2_dim = state_dim * state_dim
        self.h3_dim = obs_dim * obs_dim

        if self.use_f1_branch:
            self.f1_encoder = nn.Sequential(
                nn.Linear(self.f1_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.main_encoder = nn.Sequential(
                nn.Linear(self.main_input_dim, hidden_dim * 2),
                nn.Tanh(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.Tanh(),
            )
            fused_dim = hidden_dim * 3
        else:
            self.f1_encoder = None
            self.main_encoder = nn.Sequential(
                nn.Linear(self.main_input_dim, hidden_dim * 2),
                nn.Tanh(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.Tanh(),
            )
            fused_dim = hidden_dim * 2

        self.fc_in1 = nn.Linear(fused_dim, self.h1_dim)
        self.fc_in2 = nn.Linear(fused_dim, self.h2_dim)
        self.fc_in3 = nn.Linear(fused_dim, self.h3_dim)

        self.gru_q = nn.GRU(self.h1_dim, self.h1_dim, batch_first=True)
        self.gru_sigma = nn.GRU(
            self.h2_dim + self.h1_dim, self.h2_dim, batch_first=True
        )
        self.fc_sigma_to_s = nn.Linear(self.h2_dim, self.h3_dim)
        self.gru_s = nn.GRU(self.h3_dim + self.h3_dim, self.h3_dim, batch_first=True)

        self.fc_out = nn.Sequential(
            nn.Linear(self.h2_dim + self.h3_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, state_dim * obs_dim),
        )

        nn.init.zeros_(self.fc_out[-1].weight)
        nn.init.zeros_(self.fc_out[-1].bias)

    def encode_inputs(self, inputs):
        if self.use_f1_branch:
            f1_inputs = inputs[..., : self.f1_dim]
            main_inputs = inputs[..., self.f1_dim :]
            f1_embed = self.f1_encoder(f1_inputs)
            main_embed = self.main_encoder(main_inputs)
            return torch.cat([f1_embed, main_embed], dim=-1)
        return self.main_encoder(inputs)

    def forward(self, inputs, hidden_states=None):
        batch_size = inputs.size(0)

        if hidden_states is None:
            device = inputs.device
            h_q_0 = torch.zeros(1, batch_size, self.h1_dim, device=device)
            h_sigma_0 = torch.zeros(1, batch_size, self.h2_dim, device=device)
            h_s_0 = torch.zeros(1, batch_size, self.h3_dim, device=device)
        else:
            h_q_0, h_sigma_0, h_s_0 = hidden_states

        fused_inputs = self.encode_inputs(inputs)

        x1 = torch.tanh(self.fc_in1(fused_inputs))
        out_q, h_q_n = self.gru_q(x1, h_q_0)

        x2 = torch.tanh(self.fc_in2(fused_inputs))
        gru_sigma_input = torch.cat([x2, out_q], dim=-1)
        out_sigma, h_sigma_n = self.gru_sigma(gru_sigma_input, h_sigma_0)

        x3 = torch.tanh(self.fc_in3(fused_inputs))
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
