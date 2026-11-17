import os
import torch
from torch import nn

class CVAE(nn.Module):

    def __init__(self, input_dim, condition_dim, latent_dim, learning_rate=0.0001, weight_decay=1e-06, focus2_mode=False, mode=None, head_mode='single'):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mode = mode or ('focus2' if focus2_mode else 'baseline')
        self.focus2_mode = self.mode != 'baseline'
        self.head_mode = head_mode
        self.reconstruction_weight = float(os.environ.get('CVAE_RECONSTRUCTION_WEIGHT', '5.0'))
        self.target_external_weight = float(os.environ.get('CVAE_TARGET_EXTERNAL_WEIGHT', '1.0'))
        self.trend_external_weight = float(os.environ.get('CVAE_TREND_EXTERNAL_WEIGHT', '6.0'))
        self.target_vector_weight = float(os.environ.get('CVAE_TARGET_VECTOR_WEIGHT', '1.8'))
        self.target_residual_weight = float(os.environ.get('CVAE_TARGET_RESIDUAL_WEIGHT', '0.0'))
        self.input_residual_weight = float(os.environ.get('CVAE_INPUT_RESIDUAL_WEIGHT', '0.0'))
        self.mean_target_weight = float(os.environ.get('CVAE_MEAN_TARGET_WEIGHT', '24.0'))
        self.mean_l1_weight = float(os.environ.get('CVAE_MEAN_L1_WEIGHT', '12.0'))
        self.score_head_mse_weight = float(os.environ.get('CVAE_SCORE_HEAD_MSE_WEIGHT', '24.0'))
        self.score_head_l1_weight = float(os.environ.get('CVAE_SCORE_HEAD_L1_WEIGHT', '12.0'))
        self.level_classification_weight = float(os.environ.get('CVAE_LEVEL_CLASSIFICATION_WEIGHT', '1.0'))
        self.adjacent_trend_weight = float(os.environ.get('CVAE_ADJACENT_TREND_WEIGHT', '4.0'))
        self.adjacent_direction_weight = float(os.environ.get('CVAE_ADJACENT_DIRECTION_WEIGHT', '4.0'))
        self.ranking_weight = float(os.environ.get('CVAE_RANKING_WEIGHT', '3.0'))
        self.correlation_weight = float(os.environ.get('CVAE_CORRELATION_WEIGHT', '3.0'))
        self.kl_weight = float(os.environ.get('CVAE_KL_WEIGHT', '0.5'))
        self.encoder = nn.Sequential(nn.Linear(input_dim + condition_dim, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU())
        self.encoder_fc2_mean = nn.Linear(32, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(32, latent_dim)
        self.mean_decoder = nn.Sequential(nn.Linear(32 + condition_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU())
        self.mean_head = nn.Linear(32, 1)
        score_in = 32 + condition_dim if self.mode == 'dual2' else input_dim
        self.score_head = nn.Linear(score_in, 1)
        if self.head_mode == 'dual' or self.mode == 'dual2':
            self.score_head_d = nn.Linear(score_in, 1)
        self.residual_decoder = nn.Sequential(nn.Linear(latent_dim + condition_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU())
        self.residual_head = nn.Linear(32, input_dim)

    def encode(self, x, c):
        xc = torch.cat([x, c], dim=-1)
        encoded_features = self.encoder(xc)
        return (encoded_features, self.encoder_fc2_mean(encoded_features), self.encoder_fc2_logvar(encoded_features))

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, encoded_features, z, c):
        mean_features = self.mean_decoder(torch.cat([encoded_features, c], dim=-1))
        mean_pred = self.mean_head(mean_features)
        residual_features = self.residual_decoder(torch.cat([z, c], dim=-1))
        residual_raw = self.residual_head(residual_features)
        residual_zero_mean = residual_raw - residual_raw.mean(dim=1, keepdim=True)
        reconstructed = mean_pred.expand(-1, self.input_dim) + residual_zero_mean
        return reconstructed

    def forward(self, x, c):
        (encoded_features, mu, logvar) = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(encoded_features, z, c)
        if self.mode == 'dual2':
            self._last_score_input = torch.cat([encoded_features, c], dim=-1)
        return (recon, mu, logvar)

class ScoringModel(nn.Module):
    def __init__(self, group, arm, seed, cfg):
        super().__init__()
        assert group == 'full' and arm == 'reconstruction'
        assert cfg['readout'] == 'original' and cfg['output'] == 'sigmoid'
        self.group = group
        self.cfg = cfg
        self.network = CVAE(8, 16, 16)

    @property
    def has_cvae(self):
        return True

    def forward(self, x, c):
        self.rec_target = x
        recon, mu, lv = self.network(x, c)
        raw = self.network.score_head(recon).flatten()
        score = 100 * torch.sigmoid(raw / self.cfg['temperature'])
        return score, recon, mu, lv
