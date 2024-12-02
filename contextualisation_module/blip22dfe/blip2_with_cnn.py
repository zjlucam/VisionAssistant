import torch
from torch import nn

class Blip2WithCNNFeatures(nn.Module):
    def __init__(self, base_blip_model, cnn_feature_dim):
        super().__init__()
        self.base_model = base_blip_model
        self.hidden_size = base_blip_model.config.text_config.hidden_size
        self.cnn_projector = nn.Linear(cnn_feature_dim, self.hidden_size)
        self.vision_projector = nn.Linear(base_blip_model.vision_model.config.hidden_size, self.hidden_size)
        self.fusion_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, pixel_values, input_ids, attention_mask, cnn_features, labels=None):
        vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
        vision_hidden_states = vision_outputs.last_hidden_state
        projected_vision_features = self.vision_projector(vision_hidden_states)
        projected_cnn_features = self.cnn_projector(cnn_features).unsqueeze(1)
        combined_features = torch.cat([projected_vision_features, projected_cnn_features.repeat(1, vision_hidden_states.size(1), 1)], dim=2)
        fused_features = self.fusion_layer(combined_features)
        text_embeddings = self.base_model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([fused_features, text_embeddings], dim=1)
        extended_attention_mask = torch.cat([torch.ones(fused_features.size()[:-1], dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=1)
        outputs = self.base_model.language_model(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask, labels=labels)
        return outputs
