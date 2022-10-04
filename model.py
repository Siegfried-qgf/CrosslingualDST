from transformers import MT5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack,T5Block,T5Attention,T5LayerNorm,T5LayerFF,T5LayerCrossAttention
from transformers.models.t5.modeling_t5 import T5Config
import copy
import torch.nn as nn
import torch

class myMT5(MT5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = myMT5Stack(encoder_config, self.shared)

class myMT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        list=[]
        for i in range(int(config.num_layers / 2)):
            list.append(T5Block(config, has_relative_attention_bias=bool(i == 0)))
        list.append(myMT5Block(config, has_relative_attention_bias=bool(i == 0)))
        for i in range(int((config.num_layers/2+1)),int(config.num_layers)):
            list.append(T5Block(config, has_relative_attention_bias=bool(i == 0)))
        self.block = nn.ModuleList(
           list
        )

class myMT5Block(T5Block):
    def __init__(self, config,has_relative_attention_bias=False):
        super().__init__(config,has_relative_attention_bias)
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(myT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))


class myT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs