import torch
import torch.nn as nn
from kn_util.general import get_logger, global_registry
import math
import sys
import os
import copy

logger = get_logger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.warn(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root)."""
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, seq_type, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.query_other = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_other = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_other = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.seq_type = seq_type

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        hidden_states_other,
        attention_mask,
        output_attention_probs=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        other_query_layer = self.query_other(hidden_states)
        other_key_layer = self.key_other(hidden_states_other)
        other_value_layer = self.value_other(hidden_states_other)

        other_query_layer = self.transpose_for_scores(other_query_layer)
        other_key_layer = self.transpose_for_scores(other_key_layer)
        other_value_layer = self.transpose_for_scores(other_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        other_attention_scores = torch.matmul(
            other_query_layer, other_key_layer.transpose(-1, -2)
        )
        other_attention_scores = other_attention_scores / math.sqrt(
            self.attention_head_size
        )

        if self.seq_type == "TXT":
            attention_scores = attention_scores + attention_mask
        elif self.seq_type == "VIS":
            other_attention_scores = other_attention_scores + attention_mask
        else:
            print("EROOR")
            exit()

        attention_scores = torch.cat([attention_scores, other_attention_scores], dim=-1)
        value_layer = torch.cat([value_layer, other_value_layer], dim=-2)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if output_attention_probs:
            return context_layer, attention_probs
        else:
            return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, seq_type, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(seq_type, config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        input_tensor,
        input_tensor_other,
        attention_mask,
        output_attention_probs=False,
    ):
        self_output = self.self(
            input_tensor,
            input_tensor_other,
            attention_mask,
            output_attention_probs=output_attention_probs,
        )
        if output_attention_probs:
            self_output, attention_probs = self_output
        attention_output = self.output(self_output, input_tensor)
        if output_attention_probs:
            return attention_output, attention_probs
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention_text = BertAttention("TXT", config)
        self.intermediate_text = BertIntermediate(config)
        self.output_text = BertOutput(config)

        self.attention_visual = BertAttention("VIS", config)
        self.intermediate_visual = BertIntermediate(config)
        self.output_visual = BertOutput(config)

    def forward(
        self,
        hidden_states,
        hidden_states_other,
        attention_mask,
        output_attention_probs=False,
    ):
        attention_output = self.attention_text(
            hidden_states,
            hidden_states_other,
            attention_mask,
            output_attention_probs=output_attention_probs,
        )
        attention_output_other = self.attention_visual(
            hidden_states_other,
            hidden_states,
            attention_mask,
            output_attention_probs=output_attention_probs,
        )
        if output_attention_probs:
            attention_output, attention_probs = attention_output
            attention_output_other, attention_probs_other = attention_output_other

        intermediate_output = self.intermediate_text(attention_output)
        layer_output = self.output_text(intermediate_output, attention_output)

        intermediate_output_other = self.intermediate_visual(attention_output_other)
        layer_output_other = self.output_visual(
            intermediate_output_other, attention_output_other
        )
        if output_attention_probs:
            return (
                layer_output,
                layer_output_other,
                attention_probs,
                attention_probs_other,
            )
        else:
            return layer_output, layer_output_other


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        hidden_states_other,
        attention_mask,
        output_all_encoded_layers=False,
        output_attention_probs=False,
    ):
        all_encoder_layers = []
        all_attention_probs = []
        for layer_module in self.layer:
            layer_out = layer_module(
                hidden_states,
                hidden_states_other,
                attention_mask,
                output_attention_probs=output_attention_probs,
            )
            if output_attention_probs:
                (
                    hidden_states,
                    hidden_states_other,
                    attention_probs,
                    attention_probs_other,
                ) = layer_out
                all_attention_probs.append([attention_probs, attention_probs_other])
            else:
                hidden_states, hidden_states_other = layer_out
            if output_all_encoded_layers:
                all_encoder_layers.append([hidden_states, hidden_states_other])
        if not output_all_encoded_layers:
            all_encoder_layers.append([hidden_states, hidden_states_other])
        if output_attention_probs:
            return all_encoder_layers, all_attention_probs
        else:
            return all_encoder_layers


# class BertPooler(nn.Module):
#     def __init__(self, config):
#         super(BertPooler, self).__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()

#     def forward(self, hidden_states):
#         # We "pool" the model by simply taking the hidden state corresponding
#         # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output


class BertPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config

    def init_bert_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# class BertModel(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertModel, self).__init__(config)
#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)
#         self.pooler = BertPooler(config)
#         self.apply(self.init_bert_weights)

#     def forward(
#         self,
#         input_ids,
#         token_type_ids=None,
#         attention_mask=None,
#         output_all_encoded_layers=True,
#     ):
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)
#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)

#         # We create a 3D attention mask from a 2D tensor mask.
#         # Sizes are [batch_size, 1, 1, to_seq_length]
#         # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
#         # this attention mask is more simple than the triangular masking of causal attention
#         # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

#         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#         # masked positions, this operation will create a tensor which is 0.0 for
#         # positions we want to attend and -10000.0 for masked positions.
#         # Since we are adding it to the raw scores before the softmax, this is
#         # effectively the same as removing these entirely.
#         extended_attention_mask = extended_attention_mask.to(
#             dtype=next(self.parameters()).dtype
#         )  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         embedding_output = self.embeddings(input_ids, token_type_ids)
#         encoded_layers = self.encoder(
#             embedding_output,
#             extended_attention_mask,
#             output_all_encoded_layers=output_all_encoded_layers,
#         )
#         sequence_output = encoded_layers[-1]
#         pooled_output = self.pooler(sequence_output)
#         if not output_all_encoded_layers:
#             encoded_layers = encoded_layers[-1]
#         return encoded_layers, pooled_output


class VLUniformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()