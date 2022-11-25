import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip import CLIPTextModel
from kn_util.torch import freeze_module
from einops import repeat, rearrange


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class PromptCLIP(nn.Module):

    def __init__(self, num_query=50, num_prompt=1, pretrained="openai/clip-vit-large-patch14-336") -> None:
        super().__init__()
        clip_text = CLIPTextModel.from_pretrained(pretrained)
        self.num_query = num_query
        self.num_prompt = num_prompt

        self.config = clip_text.config
        self.embeddings = clip_text.text_model.embeddings
        self.encoder = clip_text.text_model.encoder
        self.final_layer_norm = clip_text.text_model.final_layer_norm

        freeze_module(self.encoder)
        freeze_module(self.embeddings)
        freeze_module(self.final_layer_norm)

        self.prompt = nn.Embedding(num_prompt * num_query, clip_text.config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)

        input_ids = repeat(input_ids, "b lt -> (b i) lt", i=self.num_query)
        attention_mask = repeat(attention_mask, "b lt -> (b i) lt", i=self.num_query)

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        seq_len = input_ids.shape[-1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324

        # HACK
        prompt = rearrange(self.prompt.weight, "(nq npt) d -> nq npt d", nq=self.num_query)
        hidden_states = torch.cat([prompt, hidden_states], dim=1)
        attention_mask = torch.cat([
            torch.ones(
                (self.num_query, self.num_prompt), dtype=attention_mask.dtype, device=input_ids.device), attention_mask
        ],
                                   dim=1)
        causal_attention_mask = self._build_causal_attention_mask(self.num_query, seq_len + self.num_prompt,
                                                                  hidden_states.dtype).to(hidden_states.device)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states,
                                       attention_mask=attention_mask,
                                       causal_attention_mask=causal_attention_mask,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states)

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=input_ids.device),
                                          input_ids.to(torch.int).argmax(dim=-1)]

        return pooled_output

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class CLIPPromptQuery(nn.Module):

    def __init__(self, num_query=30, num_prompt=1, pretrained="openai/clip-vit-large-patch14-336") -> None:
        super().__init__()
        self.prompt_clip = PromptCLIP(num_query, num_prompt, pretrained)

    def forward(self, text_inds, text_mask):
        B, Lt = text_inds.shape
        context_vectors_list = []
        for i in range(B):
            context_vectors = self.prompt_clip(input_ids=text_inds[i:i + 1], attention_mask=text_mask[i:i + 1])
            context_vectors_list += [context_vectors]
        context_vectors = torch.stack(context_vectors_list, dim=0)
        return context_vectors  # B, Nq, D

# class InitialQueryGenerator

if __name__ == "__main__":
    model = CLIPPromptQuery(num_prompt=50)
    model = model.cuda()
    B = 16
    Lt = 10
    text_inds = torch.arange(Lt).repeat(B, 1).cuda()
    text_mask = torch.ones((B, Lt), dtype=torch.long).cuda()
    embs = model(text_inds, text_mask)
    print(torch.cuda.max_memory_allocated()/(1024**3))
