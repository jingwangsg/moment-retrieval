import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat, rearrange, reduce
from kn_util.basic import registry
from .segformerx import SegFormerXFPN, SegFormerX
from .ms_pooler import MultiScaleRoIAlign1D
from misc import inverse_sigmoid, cw2se, calc_iou_score_gt
from .loss import l1_loss, focal_loss
from kn_util.nn_utils.layers import MLP
from kn_util.nn_utils import clones
from kn_util.basic import registry
from torchvision.ops import sigmoid_focal_loss


class QueryBasedDecoder(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 ff_dim,
                 num_query,
                 num_layers=4,
                 num_scales=4,
                 pooler_resolution=16,
                 dim_init_ref=1,
                 dropout=0.1,
                 loss_cfg=None) -> None:
        super().__init__()
        self.query_embeddings = nn.Embedding(num_query, d_model)

        bbox_head = MLP(d_model, d_model, 2, 3)
        nn.init.constant_(bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_head.layers[-1].bias.data, 0)
        self.bbox_heads = clones(bbox_head, num_layers)
        self.reference_head = MLP(d_model, d_model, dim_init_ref, 3)
        score_head = MLP(d_model, d_model, 1)
        self.score_heads = clones(score_head, num_layers)
        self.num_layers = num_layers
        torch.nn.init.constant_(self.bbox_heads[0].layers[-1].bias.data[1:], -2.0)
        # make sure first offset is reasonable

        layer = nn.TransformerDecoderLayer(d_model, nhead, ff_dim, dropout=dropout, batch_first=True)
        self.layers = clones(layer, num_layers)

        # build pooler
        self.pooler = MultiScaleRoIAlign1D(output_size=pooler_resolution)

        # prepare for processing pooled_feat
        pool_ffn = MLP(d_model * pooler_resolution * num_scales,\
                        d_model, d_model)
        self.pool_ffns = clones(pool_ffn, num_layers)
        self.loss_cfg = loss_cfg

    def get_initital_reference(self, offset, reference_no_sigmoid):
        if reference_no_sigmoid.shape[-1] == 1:
            # assume temporal length of first prediction totally depends on offset predicted by boxx_head
            offset[..., :1] += reference_no_sigmoid
        else:
            offset += reference_no_sigmoid

        return offset

    def compute_loss(self, score, proposal, gt):
        score = score.squeeze(-1)
        B, Nq = score.shape

        ret_dict = dict()
        loss = 0.0

        expanded_proposal = rearrange(proposal, "b nq i -> (b nq) i")
        expanded_gt = repeat(gt, "b i -> (b nq) i", nq=Nq)
        iou_score = calc_iou_score_gt(expanded_proposal, expanded_gt, "giou")
        assign_score = rearrange(iou_score, "(b nq) -> b nq", nq=Nq)
        topk = self.loss_cfg["assign_topk"]
        indices = torch.topk(assign_score, k=topk, dim=1).indices
        # assign indices as positive, others as negative (background class)
        score_gt = assign_score
        positive_proposal = []
        for idx, inds in enumerate(indices):
            score_gt[idx, inds] = 1
            positive_proposal += [proposal[idx, inds]]
        positive_proposal = torch.stack(positive_proposal, dim=0)

        if "l1_loss" in self.loss_cfg:
            # only calculates l1 loss for positive
            # no need to compute l1 loss for negative!
            l1_loss_val = l1_loss(positive_proposal, gt)
            loss += l1_loss_val * self.loss_cfg["l1_loss"]
            ret_dict["l1_loss"] = l1_loss_val

        if "focal_loss" in self.loss_cfg:
            # focal_loss_val = focal_loss(score, proposal, gt)
            focal_loss_val = sigmoid_focal_loss(score, score_gt, reduction="mean")
            loss += focal_loss_val * self.loss_cfg["focal_loss"]
            ret_dict["focal_loss"] = focal_loss_val

        ret_dict["loss"] = loss
        ret_dict["topk_indices"] = indices
        return ret_dict

    def forward(self, feat_lvls, mask_lvls, gt=None, mode="tensor"):
        B = feat_lvls[0].shape[0]

        query_embeddings = self.query_embeddings.weight
        memory = feat_lvls[-1]
        mask = mask_lvls[-1]
        tgt = repeat(query_embeddings, "nq d -> b nq d", b=B)
        reference = None
        loss = 0.0
        ret_dict = dict()

        for idx, layer in enumerate(self.layers):
            output = layer(tgt, memory, memory_key_padding_mask=~mask)  # B, Nq, D

            offset = self.bbox_heads[idx](output)
            score_logits = self.score_heads[idx](output)
            score = score_logits.sigmoid()
            reference_no_sigmoid = self.reference_head(query_embeddings)

            if idx == 0:
                offset = self.get_initital_reference(offset, reference_no_sigmoid)
            else:
                offset = inverse_sigmoid(reference) + offset
            reference = offset.sigmoid()
            proposal = cw2se(reference)
            """<DEBUG> output_proposal"""
            if registry.get_object("output_proposal", False):
                registry.set_object(f"proposal_{idx}", proposal.detach().cpu().numpy())
            """<DEBUG>"""

            # calculate loss
            if mode == "train":
                cur_loss_dict = self.compute_loss(score, proposal, gt)
                weight = self.loss_cfg["aux_loss"] if idx == self.num_layers - 1 else 1.0
                loss += cur_loss_dict["loss"] * weight

                indices = cur_loss_dict.pop("topk_indices")
                # print(f"========indices {idx}==========")
                # print(indices)

                for loss_nm, loss_val in cur_loss_dict.items():
                    ret_dict[f"stage{idx}_{loss_nm}"] = loss_val

            pooled_feat_list = self.pooler(feat_lvls, proposal)
            # [[(Nq,Rp,D)...x B]...x N_lvls]
            pooled_feat_list = [torch.stack(x, dim=0) for x in pooled_feat_list]
            pooled_feat = torch.stack(pooled_feat_list, dim=0)
            # B, N_lvls, Nq, Rp, D
            pooled_feat = rearrange(pooled_feat, "nlvl b nq rd d -> b nq (nlvl rd d)")

            pooled_feat = self.pool_ffns[idx](pooled_feat)
            tgt = pooled_feat + query_embeddings

        if registry.get_object("output_proposal", False):
            import ipdb
            ipdb.set_trace()  #FIXME

        if mode == "train":
            ret_dict["loss"] = loss
            return ret_dict
        else:
            ret_dict["boxxes"] = proposal
            ret_dict["score"] = score.squeeze(-1)
            return ret_dict


class MultiScaleTemporalDetr(nn.Module):

    def __init__(self, backbone, head: QueryBasedDecoder) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask, gt=None, mode="tensor", **kwargs):
        vid_feat_lvls, vid_mask_lvls = self.backbone(vid_feat, vid_mask, txt_feat, txt_mask)
        ret_dict = self.head(vid_feat_lvls, vid_mask_lvls, gt=gt, mode=mode)
        return ret_dict
