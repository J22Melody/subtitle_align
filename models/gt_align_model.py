import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_text_model import BertTextModel
import numpy as np
from utils import calc_1d_iou, calc_align_metrics

import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class GtInvAlignTransformer(nn.Module):
    """
    Alignment model taking as input a video feature sequence and a text sentence

    # some code borrowed from DETR https://github.com/facebookresearch/detr/blob/master/models/detr.py
    """

    def __init__(self, opts, dataloader):
        super().__init__()

        self.opts = opts
        self.dataloader = dataloader
            
        # TODO: Add as params 
        self.d_vid = opts.feature_dim # visual backbone (I3D, Swin, pose, etc.) feature dim 
        self.d_txt = 768 # BERT feature dim

        # text embedding model 
        if not self.opts.finetune_bert:
            self.text_model = BertTextModel(opts.bert_model).requires_grad_(False)
        else:
            self.text_model = BertTextModel(opts.bert_model)

        # input projections for 2 modalities 
        self.input_proj_vid = nn.Conv1d(self.d_vid, opts.d_model, kernel_size=1)
        self.input_proj_txt = nn.Conv1d(self.d_txt, opts.d_model, kernel_size=1)

        # adapter from different visual feature dim to pretrained dim
        self.input_proj_vid_adapt = None
        if opts.feature_dim_adapt:
            self.input_proj_vid_adapt = nn.Conv1d(opts.feature_dim_adapt, self.d_vid, kernel_size=1)

        # embedding for original subtitle 
        ref_vec_dim = 0
        if self.opts.concatenate_prior:
            ref_vec_dim = ref_vec_dim + 1
        if self.opts.concatenate_segmentation:
            ref_vec_dim = ref_vec_dim + 6

        if ref_vec_dim > 0:
            self.ref_vec_embedding = nn.Linear(ref_vec_dim, opts.d_model)
            self.reproject_concatenate = nn.Linear(opts.d_model * 2, opts.d_model)

        # self positional enc
        self.positional_enc = PositionalEncoding(d_model=opts.d_model)

        # transformer
        self.transformer = nn.Transformer(
                                          num_encoder_layers=opts.n_enc_layers,
                                          num_decoder_layers=opts.n_dec_layers,
                                          nhead=opts.n_heads,
                                          d_model=opts.d_model,
                                          dropout=opts.transformer_dropout,
                                          )

        # FC 
        if self.opts.pos_weight > 0:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.opts.pos_weight))
        else:
            self.loss = nn.BCEWithLogitsLoss()
        n_classes = 1

        self.fc = MLP(input_dim=opts.d_model, hidden_dim=256, output_dim=n_classes, num_layers=3)

    def forward(self, data_dict):

        out = {}
        
        vid_inp = data_dict['feats'].type(torch.FloatTensor).cuda() # ( B, T, C )
        vid_inp = vid_inp.permute([0,2,1]) # (B, C, T)    
        
        text_inp = data_dict['txt'] # ( B, )
        
        target_vector = data_dict['gt_vec'].type(torch.FloatTensor).cuda()

        txt_emb = self.text_model(text_inp) # (B, 768, T_txt, 1) 
        
        txt_emb = txt_emb.type(torch.FloatTensor).cuda()

        assert self.opts.n_dec_layers >= 2, 'Should have more than one decoder layer with mulitple decoder inputs'
        txt_emb = txt_emb.squeeze(-1) # (B, 768, T_txt)

        # -- project text  
        txt_emb = self.input_proj_txt(txt_emb) * math.sqrt(self.opts.d_model) # (B, C, 1)
        txt_emb = txt_emb.permute([2,0,1]) # (1, B, C)

        # -- positional encoding
        if self.opts.positional_encoding_text:
            txt_emb = self.positional_enc(txt_emb)

        # -- project vid  
        if self.input_proj_vid_adapt:
            vid_inp = self.input_proj_vid_adapt(vid_inp)
        vid_emb = self.input_proj_vid(vid_inp) # (B, C, T)
        vid_emb = vid_emb.permute([2,0,1]) # (T, B, C)       

        pr_vec = None
        seg_vec = None
        ref_inp_lst = []

        if self.opts.concatenate_prior:
            pr_vec = data_dict['pr_vec'].type(torch.FloatTensor).cuda()
            ref_inp_lst.append(pr_vec)

        if self.opts.concatenate_segmentation:
            seg_vec = torch.cat((data_dict['seg_sign_feats'], data_dict['seg_sent_feats']), dim=2)
            seg_vec = seg_vec.type(torch.FloatTensor).cuda()
            seg_vec = torch.exp(seg_vec)
            seg_vec = seg_vec.view(seg_vec.shape[0], pr_vec.shape[1], -1, seg_vec.shape[2])
            seg_vec = seg_vec.mean(dim=2)
            ref_inp_lst.append(seg_vec)

        if len(ref_inp_lst) > 0:
            if len(ref_inp_lst) > 1:
                ref_inp = torch.cat(ref_inp_lst, dim=2)
            else:
                ref_inp = ref_inp_lst[0]
            ref_inp = self.ref_vec_embedding(ref_inp)
            ref_inp = ref_inp.permute([1,0,2])
            vid_emb = torch.cat((vid_emb,ref_inp),2)
            vid_emb = self.reproject_concatenate(vid_emb)

        vid_emb = vid_emb * math.sqrt(self.opts.d_model)
        vid_emb = self.positional_enc(vid_emb)

        # inverted transformer
        trf_output = self.transformer(src=txt_emb, tgt=vid_emb) # (B, C)

        hs = trf_output

        hs = hs.permute([1,0,2])

        lin_layer = self.fc(hs).squeeze(-1)

        target_vector = target_vector.squeeze(-1).cuda()
        loss = self.loss(lin_layer, target_vector)

        out['loss'] = loss

        outputs = torch.sigmoid(lin_layer.detach())

        out['preds'] = outputs.cpu().numpy()
        if self.opts.concatenate_prior: 
            out['pr_vec'] = pr_vec.cpu().numpy()
        else: 
            out['pr_vec'] = target_vector.cpu().numpy()
        out['gt_vec'] = target_vector.cpu().numpy()

        # -- calc f1 metrics 
        if target_vector.sum().item() > 0:
            correct, tp, fp, fn, total = calc_align_metrics(outputs.round().detach().cpu().numpy(),
                                                        target_vector.detach().cpu().numpy(),
                                                        inp_fmt='vector',
                                                        BACKGROUND_LABEL = 0,
                                                        overlaps = (0.5,)
                                                        )
            out.update( {'correct': correct, 'tp': tp, 'fp': fp, 'fn': fn, 'total_frames': total} )

        if self.opts.concatenate_prior: 
            if target_vector.sum().item() > 0:
                base_outputs = data_dict['pr_vec'].detach().int().squeeze(-1).cpu().numpy()
                correct_b, tp_b, fp_b, fn_b, total_b = calc_align_metrics(pr_vec.detach().cpu().numpy(),
                                                                target_vector.detach().cpu().numpy(),
                                                                inp_fmt='vector',
                                                                BACKGROUND_LABEL = 0,
                                                                overlaps = (0.5,)
                                                                )

                out.update( {'correct_b': correct_b, 'tp_b': tp_b, 'fp_b': fp_b, 'fn_b': fn_b, 'total_frames_b': total_b} )

        return out

