import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from utils.layers import *
from utils.data_utils import *
from utils.other_utils import *
    
class BERT_basic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.pretrain_model = AutoModel.from_pretrained(args.pretrain_model, return_dict=True)
        if args.load_pretrained_model_path is not None:
            model_state_dict = torch.load(args.load_pretrained_model_path)
            self.pretrain_model.load_state_dict(model_state_dict)
        self.sent_dim = self.pretrain_model.config.hidden_size
        self.fc = MLP(self.sent_dim, self.sent_dim, 1, 0, 0.0, layer_norm=True)
        self.loss_func =  nn.CrossEntropyLoss(reduction='mean')
        self.epoch_idx = 0

        if self.args.is_R3F and self.args.R3F_noise_type == "normal":
            self.noise_sampler = torch.distributions.normal.Normal(
                loc=0.0, scale=1e-6
            )
        elif self.args.is_R3F and self.args.R3F_noise_type == "uniform":
            self.noise_sampler = torch.distributions.uniform.Uniform(
                low=-1e-6, high=1e-6
            )
   

    def forward(self, lm_input, return_sent_vect=False):

        outouts = self.pretrain_model(**lm_input)
        sent_vect = outouts.pooler_output # (bs*nc, sent_vect_dims)
        logits = self.fc(sent_vect)

        if not return_sent_vect:
            return logits
        else:
            return logits, sent_vect
        
    def predict(self, input_data):
        bs = len(input_data['example_id'])
        nc = len(input_data['LM_input']['input_ids'])//bs

        logits = self.forward(input_data['LM_input']).reshape(bs,nc)
        return logits

    def compute_loss(self, input_data, labels):

        bs = len(input_data['example_id'])
        nc = len(input_data['LM_input']['input_ids'])//bs

        logits = self.forward(input_data['LM_input']).reshape(bs,nc)
        loss = self.loss_func(logits, labels)

        return loss, logits

    def compute_CET_loss(self, input_data, labels):

        bs = len(input_data['example_id'])
        nc = input_data['LM_input']['input_ids'].shape[0]//bs
        topk = self.args.CET_topk

        logits = self.forward(input_data['LM_input']).reshape(bs,nc)
        assert logits.shape == (len(labels), nc)
        prob_score = torch.softmax(logits, dim=-1)

        # batch_ref_cnt equals to ref_cnt_1+ref_cnt_2+... in one batch
        batch_ref_cnt = np.sum(input_data['ref_cnt']).item()

        if batch_ref_cnt == 0:
            joint_prob_score = prob_score
        else:
            # (nc*batch_ref_cnt, seq_len)
            assert input_data['ref_LM_input']['input_ids'].shape[0] == nc*batch_ref_cnt
            # (nc*batch_ref_cnt, )
            # ref_logits = self.forward(input_data['ref_LM_input'])
            num_chunk = (batch_ref_cnt-1)//self.args.batch_size + 1 
            ref_logits_lst = []
            for chunk_input_ids, chunk_attention_mask in zip(input_data['ref_LM_input']['input_ids'].chunk(num_chunk, 0), \
                                                            input_data['ref_LM_input']['attention_mask'].chunk(num_chunk, 0)):
                chunk_data = {
                    'input_ids': chunk_input_ids,
                    'attention_mask': chunk_attention_mask,
                }
                ref_logits_lst.append(self.forward(chunk_data))

            ref_logits = torch.cat(ref_logits_lst, dim=0)

            # (bs,nc)
            ref_prob_score = torch.zeros_like(prob_score).to(prob_score.device) 
            ref_accum = 0
            for tmp_i in range(bs):
                ref_cnt = input_data['ref_cnt'][tmp_i]
                ref_accum += ref_cnt
                if ref_cnt == 0:
                    continue
                # (nc, ref_cnt)
                ref_logits_onesample = ref_logits[nc*(ref_accum-ref_cnt):nc*ref_accum].reshape(nc, ref_cnt)
                # (nc, ref_cnt)
                ref_prob_score_onesample = torch.softmax(ref_logits_onesample, dim=0) # softmax in each ref samples
                # (nc,)
                ref_prob_score[tmp_i] = torch.mean(ref_prob_score_onesample, dim=1)

            # (bs,1)
            ref_weight = torch.tensor(input_data['ref_cnt']).float().to(prob_score.device).reshape(-1,1)
            ref_weight[ref_weight>0] = 1-self.args.CET_weight
            # (bs,nc)
            joint_prob_score = (1-ref_weight)*prob_score + ref_weight*ref_prob_score
            

        loss = F.nll_loss(torch.log(joint_prob_score+1e-10),labels)

        return loss, joint_prob_score

    def compute_CET_joint_loss(self, input_data, labels):

        bs = len(input_data['example_id'])
        nc = input_data['LM_input']['input_ids'].shape[0]//bs
        topk = self.args.CET_topk

        # (bs, nc)
        logits = self.forward(input_data['LM_input']).reshape(bs, nc)
        assert logits.shape == (len(labels), nc)
        # (bs, )
        loss_anchor = nn.CrossEntropyLoss(reduction='none')(logits, labels)

        # batch_ref_cnt equals to ref_cnt_1+ref_cnt_2+... in one batch
        batch_ref_cnt = np.sum(input_data['ref_cnt']).item()

        if batch_ref_cnt == 0:
            loss_joint = loss_anchor
        else:
            # (nc*batch_ref_cnt, seq_len)
            assert input_data['ref_LM_input']['input_ids'].shape[0] == nc*batch_ref_cnt
            # (nc*batch_ref_cnt, )
            # ref_logits = self.forward(input_data['ref_LM_input'])
            num_chunk = (batch_ref_cnt-1)//self.args.batch_size + 1 
            ref_logits_lst = []
            for chunk_input_ids, chunk_attention_mask in zip(input_data['ref_LM_input']['input_ids'].chunk(num_chunk, 0), \
                                                            input_data['ref_LM_input']['attention_mask'].chunk(num_chunk, 0)):
                chunk_data = {
                    'input_ids': chunk_input_ids,
                    'attention_mask': chunk_attention_mask,
                }
                ref_logits_lst.append(self.forward(chunk_data))  
            ref_logits = torch.cat(ref_logits_lst, dim=0)

            # (bs,)
            loss_joint = torch.zeros(bs,).to(logits.device) 
            ref_accum = 0
            for tmp_i in range(bs):
                ref_cnt = input_data['ref_cnt'][tmp_i]
                ref_accum += ref_cnt
                if ref_cnt == 0:
                    loss_joint[tmp_i] = loss_anchor[tmp_i]
                    continue
                # (ref_cnt, nc)
                ref_logits_onesample = ref_logits[nc*(ref_accum-ref_cnt):nc*ref_accum].reshape(nc, ref_cnt).T
                # (ref_cnt, )
                sum_ref_loss = nn.CrossEntropyLoss(reduction='sum')(ref_logits_onesample, torch.tensor([labels[tmp_i]]*ref_cnt, device=labels.device))
                
                loss_joint[tmp_i] = self.args.CET_weight * loss_anchor[tmp_i] + \
                                        (1-self.args.CET_weight) * (sum_ref_loss/ref_cnt)

        loss = loss_joint.mean()

        return loss, logits

    def compute_BSS_loss(self, input_data, labels):

        bs = len(input_data['example_id'])
        nc = len(input_data['LM_input']['input_ids'])//bs

        logits, sent_vect = self.forward(input_data['LM_input'], return_sent_vect=True)
        logits = logits.reshape(bs,nc)
        loss = self.loss_func(logits, labels)

        # sent_vect: (bs*nc, sent_vect_dim)
        u,s,v = torch.svd(sent_vect)
        BSS_loss = s[-1].pow(2)

        total_loss = loss + self.args.BSS_weight*BSS_loss

        return total_loss, logits

    def compute_R3F_loss(self, input_data, labels):
        """
            Compute the R3F loss for the given sample.
        """
        bs = len(input_data['example_id'])
        nc = len(input_data['LM_input']['input_ids']) // bs

        input_ids = input_data['LM_input']['input_ids']
        attention_mask = input_data['LM_input']['attention_mask']
        token_embeddings = self.pretrain_model.embeddings(input_data['LM_input']['input_ids'])
        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sent_vect = outputs.pooler_output  # (bs*nc, sent_vect_dims)
        input_logits = self.fc(sent_vect).reshape(bs, nc)

        noise = self.noise_sampler.sample(sample_shape=token_embeddings.shape).to(
            token_embeddings
        )
        noised_embeddings = token_embeddings.detach().clone() + noise
        noised_outputs = self.pretrain_model(
            inputs_embeds=noised_embeddings,
            attention_mask=attention_mask,
        )
        noised_sent_vect = noised_outputs.pooler_output
        noised_logits = self.fc(noised_sent_vect).reshape(bs, nc)

        symm_kl = get_symm_kl(noised_logits, input_logits)
        
        ce_loss = self.loss_func(input_logits, labels)
        loss = ce_loss + self.args.R3F_lambda * symm_kl

        return loss, input_logits