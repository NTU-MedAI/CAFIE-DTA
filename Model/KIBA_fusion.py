# -*- coding: utf-8 -*-

import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import DataHelper as DH
import emetrics as EM
from tqdm import  trange
from arg_information_KIBA import *
from GNN import GNN_feature
from information import get_infomatrion_KIBA
from information import get_infomatrion_DAVIS
from result_process import protein_infomation_KIBA
from MCAT import COMPOUND_MCAT_KIBA
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Transformer Parameters
d_model = 128  # Embedding Size
d_ff = 512  # FeedForward dimension
d_k = d_v = 32  # dimension of K(=Q), V
n_layers = 1  # number of Encoder
n_heads = 4  # number of heads in Multi-Head Attention



# MultiHeadCrossAttention Parameters
num_heads = 4
query_dim =128
key_dim = 256
value_dim =256



class MultiHeadCrossAttention_INTERATION_G(nn.Module):
    def __init__(self, num_heads, query_dim, key_dim, value_dim,dropout=0.1):
        super(MultiHeadCrossAttention_INTERATION_G, self).__init__()

        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim


        self.query_transform = nn.Linear(query_dim, key_dim * num_heads)
        self.key_transform = nn.Linear(key_dim, key_dim * num_heads)
        self.value_transform = nn.Linear(value_dim, value_dim * num_heads)

        self.dropout = nn.Dropout(dropout)
        self.output_transform = nn.Linear(value_dim * num_heads,8*value_dim)
        self.line1=nn.Linear(2048,128)
    def forward(self, query, key, value):
        assert query.size(-2) == 32 and query.size(-1) == 128, "Input query size must be 32x128"
        assert key.size(-2) == 32 and key.size(-1) == 256, "Input key size must be 32x128"
        assert value.size(-2) == 32 and value.size(-1) == 256, "Input value size must be 32x128"

        batch_size = query.size(0)

        transformed_query = self.query_transform(query).view(batch_size, self.num_heads, -1, self.key_dim)
        transformed_key = self.key_transform(key).view(batch_size, self.num_heads, -1, self.key_dim)
        transformed_value = self.value_transform(value).view(batch_size, self.num_heads, -1, self.value_dim)

        # Compute attention scores
        scores = torch.matmul(transformed_query, transformed_key.transpose(-2, -1))

        # Normalize scores
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum of values
        attended_values = torch.matmul(attention_weights, transformed_value)

        # Reshape and linear transformation
        attended_values = attended_values.view(batch_size, -1, self.value_dim * self.num_heads)
        output = self.output_transform(attended_values)
        output = output.mean(dim=1)
        output=self.line1(output)
        return output

query_dim_T=256
key_dim_T=128
value_dim_T=128
class MultiHeadCrossAttention_INTERATION_T(nn.Module):
    def __init__(self, num_heads, query_dim_T, key_dim_T, value_dim_T,dropout=0.1):
        super(MultiHeadCrossAttention_INTERATION_T, self).__init__()

        self.num_heads = num_heads
        self.query_dim = query_dim_T
        self.key_dim = key_dim_T
        self.value_dim = value_dim_T


        self.query_transform = nn.Linear(query_dim_T, key_dim * num_heads)
        self.key_transform = nn.Linear(key_dim_T, key_dim * num_heads)
        self.value_transform = nn.Linear(value_dim_T, value_dim * num_heads)

        self.dropout = nn.Dropout(dropout)
        self.output_transform = nn.Linear(value_dim_T * num_heads,8*value_dim)
        self.line1=nn.Linear(2048,128)
    def forward(self, query, key, value):
        assert query.size(-2) == 32 and query.size(-1) == 256, "Input query size must be 32x128"
        assert key.size(-2) == 32 and key.size(-1) == 128, "Input key size must be 32x128"
        assert value.size(-2) == 32 and value.size(-1) == 128, "Input value size must be 32x128"

        batch_size = query.size(0)

        transformed_query = self.query_transform(query).view(batch_size, self.num_heads, -1, self.key_dim)
        transformed_key = self.key_transform(key).view(batch_size, self.num_heads, -1, self.key_dim)
        transformed_value = self.value_transform(value).view(batch_size, self.num_heads, -1, self.value_dim)

        # Compute attention scores
        scores = torch.matmul(transformed_query, transformed_key.transpose(-2, -1))

        # Normalize scores
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum of values
        attended_values = torch.matmul(attention_weights, transformed_value)

        # Reshape and linear transformation
        attended_values = attended_values.view(batch_size, -1, self.value_dim * self.num_heads)
        output = self.output_transform(attended_values)
        output = output.mean(dim=1)
        output=self.line1(output)
        return output



class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads, query_dim, key_dim, value_dim, random, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.random = random

        self.query_transform = nn.Linear(query_dim, key_dim * num_heads)
        self.key_transform = nn.Linear(key_dim, key_dim * num_heads)
        self.value_transform = nn.Linear(value_dim, value_dim * num_heads)

        self.dropout = nn.Dropout(dropout)
        self.output_transform = nn.Linear(value_dim * num_heads, 8 * value_dim)
        self.line = nn.Linear(value_dim, 8 * value_dim)
        self.line1=nn.Linear(2048,128)

    def forward(self, query, key, value, random):
        # value=self.line(value)
        # print("value:",value.size())
        assert query.size(-2) == random and query.size(-1) == 128, "Input query size must be 32x128"
        assert key.size(-2) == random and key.size(-1) == 256, "Input key size must be 32x128"
        assert value.size(-2) == random and value.size(-1) == 256, "Input value size must be 32x128"

        batch_size = query.size(0)

        transformed_query = self.query_transform(query).view(batch_size, self.num_heads, -1, self.key_dim)
        transformed_key = self.key_transform(key).view(batch_size, self.num_heads, -1, self.key_dim)
        transformed_value = self.value_transform(value).view(batch_size, self.num_heads, -1, self.value_dim)

        # Compute attention scores
        scores = torch.matmul(transformed_query, transformed_key.transpose(-2, -1))

        # Normalize scores
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum of values
        attended_values = torch.matmul(attention_weights, transformed_value)

        # Reshape and linear transformation
        attended_values = attended_values.view(batch_size, -1, self.value_dim * self.num_heads)
        output = self.output_transform(attended_values)
        output = output.mean(dim=1)
        output=self.line1(output)
        return output, attention_weights


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb,label,smiles,proteins):
        self.texta = texta
        self.textb = textb
        self.label = label
        self.smiles =smiles
        self.proteins =proteins
    def __getitem__(self, item):
        return self.texta[item], self.textb[item], self.label[item],self.smiles[item],self.proteins[item]

    def __len__(self):
        return len(self.texta)


def BatchPad(batch_data, pad=0):
    texta, textb, label,smiles,proteins = list(zip(*batch_data))

    max_len_a = max([len(seq_a) for seq_a in texta])
    max_len_b = max([len(seq_b) for seq_b in textb])
    texta = [seq + [pad] * (max_len_a - len(seq)) for seq in texta]
    textb = [seq + [pad] * (max_len_b - len(seq)) for seq in textb]
    texta = torch.LongTensor(texta)
    textb = torch.LongTensor(textb)
    label = torch.FloatTensor(label)
    return (texta, textb, label,smiles,proteins)


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
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    # seq_q=seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]

        ##residual, batch_size = input_Q, input_Q.size(0)
        batch_size, seq_len, model_len = input_Q.size()
        residual_2D = input_Q.view(batch_size * seq_len, model_len)
        residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]

        # enc_outputs: [batch_size, src_len, d_model]
        # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder_Protein(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder_Protein, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.stream0 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.stream1 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.stream2 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len]

        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        stream0 = enc_outputs

        enc_self_attns0, enc_self_attns1, enc_self_attns2 = [], [], []
        for layer in self.stream0:
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            stream0, enc_self_attn0 = layer(stream0, enc_self_attn_mask)
            enc_self_attns0.append(enc_self_attn0)

        # skip connect -> stream0
        stream1 = stream0 + enc_outputs
        stream2 = stream0 + enc_outputs
        for layer in self.stream1:
            stream1, enc_self_attn1 = layer(stream1, enc_self_attn_mask)
            enc_self_attns1.append(enc_self_attn1)

        for layer in self.stream2:
            stream2, enc_self_attn2 = layer(stream2, enc_self_attn_mask)
            enc_self_attns2.append(enc_self_attn2)

        return torch.cat((stream1, stream2), 2), enc_self_attns0, enc_self_attns1, enc_self_attns2


class Encoder_Compound(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder_Compound, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.stream0 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.stream1 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.stream2 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len]

        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        stream0 = enc_outputs

        enc_self_attns0, enc_self_attns1, enc_self_attns2 = [], [], []
        for layer in self.stream0:
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            stream0, enc_self_attn0 = layer(stream0, enc_self_attn_mask)
            enc_self_attns0.append(enc_self_attn0)

        # skip connect -> stream0
        stream1 = stream0 + enc_outputs
        stream2 = stream0 + enc_outputs
        for layer in self.stream1:
            stream1, enc_self_attn1 = layer(stream1, enc_self_attn_mask)
            enc_self_attns1.append(enc_self_attn1)

        for layer in self.stream2:
            stream2, enc_self_attn2 = layer(stream2, enc_self_attn_mask)
            enc_self_attns2.append(enc_self_attn2)

        return torch.cat((stream1, stream2), 2), enc_self_attns0, enc_self_attns1, enc_self_attns2



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoderD = Encoder_Compound(DH.drugSeq_vocabSize)
        self.encoderT = Encoder_Protein(DH.targetSeq_vocabSize)
        self.liner = nn.Linear(25, 128)
        self.fc0 = nn.Sequential(
            nn.Linear(6 * d_model, args.embedding_size * d_model, bias=False),  # (512,2048)
            nn.LayerNorm(args.embedding_size * d_model),
            nn.Dropout(args.Dropout1),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(args.embedding_size * d_model,6*d_model ,bias=False),  # (2048,512)
            nn.LayerNorm(6 * d_model),
            nn.Dropout(args.Dropout2),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(6 * d_model, 1, bias=False)

    def forward(self, input_Drugs, input_Tars, smiles,protein):
        # input: [batch_size, src_len]
        # enc_outputs: [batch_size, src_len, d_model]
        # enc_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_Drugs, enc_attnsD0, enc_attnsD1, enc_attnsD2  = self.encoderD(input_Drugs)
        enc_Tars, enc_attnsT0, enc_attnsT1, enc_attnsT2= self.encoderT(input_Tars)

        enc_Drugs_2D0 = torch.sum(enc_Drugs, dim=1)
        enc_Drugs_2D1 = enc_Drugs_2D0.squeeze()
        enc_Tars_2D0 = torch.sum(enc_Tars, dim=1)
        enc_Tars_2D1 = enc_Tars_2D0.squeeze()


        protein_infomation_Davis_middle= self.liner(protein_infomation_KIBA(protein).cuda())
        model1 = MultiHeadCrossAttention(num_heads, query_dim, key_dim, value_dim, random).cuda()
        protein_attention_feature, attention1 = model1(protein_infomation_Davis_middle,enc_Tars_2D1 ,enc_Tars_2D1 ,len(protein))
        compound_attention_feature= COMPOUND_MCAT_KIBA(smiles, len(smiles),smiles).cuda()
        compound_attention_feature_tensor=compound_attention_feature.clone().detach()

        fc = torch.cat((enc_Drugs_2D1,enc_Tars_2D1,compound_attention_feature_tensor,protein_attention_feature), 1)  # (32,128*3)

        fc0 = self.fc0(fc)
        fc1 = self.fc1(fc0)
        fc2 = self.fc2(fc1)
        affi = fc2.squeeze()
        # print('affi',affi)

        return affi, enc_attnsD0, enc_attnsT0,enc_attnsD1, enc_attnsT1


if __name__ == '__main__':
    seed_torch()

    

    '''# davis dataset
    smile_maxlenDA, proSeq_maxlenDA = 100, 1000
    trainDV_num, testDV_num = 25046, 5010
    fpath_davis = 'D:/zl/transDTA-version4/data_davis/'
    '''

    smile_maxlenKB, proSeq_maxlenKB = args.smile_maxlenKB, args.proSeq_maxlenKB
    trainKB_num, testKB_num = args.trainKB_num, args.testKB_num
    valDV_num = args.valDV_num
    fpath_davis = args.dataset_path
    # davis -> logspance_trans=True. kiba -> logspance_trans=False
    # davis: affinity=pKd. kiba: affinity=kiba score
    drug, target, affinity = DH.LoadData(fpath_davis, logspance_trans=args.logspance_trans)
    drug_seqs, target_seqs, affiMatrix = DH.GetSamples(args.dataset, drug, target, affinity)
    labeled_drugs, labeled_targets, smiles, proteins = DH.LabelDT(drug_seqs, target_seqs,
                                                                  smile_maxlenKB, proSeq_maxlenKB)

    # shuttle
    labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle,smiles ,proteins\
        = DH.Shuttle(labeled_drugs, labeled_targets, affiMatrix,smiles,proteins)


    smiles1= smiles[0:19709]
    proteins1 = proteins[0:19709]
    Drugs_fold1 = labeledDrugs_shuttle[0:19709]
    Targets_fold1 = labeledTargets_shuttle[0:19709]
    affiMatrix_fold1 = affiMatrix_shuttle[0:19709]

    smiles2= smiles[19709:39418]
    proteins2 = proteins[19709:39418]
    Drugs_fold2 = labeledDrugs_shuttle[19709:39418]
    Targets_fold2 = labeledTargets_shuttle[19709:39418]
    affiMatrix_fold2 = affiMatrix_shuttle[19709:39418]

    smiles3= smiles[39418:59127]
    proteins3 = proteins[39418:59127]
    Drugs_fold3 = labeledDrugs_shuttle[39418:59127]
    Targets_fold3 = labeledTargets_shuttle[39418:59127]
    affiMatrix_fold3 = affiMatrix_shuttle[39418:59127]

    smiles4= smiles[59127:78836]
    proteins4 = proteins[59127:78836]
    Drugs_fold4 = labeledDrugs_shuttle[59127:78836]
    Targets_fold4 = labeledTargets_shuttle[59127:78836]
    affiMatrix_fold4 = affiMatrix_shuttle[59127:78836]

    smiles5= smiles[78836:98545]
    proteins5 = proteins[78836:98545]
    Drugs_fold5 = labeledDrugs_shuttle[78836:98545]
    Targets_fold5 = labeledTargets_shuttle[78836:98545]
    affiMatrix_fold5 = affiMatrix_shuttle[78836:98545]

    smiles6= smiles[98545:118254]
    proteins6 = proteins[98545:118254]
    Drugs_fold6 = labeledDrugs_shuttle[98545:118254]
    Targets_fold6 = labeledTargets_shuttle[98545:118254]
    affiMatrix_fold6 = affiMatrix_shuttle[98545:118254]


    # 98545
    train1_smiles=np.hstack((smiles1, smiles2, smiles3, smiles4, smiles5))
    train1_proteins=np.hstack((proteins1, proteins2, proteins3, proteins4, proteins5))
    train1_drugs = np.hstack((Drugs_fold1, Drugs_fold2, Drugs_fold3, Drugs_fold4, Drugs_fold5))
    train1_targets = np.hstack((Targets_fold1, Targets_fold2, Targets_fold3, Targets_fold4, Targets_fold5))
    train1_affinity = np.hstack(
        (affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5))
    # 98545
    train2_smiles = np.hstack((smiles2, smiles3, smiles4, smiles5,smiles1))
    train2_proteins = np.hstack((proteins2, proteins3, proteins4, proteins5, proteins1))
    train2_drugs = np.hstack((Drugs_fold2, Drugs_fold3, Drugs_fold4, Drugs_fold5, Drugs_fold1))
    train2_targets = np.hstack((Targets_fold2, Targets_fold3, Targets_fold4, Targets_fold5, Targets_fold1))
    train2_affinity = np.hstack(
        (affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5, affiMatrix_fold1))
    # 98545
    train3_smiles = np.hstack((smiles3, smiles4, smiles5, smiles1,smiles2))
    train3_proteins = np.hstack((proteins3, proteins4, proteins5, proteins1, proteins2))
    train3_drugs = np.hstack((Drugs_fold3, Drugs_fold4, Drugs_fold5, Drugs_fold1, Drugs_fold2))
    train3_targets = np.hstack((Targets_fold3, Targets_fold4, Targets_fold5, Targets_fold1, Targets_fold2))
    train3_affinity = np.hstack(
        (affiMatrix_fold3, affiMatrix_fold4, affiMatrix_fold5, affiMatrix_fold1, affiMatrix_fold2))
    # 98545
    train4_smiles = np.hstack(( smiles4, smiles5, smiles1, smiles2,smiles3))
    train4_proteins = np.hstack((proteins4, proteins5, proteins1, proteins2, proteins3))
    train4_drugs = np.hstack((Drugs_fold4, Drugs_fold5, Drugs_fold1,  Drugs_fold2, Drugs_fold3))
    train4_targets = np.hstack((Targets_fold4, Targets_fold5, Targets_fold1, Targets_fold2, Targets_fold3))
    train4_affinity = np.hstack(
        (affiMatrix_fold4, affiMatrix_fold5, affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3))
    # 98545
    train5_smiles = np.hstack((smiles5, smiles1, smiles2, smiles3,smiles4))
    train5_proteins = np.hstack((proteins5, proteins1, proteins2, proteins3, proteins4))
    train5_drugs = np.hstack((Drugs_fold5, Drugs_fold1, Drugs_fold2, Drugs_fold3, Drugs_fold4))
    train5_targets = np.hstack((Targets_fold5, Targets_fold1, Targets_fold2, Targets_fold3, Targets_fold4))
    train5_affinity = np.hstack(
        (affiMatrix_fold5, affiMatrix_fold1, affiMatrix_fold2, affiMatrix_fold3, affiMatrix_fold4))

    model_fromTrain1 = '/home/xt/Download/CPA1/fusion/youhua/KIBA_result/model_youhua1.pth'
    model_fromTrain2 = '/home/xt/Download/CPA1/fusion/youhua/KIBA_result/model_youhua2.pth'
    model_fromTrain3 = '/home/xt/Download/CPA1/fusion/youhua/KIBA_result/model_youhua3.pth'
    model_fromTrain4 = '/home/xt/Download/CPA1/fusion/youhua/KIBA_result/model_youhua4.pth'
    model_fromTrain5 = '/home/xt/Download/CPA1/fusion/youhua/KIBA_result/model_youhua5.pth'

    MSE, CI, RM2 = [], [], []
    for count in range(4):
        model = Transformer().cuda()
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=args.Learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode=args.scheduler_mode, factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=args.scheduler_verbose)
        EPOCHS, batch_size, accumulation_steps = args.EPOCH,args.batch_size,args.accumulation_steps  
        trainEP_loss_list = []
        valEP_loss_list = []
        min_train_loss = 1000  # save best model in train
        min_val_loss = 100000  # save best model in val

        if count == 0:
            train_iter = DatasetIterater(train1_drugs,  train1_targets,  train1_affinity,train1_smiles,train1_proteins)
        elif count == 1:
            train_iter = DatasetIterater(train2_drugs,  train2_targets,  train2_affinity,train2_smiles,train2_proteins)

        elif count == 2:
            train_iter = DatasetIterater(train3_drugs,  train3_targets,  train3_affinity,train3_smiles,train3_proteins)

        elif count == 3:
            train_iter = DatasetIterater(train4_drugs,  train4_targets,  train4_affinity,train4_smiles,train4_proteins)
        else:
            train_iter = DatasetIterater(train5_drugs,  train5_targets,  train5_affinity,train5_smiles,train5_proteins)

        val_iter=DatasetIterater(Drugs_fold6, Targets_fold6,  affiMatrix_fold6,smiles6,proteins6)
        test_iter = DatasetIterater(Drugs_fold6, Targets_fold6,  affiMatrix_fold6,smiles6,proteins6)
        train_loader = Data.DataLoader(train_iter, batch_size, False, collate_fn=BatchPad,pin_memory=True, drop_last=True)
        test_loader = Data.DataLoader(test_iter, batch_size, False, collate_fn=BatchPad, pin_memory=True,drop_last=True)
        val_loader= Data.DataLoader(val_iter, batch_size, False, collate_fn=BatchPad, pin_memory=True,drop_last=True)
        '''
        ###############
        ##Train Process
        ###############
        '''
        seed_torch(seed=2)
        train_obs, train_pred = [], []
        for epoch in trange(EPOCHS,colour="green",desc="EPOCH Times"):
            torch.cuda.synchronize()
            start = time.time()

            model.train()
            train_sum_loss = 0
            for train_batch_idx, (SeqDrug, SeqTar, real_affi, smiles,proteins) in enumerate(train_loader):
                SeqDrug, SeqTar, real_affi = SeqDrug.cuda(), SeqTar.cuda(), real_affi.cuda()
                pre_affi, enc_attnD0, enc_attnT0 , enc_attnsD1, enc_attnsT1\
                    = model(SeqDrug, SeqTar, smiles, proteins)

                train_loss = criterion(pre_affi, real_affi)
                train_sum_loss += train_loss.item()

                train_loss.backward()
                
                if ((train_batch_idx + 1) % accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # if ((train_batch_idx + 1) % 300) == 0:
                #     print('\n')
                #     print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(train_loss))

                if (train_batch_idx + 1) == (trainKB_num // batch_size + 1):
                    train_epoch_loss = train_sum_loss / train_batch_idx
                    scheduler.step(train_epoch_loss)
                    trainEP_loss_list.append(train_epoch_loss)
                    print('\n')
                    print('Epoch:', '%04d' % (epoch + 1), 'train_epoch_loss = ', '{:.6f}'.format(train_epoch_loss))
                    # save best train model
                    if train_epoch_loss < min_train_loss:
                        min_train_loss = train_epoch_loss
                        if count == 0:
                            torch.save(model.state_dict(), model_fromTrain1)
                            print('Best model in train1 from', '%04d' % (epoch + 1), 'Epoch', 'at',
                                    format(model_fromTrain1))
                        elif count == 1:
                             torch.save(model.state_dict(), model_fromTrain2)
                             print('Best model in train2 from', '%04d' % (epoch + 1), 'Epoch', 'at',
                                   format(model_fromTrain2))
                        elif count == 2:
                            torch.save(model.state_dict(), model_fromTrain3)
                            print('Best model in train3 from', '%04d' % (epoch + 1), 'Epoch', 'at',
                                  format(model_fromTrain3))
                        elif count == 3:
                            torch.save(model.state_dict(), model_fromTrain4)
                            print('Best model in train4 from', '%04d' % (epoch + 1), 'Epoch', 'at',
                                  format(model_fromTrain4))
                        else:
                            torch.save(model.state_dict(), model_fromTrain5)
                            print('Best model in train5 from', '%04d' % (epoch + 1), 'Epoch', 'at',
                                  format(model_fromTrain5))

            val_sum_loss = 0
            val_obs, val_pred = [], []
            with torch.no_grad():
                for val_batch_idx,(SeqDrug, SeqTar, real_affi, smiles,proteins) in enumerate(val_loader):
                    SeqDrug, SeqTar, real_affi = SeqDrug.cuda(), SeqTar.cuda(), real_affi.cuda()
                    pre_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1\
                        = model(SeqDrug, SeqTar, smiles, proteins)

                    val_obs.extend(real_affi.tolist())
                    val_pred.extend(pre_affi.tolist())

            print('val_MSE:', '{:.3f}'.format(EM.get_MSE(val_obs, val_pred)))
            print('val_CI:', '{:.3f}'.format(EM.get_cindex(val_obs, val_pred)))
            print('val_rm2:', '{:.3f}'.format(EM.get_rm2(val_obs, val_pred)))
            torch.cuda.synchronize()
            print('Time taken for 1 epoch is {:.4f} minutes'.format((time.time() - start) / 60))
            print('\n')

        #np.savetxt('trainLossMean_list.csv', trainEP_loss_list, delimiter=',')
        ##np.savetxt('valLossMean_list.csv', valEP_loss_list, delimiter=',')


            if epoch==49 or epoch==99 or epoch==149 or epoch==199or epoch==249or epoch==299or epoch==349or epoch==399:
                '''
            ###############
            ##Test Process
            ###############
                '''
                predModel = Transformer().cuda()
                if count == 0:
                    predModel.load_state_dict(torch.load(model_fromTrain1))
                elif count == 1:
                    predModel.load_state_dict(torch.load(model_fromTrain2))
                elif count == 2:
                    predModel.load_state_dict(torch.load(model_fromTrain3))
                elif count == 3:
                    predModel.load_state_dict(torch.load(model_fromTrain4))
                else:
                    predModel.load_state_dict(torch.load(model_fromTrain5))
                predModel.eval()  # -> model.train(), keep Batch Normalization and avoid Dropout

                train_obs, train_pred = [], []
                '''val_obs, val_pred = [], []'''
                test_obs, test_pred = [], []
                # DrugSeqs_buf, TarSeqs_buf = [], []

                with torch.no_grad():
                    for (DrugSeqs, TarSeqs, real_affi,smiles,proteins) in train_loader:
                        DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
                        pred_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1 \
                            = predModel(DrugSeqs, TarSeqs,smiles,proteins)

                        train_obs.extend(real_affi.tolist())
                        train_pred.extend(pred_affi.tolist())

                    '''for (DrugSeqs, TarSeqs, real_affi) in val_loader:
                        drug = DH.IntToToken(DrugSeqs, DH.drugSeq_vocab)
                        protein = DH.IntToToken(TarSeqs, DH.targetSeq_vocab)
                        DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
                        pred_affi, enc_attnD0, enc_attnT0, enc_attnsD1, enc_attnsT1, enc_attnsD2, enc_attnsT2\
                                                   = predModel(DrugSeqs, TarSeqs, drug,protein)  # pred_affi: [batch_affini]
                        val_obs.extend(real_affi.tolist())
                        val_pred.extend(pred_affi.tolist())'''

                    for (DrugSeqs, TarSeqs, real_affi,smiles,proteins) in test_loader:
                        DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
                        pred_affi, enc_attnD0, enc_attnT0 , enc_attnsD1, enc_attnsT1\
                            = predModel(DrugSeqs, TarSeqs,smiles,proteins)  # pred_affi: [batch_affini]

                        test_obs.extend(real_affi.tolist())
                        test_pred.extend(pred_affi.tolist())

                ##np.savetxt('test_obs.csv', test_obs, delimiter=',')
                # np.savetxt('DrugSeqs_buf.csv', DrugSeqs_buf, delimiter=',')

                print('train_MSE:', '{:.3f}'.format(EM.get_MSE(train_obs, train_pred)))
                print('train_CI:', '{:.3f}'.format(EM.get_cindex(train_obs, train_pred)))
                print('train_rm2:', '{:.3f}'.format(EM.get_rm2(train_obs, train_pred)))

                '''print('\n')
                print('val_MSE:', '{:.3f}'.format(EM.get_MSE(val_obs, val_pred)))
                print('val_CI:', '{:.3f}'.format(EM.get_cindex(val_obs, val_pred)))
                print('val_rm2:', '{:.3f}'.format(EM.get_rm2(val_obs, val_pred)))'''

                print('\n')

                print('test_MSE:', '{:.3f}'.format(EM.get_MSE(test_obs, test_pred)))
                print('test_CI:', '{:.3f}'.format(EM.get_cindex(test_obs, test_pred)))
                print('test_rm2:', '{:.3f}'.format(EM.get_rm2(test_obs, test_pred)))


        '''
        mse = EM.get_MSE(test_obs, test_pred)
        ci = EM.get_cindex(test_obs, test_pred)
        rm2 = EM.get_rm2(test_obs, test_pred)

        MSE.append(mse)
        CI.append(ci)
        RM2.append(rm2)

    np.savetxt('mse.csv', MSE, delimiter=',')
    np.savetxt('CI.csv', CI, delimiter=',')
    np.savetxt('RM2.csv', RM2, delimiter=',')
    '''