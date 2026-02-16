# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import datetime
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForTokenClassification, BertTokenizer)
import time
from urllib.parse import urlparse
import requests
import zipfile
from pathlib import Path
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.distributions import Categorical
from torch_struct import TreeCRF
from copy import deepcopy
from torch.distributions.utils import lazy_property
import json
import glob
from lxml import etree
import xml.etree.ElementTree as ET
from lxml.builder import E
import re

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

GROBID_API_URL = 'https://orkg.org/grobid/api/processFulltextDocument'

class TreeCRFVanilla(nn.Module):

    def __init__(self, log_potentials, lengths=None):
        self.log_potentials = log_potentials
        self.lengths = lengths
        return

    @lazy_property
    def entropy(self):
        batch_size = self.log_potentials.size(0)
        device = self.log_potentials.device
        return torch.zeros(batch_size).to(device)

    @lazy_property
    def partition(self):
        # Inside algorithm
        device = self.log_potentials.device
        batch_size = self.log_potentials.size(0)
        max_len = self.log_potentials.size(1)
        label_size = self.log_potentials.size(3)

        beta = torch.zeros_like(self.log_potentials).to(device)
        for i in range(max_len):
            beta[:, i, i] = self.log_potentials[:, i, i]
        for d in range(1, max_len):
            for i in range(max_len - d):
                j = i + d
                before_lse_1 = beta[:, i, i:j].view(batch_size, d, label_size, 1)
                before_lse_2 = beta[:, i + 1: j + 1, j].view(batch_size, d, 1, label_size)
                before_lse = (before_lse_1 + before_lse_2).reshape(batch_size, -1)
                after_lse = torch.logsumexp(before_lse, -1).view(batch_size, 1)
                beta[:, i, j] = self.log_potentials[:, i, j] + after_lse
        if (self.lengths is None):
            before_lse = beta[:, 0, max_len - 1]
        else:
            before_lse = tmu.batch_index_select(beta[:, 0], self.lengths - 1)
        log_z = torch.logsumexp(before_lse, -1)
        return log_z

    @lazy_property
    def argmax(self):
        raise NotImplementedError('slow argmax not implemented!')
        return
    
def get_structure_smoothing_mask_v1(mask, lengths, ratio):
    """
    Args:
      mask:
      lengths:
      ratio: Float,
      label_size: Int,

    Returns:
      mask_smooth:
    """
    # print('DEBUG, mask.size = ', mask.size())
    mask_observed = mask
    mask_rejected = (1 - mask) * (1 - ratio)
    mask_smooth = mask_observed + mask_rejected
    return mask_smooth

class TreeCRFLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.potential_normalization = config.potential_normalization
        self.observed_label_size = config.observed_label_size
        self.dropout = config.state_dropout_p
        self.dropout_mode = config.state_dropout_mode
        self.structure_smoothing = config.structure_smoothing_p
        self.decode_method = config.decode_method
        self.use_vanilla_crf = config.use_vanilla_crf
        self.no_batchify = config.no_batchify
        return

    def forward(self, log_potentials, mask, lengths):
        """Partially marginalize the given tree

        Args:
          log_potentials: torch.FloatTensor,
            size=[batch, max_len, max_len, label_size]
          mask: torch.FloatTensor,
            size=[batch, max_len, max_len, label_size]. 1 = not masked, 0 = masked
          lengths: torch.LongTensor, size=[batch]

        Returns:
          log_prob: torch.FloatTensor, size=[batch]
          entropy: torch.FloatTensor, size=[batch]
        """
        inspect = {}

        device = log_potentials.device
        batch_size = log_potentials.size(0)
        max_len = log_potentials.size(1)
        label_size = mask.size(-1)

        if(self.use_vanilla_crf): TreeCRF_ = TreeCRFVanilla
        else: TreeCRF_ = TreeCRF

        if (self.potential_normalization):
            lp_mean = log_potentials.reshape(batch_size, -1).mean(-1)
            lp_std = log_potentials.reshape(batch_size, -1).std(-1)
            log_potentials = log_potentials - lp_mean.view(batch_size, 1, 1, 1)
            log_potentials = log_potentials / lp_std.view(batch_size, 1, 1, 1)

        full_crf = TreeCRF_(log_potentials, lengths=lengths)
        z_full = full_crf.partition
        inspect['z_full'] = z_full.mean().item()
        entropy = full_crf.entropy

        # State dropout
        if (self.dropout > 0.0):
            dropout_dist = Categorical(torch.tensor([self.dropout, 1. - self.dropout]))

            # observed mask
            # [batch, max_len, max_len]
            dropout_mask_observed = dropout_dist.sample(mask.size()[:-1]).to(device)
            # [batch, max_len, max_len, observed_label_size]
            dropout_mask_observed = dropout_mask_observed.unsqueeze(-1) \
                .repeat(1, 1, 1, self.observed_label_size)
            ones_latent_ = torch.ones(mask.size())[:, :, :, self.observed_label_size:]
            ones_latent_ = ones_latent_.long()
            # [batch, max_len, max_len, latent_label_size]
            ones_latent_ = ones_latent_.to(device)
            dropout_mask_observed = torch.cat(
                [dropout_mask_observed, ones_latent_], dim=3)

            # latent mask
            dropout_mask_latent = dropout_dist.sample(mask.size()).to(device)
            if (self.dropout_mode == 'full'):
                mask *= dropout_mask_observed
                dropout_mask_latent[:, :, :, :self.observed_label_size] = 1.
                mask *= dropout_mask_latent
            elif (self.dropout_mode == 'latent'):
                dropout_mask_latent[:, :, :, :self.observed_label_size] = 1.
                mask *= dropout_mask_latent
            else:
                raise NotImplementedError('Illegal dropout mode %s' % self.dropout_mode)

        # Structure smoothing
        if (self.structure_smoothing < 1.0):
            if (self.dropout > 0.0):
                raise ValueError('do not support state dropout when doing smoothing!')
            mask_smooth = get_structure_smoothing_mask_v1(
                mask, lengths, self.structure_smoothing)
            smoothed_potentials = log_potentials + torch.log(mask_smooth + 1e-10)
            smoothed_crf = TreeCRF_(smoothed_potentials, lengths=lengths)
            z_smooth = smoothed_crf.partition
            log_prob_smooth = z_smooth - z_full
            inspect['z_smooth'] = z_smooth.mean().item()
        else:
            log_prob_smooth = torch.zeros(batch_size) - 1

        masked_potentials = log_potentials - 1000000 * (1 - mask)
        if(self.no_batchify):
          z_partial = []
          for i in range(batch_size):
            potential_i = masked_potentials[i].unsqueeze(0)
            len_i = lengths[i].unsqueeze(0)
            z_partial.append(TreeCRF_(potential_i, len_i).partition[0])
          z_partial = torch.stack(z_partial)
        else:
          masked_crf = TreeCRF_(masked_potentials, lengths=lengths)
          z_partial = masked_crf.partition
        inspect['z_partial'] = z_partial.mean().item()

        log_prob = z_partial - z_full
        return log_prob, log_prob_smooth, entropy, inspect

    def decode(self, log_potentials, lengths):
        """Decode the max-prob tree

        Args:
          log_potentials: torch.FloatTensor,
            size=[batch, max_len, max_len, label_size]
          mask: torch.FloatTensor,
            size=[batch, max_len, max_len, label_size]. 1 = not masked, 0 = masked

        Returns:
          trees: torch.LongTensor, size=[batch, max_len, max_len]
            trees[bi, j, k] = l means for the sentence bi in a batch, there is a
            constituent labeled l (l != 0) ranging from location j to
        """
        label_size = log_potentials.size(-1)
        device = log_potentials.device

        if(self.decode_method == 'argmax'):
          crf = TreeCRF(log_potentials, lengths=lengths)
          trees = crf.argmax
        elif(self.decode_method == 'marginal'):
          crf = TreeCRF(log_potentials, lengths=lengths)
          marginals = crf.marginals
          crf_marginal = TreeCRF((marginals + 1e-10).log(), lengths=lengths)
          trees = crf_marginal.argmax
        else: 
          raise NotImplementedError(
            'decode method %s not implemented' % self.decode_method)

        ind = 1 + torch.arange(label_size).to(device).view(1, 1, 1, -1)
        trees = (trees * ind).sum(dim=-1)
        trees = trees - 1
        return trees
    
def partial_mask_to_targets(mask):
    device = mask.device
    label_size = mask.size(-1)
    ind = 1 + torch.arange(label_size).to(device).view(1, 1, 1, -1)
    trees = (mask * ind).sum(dim=-1)
    trees = trees - 1
    tree_rej_ind = trees == -1
    trees[tree_rej_ind] = label_size - 1
    return trees

class Bilinear(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.parse_proj = nn.Parameter(
            torch.randn(config.label_size, config.hidden_size, config.hidden_size))
        return

    def forward(self, sent_states):
        # prepare for tree CRF
        label_size = self.parse_proj.size(0)
        batch_size = sent_states.size(0)
        max_len = sent_states.size(1)
        hidden_size = sent_states.size(2)
        sent_states = sent_states.view(batch_size, 1, max_len, hidden_size)
        sent_states_ = sent_states.transpose(2, 3)  # [batch, 1, hidden_size, max_len]
        parse_proj = self.parse_proj.view(1, label_size, hidden_size, hidden_size)

        # project to CRF potentials
        # [batch, 1, len, hidden] * [1, label, hidden, hidden] -> [batch, label, len, hidden]
        proj = torch.matmul(sent_states, parse_proj)
        # [batch, label, len, hidden] * [batch, 1, hidden, len] -> [batch, label, len, len]
        log_potentials = torch.matmul(proj, sent_states_)
        # [batch, label, len, len] -> [batch, label, len * len] -> [[batch, len * len, label]
        log_potentials = log_potentials.view(batch_size, label_size, -1).transpose(1, 2)
        # [[batch, len * len, label] -> [[batch, len, len, label]
        log_potentials = log_potentials.view(batch_size, max_len, max_len, label_size)
        return log_potentials


class BiAffine(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.parse_proj = nn.Parameter(
            torch.randn(config.label_size, config.hidden_size, config.hidden_size))
        self.offset_proj = nn.Parameter(
            torch.randn(config.hidden_size, config.label_size))
        self.offset = nn.Parameter(torch.randn(config.label_size))
        return

    def forward(self, sent_states):
        label_size = self.parse_proj.size(0)
        batch_size = sent_states.size(0)
        max_len = sent_states.size(1)
        hidden_size = sent_states.size(2)
        sent_states = sent_states.view(batch_size, 1, max_len, hidden_size)
        sent_states_ = sent_states.transpose(2, 3)  # [batch, 1, hidden_size, max_len]
        parse_proj = self.parse_proj.view(1, label_size, hidden_size, hidden_size)

        # project to CRF potentials

        # binear part
        # [batch, 1, len, hidden] * [1, label, hidden, hidden] -> [batch, label, len, hidden]
        proj = torch.matmul(sent_states, parse_proj)
        # [batch, label, len, hidden] * [batch, 1, hidden, len] -> [batch, label, len, len]
        log_potentials = torch.matmul(proj, sent_states_)
        # [batch, label, len, len] -> [batch, label, len * len] -> [[batch, len * len, label]
        log_potentials = log_potentials.view(batch_size, label_size, -1).transpose(1, 2)
        # [[batch, len * len, label] -> [[batch, len, len, label]
        log_potentials_0 = log_potentials.view(batch_size, max_len, max_len, label_size)

        # local offset
        sent_states_sum_0 = sent_states.view(batch_size, max_len, 1, hidden_size)
        sent_states_sum_1 = sent_states.view(batch_size, 1, max_len, hidden_size)
        # [batch, len, 1, hidden] + [batch, 1, len, hidden] -> [batch, len, len, hidden]
        sent_states_sum = (sent_states_sum_0 + sent_states_sum_1).view(batch_size, -1, hidden_size)
        offset_proj = self.offset_proj.view([1, hidden_size, -1])
        # [batch, len * len, hidden] * [1, hidden, label] -> [batch, len * len, label]
        log_potentials_1 = torch.matmul(sent_states_sum, offset_proj)
        log_potentials_1 = log_potentials_1.view(batch_size, max_len, max_len, label_size)

        offset = self.offset.view(1, 1, 1, label_size)
        log_potentials = log_potentials_0 + log_potentials_1 + offset
        return log_potentials


class DeepBiaffine(nn.Module):
    def __init__(self, config):
        super().__init__()

        config_ = deepcopy(config)
        config_.hidden_size = config.hidden_size // 2
        self.biaffine = BiAffine(config_)
        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config_.hidden_size),
            nn.Dropout(config.parser_dropout),
            nn.Linear(config_.hidden_size, config_.hidden_size),
            nn.Dropout(config.parser_dropout)
        )
        return

    def forward(self, sent_states):
        sent_states = self.linear(sent_states)
        log_potentials = self.biaffine(sent_states)
        return log_potentials

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, gather_ids, gather_masks, partial_masks, original_text=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.gather_ids = gather_ids
        self.gather_masks = gather_masks
        self.partial_masks = partial_masks
        self.original_text = original_text


class PartialPCFG(BertPreTrainedModel):

    def __init__(self, config):
        super(PartialPCFG, self).__init__(config)

        self.lambda_ent = config.lambda_ent  # try [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        self.label_size = config.label_size
        self.structure_smoothing = config.structure_smoothing_p < 1.0

        self.use_crf = config.use_crf
        if (self.use_crf is False): assert (config.latent_label_size == 1)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if (config.parser_type == 'bilinear'):
            self.parser = Bilinear(config)
        elif (config.parser_type == 'biaffine'):
            self.parser = BiAffine(config)
        elif (config.parser_type == 'deepbiaffine'):
            self.parser = DeepBiaffine(config)
        else:
            raise NotImplementedError('illegal parser type %s not implemented!' % config.parser_type)
        self.tree_crf = TreeCRFLayer(config)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks, partial_masks):
        """

        Args:
            input_ids: torch.LongTensor, size=[batch, max_len]
            token_type_ids:
            attention_mask:
            gather_ids:
            gather_masks: torch.FloatTensor, size=[batch, max_len]
            partial_masks: torch.FloatTensor, size=[batch, max_len, max_len, label_size]
                label_size = observed_label_size + latent_label_size

        Returns:
            outputs: list 
        """
        inspect = {}
        label_size = self.label_size

        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        # prepare for tree CRF
        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)
        max_len = log_potentials.size(1)
        # TODO: use vanilla span classification 
        if (self.use_crf is False):
            # [batch * max_len * max_len]
            targets = partial_mask_to_targets(partial_masks).view(-1)
            # [batch * max_len * max_len, label_size]
            prob = log_potentials.reshape(-1, label_size)
            loss = F.cross_entropy(prob, targets, reduction='none')

            # [batch, max_len, max_len]
            mask = tmu.lengths_to_squared_mask(lengths, max_len)
            # [batch, max_len, max_len] -> [batch * max_len * max_len]
            mask = torch.triu(mask.float()).view(-1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            # log_prob_sum_partial.size = [batch]
            # TODO: check partial_masks boundary, Done
            log_prob_sum_partial, log_prob_smooth, entropy, inspect_ = \
                self.tree_crf(log_potentials, partial_masks, lengths)

            if (self.structure_smoothing):
                loss = -log_prob_smooth.mean()
            else:
                loss = -log_prob_sum_partial.mean()
            loss -= self.lambda_ent * entropy.mean()

        outputs = [loss, inspect]
        return outputs
    
    def infer(self, input_ids, token_type_ids, attention_mask, gather_ids, gather_masks):
        label_size = self.label_size

        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                        attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        batch_size, sequence_length, hidden_size = sequence_output.shape
        gather_ids = gather_ids.reshape(batch_size * sequence_length, -1).repeat(1, hidden_size).reshape(
            sequence_output.shape)
        gather_output = sequence_output.gather(1, gather_ids)  # [batch, max_len, hidden_size]

        log_potentials = self.parser(gather_output)
        lengths = gather_masks.sum(1)

        if (self.use_crf is False):
            # [batch, max_len, max_len]
            trees = log_potentials.argmax(-1)
            max_len = log_potentials.size(1)
            # [batch, max_len, max_len]
            mask = tmu.lengths_to_squared_mask(lengths, max_len)
            mask = torch.triu(mask.float())
            trees = trees * mask - (1. - mask)
        else:
            trees = self.tree_crf.decode(log_potentials, lengths)

        outputs = [trees, log_potentials]  # CHANGED: Also return log_potentials
        return outputs
    
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, pos, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.pos = pos
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, gather_ids, gather_masks, partial_masks, original_text=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.gather_ids = gather_ids
        self.gather_masks = gather_masks
        self.partial_masks = partial_masks
        self.original_text = original_text
    
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, logger):
        self.logger = logger

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines
    
class Processor(DataProcessor):
    """Processor NQG data set."""

    def __init__(self, logger, dataset, latent_size):
        self.logger = logger
        if dataset == "ACE04" or dataset == "ACE05":
            self.labels = ['PER', 'LOC', 'ORG', 'GPE', 'FAC', 'VEH', 'WEA']
        elif dataset == "GENIA":
            self.labels = ['None', 'G#RNA', 'G#protein', 'G#DNA', 'G#cell_type', 'G#cell_line']
        elif dataset == "pp":
            self.labels = [
    'G#DiagnosticDevice', 'G#ElectrodeConfiguration','G#ElectrodeMaterial', 'G#Experiment','G#Modelling', 'G#PhysicalEffect',   'G#PlasmaApplication', 'G#PlasmaMedium', 'G#PlasmaProperties', 'G#PlasmaTarget', 'G#Unit', 'G#PhysicalQuantity', 'G#Species', 'G#PowerSupply', 'G#DischargeRegime', 'G#PlasmaSource',
]
        else:
            raise NotImplementedError()

        if dataset == "ACE05" or dataset == "GENIA" or dataset == "ACE04":
            self.interval = 4
        elif dataset == "pp":
            self.interval = 3
        else:
            raise NotImplementedError()

        self.latent_size = latent_size

    def get_train_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "dev")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(0, len(lines), self.interval):
            text_a = lines[i]
            label = lines[i + 1]
            examples.append(
                InputExample(guid=len(examples), text_a=text_a, pos=None, label=label))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []

        for (ex_index, example) in enumerate(tqdm(examples)):

            tokens = tokenizer.tokenize(example.text_a)

            gather_ids = list()
            for (idx, token) in enumerate(tokens):
                if (not token.startswith("##") and idx < max_seq_length - 2):
                    gather_ids.append(idx + 1)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            gather_padding = [0] * (max_seq_length - len(gather_ids))
            gather_masks = [1] * len(gather_ids) + gather_padding
            gather_ids += gather_padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(gather_ids) == max_seq_length
            assert len(gather_masks) == max_seq_length

            partial_masks = self.generate_partial_masks(example.text_a.split(' '), max_seq_length, example.label,
                                                        self.labels)

            if ex_index < 2:
                self.logger.info("*** Example ***")
                self.logger.info("guid: %s" % (example.guid))
                self.logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                self.logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                self.logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                self.logger.info(
                    "gather_ids: %s" % " ".join([str(x) for x in gather_ids]))
                self.logger.info(
                    "gather_masks: %s" % " ".join([str(x) for x in gather_masks]))
                # self.logger.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              original_text=example.text_a))

        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def generate_partial_masks(self, tokens, max_seq_length, labels, tags):

        total_tags_num = len(tags) + self.latent_size

        labels = labels.split('|')
        label_list = list()

        for label in labels:
            if not label:
                continue
            sp = label.strip().split(' ')
            start, end = sp[0].split(',')[:2]
            start = int(start)
            end = int(end) - 1
            label_list.append((start, end, sp[1]))

        mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection

        for start, end, tag in label_list:

            if start < max_seq_length and end < max_seq_length:
                tag_idx = tags.index(tag)
                mask[start][end][tag_idx] = 1
                for k in range(total_tags_num):
                    if k != tag_idx:
                        mask[start][end][k] = 0

            for i in range(l):
                if i > end:
                    continue
                for j in range(i, l):
                    if j < start:
                        continue
                    if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
                        for k in range(total_tags_num):
                            mask[i][j][k] = 0

        for i in range(l):
            for j in range(0, i):
                for k in range(total_tags_num):
                    mask[i][j][k] = 0

        for i in range(l):
            for j in range(i, l):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        if k < len(tags):
                            mask[i][j][k] = 0
                        else:
                            mask[i][j][k] = 1

        for i in range(max_seq_length):
            for j in range(max_seq_length):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        mask[i][j][k] = 0

        return mask

class InferenceProcessor:
    """Process raw text for inference"""
    
    def __init__(self, tokenizer, processor, max_seq_length=64):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.latent_size = processor.latent_size
    
    def process_text_file(self, text):
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        
        sentences = []
        
        parts = text.split('. ')
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            # Add period back (except for the last part if original text didn't end with period)
            if i < len(parts) - 1:
                sentence = part + ' .'
            else:
                # Check if original text ended with period
                if text.rstrip().endswith('.'):
                    sentence = part + ' .'
                else:
                    sentence = part
            
            sentences.append(sentence)
        
        logger.info(f"Extracted {len(sentences)} sentences from file")
        
        return sentences
    
    def sentences_to_features(self, sentences):
        features = []
        original_texts = []
        
        logger.info(f"Processing {len(sentences)} sentences...")
        
        for sentence in tqdm(sentences, desc="Tokenizing"):
            # Tokenize
            tokens = self.tokenizer.tokenize(sentence)
            
            # Build gather_ids (same logic as in training)
            gather_ids = []
            for idx, token in enumerate(tokens):
                if not token.startswith("##") and idx < self.max_seq_length - 2:
                    gather_ids.append(idx + 1)
            
            # Truncate if needed
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:self.max_seq_length - 2]
            
            # Add special tokens
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            
            # Convert to IDs
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            
            # Padding
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            
            gather_padding = [0] * (self.max_seq_length - len(gather_ids))
            gather_masks = [1] * len(gather_ids) + gather_padding
            gather_ids += gather_padding
            
            # Create dummy partial masks (all zeros for inference)
            total_tags_num = len(self.processor.labels) + self.latent_size
            partial_masks = [[[0 for _ in range(total_tags_num)] 
                             for _ in range(self.max_seq_length)] 
                            for _ in range(self.max_seq_length)]
            
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    partial_masks=partial_masks,
                    gather_ids=gather_ids,
                    gather_masks=gather_masks,
                    original_text=sentence
                )
            )
            original_texts.append(sentence)
        
        return features, original_texts
    
def pdf_to_txt(pdf_file):
    if pdf_file.endswith('.pdf'):
            #file_path = os.path.join(directory, filename)
            with open(pdf_file, 'rb') as file:
                #print(file)
                files = {'input': file}
                response = requests.post(GROBID_API_URL, files=files)
                if response.status_code == 200:
                    print(True)
                    xml_tree = etree.fromstring(response.content)
                    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
                    paper_content = extract_paper_content(xml_tree, ns)
                    paper_metadata = extract_paper_metadata(xml_tree, ns)

                    textcontent=""
                    textcontent=textcontent+"Abstract\n"
                    textcontent=textcontent+paper_metadata['abstract']+"\n\n"
                    for title, content in paper_content.items():
                        textcontent=textcontent+title+"\n"
                        textcontent=textcontent+content+"\n\n"
                    
    return textcontent

def extract_paper_metadata(tree, namespaces):
    details = {}
    # Extracting paper title
    title_elem = tree.find('.//tei:title[@type="main"]', namespaces)
    details['title'] = title_elem.text if title_elem is not None else 'Not available'

    # Extracting authors
    authors = []
    for author_elem in tree.findall('.//tei:teiHeader//tei:author', namespaces):
        author = {}
        pers_name = author_elem.find('.//tei:persName', namespaces)
        if pers_name is not None:
            author['name'] = pers_name.findtext('.//tei:forename', namespaces=namespaces, default='Not available') + " " + pers_name.findtext('.//tei:surname', namespaces=namespaces, default='Not available')
        author['orcid'] = author_elem.findtext('.//tei:idno[@type="orcid"]', namespaces=namespaces, default='Not available')
        author['affiliation'] = author_elem.findtext('.//tei:affiliation/tei:orgName', namespaces=namespaces, default='Not available')
        authors.append(author)
    details['authors'] = authors

    abstract_elem = tree.find('.//tei:abstract', namespaces)
    abstract_text = etree.tostring(abstract_elem, method='text', encoding='unicode').strip() if abstract_elem is not None else 'Not available'
    details['abstract'] = abstract_text
    # Extracting DOI
    details['doi'] = tree.findtext('.//tei:idno[@type="DOI"]', namespaces=namespaces, default='Not available')

    # Extracting date published
    publication_date = tree.findtext('.//tei:date[@type="published"]', namespaces=namespaces, default='Not available')
    # Removing any non-date characters
    details['date_published'] = re.sub('[^0-9-]', '', publication_date)
    return details

def extract_paper_content(tree, ns):
    sections = {}
    # Iterate over each head tag in the body
    for head in tree.xpath('//tei:body/tei:div/tei:head', namespaces=ns):
        heading = head.text.strip() if head.text else "Unnamed Section"
        content = []

        # Find all the subsequent p tags until the next head tag
        for sibling in head.itersiblings():
            if sibling.tag == '{http://www.tei-c.org/ns/1.0}p':
                paragraph = sibling.xpath('string(.)', namespaces=ns).strip()
                content.append(paragraph)
            elif sibling.tag == '{http://www.tei-c.org/ns/1.0}head':
                break

        sections[heading] = "\n".join(content)

    return sections


def extract_entities_from_outputs(outputs, input_ids, gather_masks, log_potentials, 
                                  tokenizer, processor, original_texts):
    """Extract entities from model outputs"""
    
    outputs = outputs.cpu().numpy()
    gather_masks = gather_masks.sum(1).cpu().numpy()
    input_ids = input_ids.cpu().numpy()
    
    if log_potentials is not None:
        probs = torch.softmax(log_potentials, dim=-1).cpu().numpy()
    
    all_predictions = []
    
    for idx, (output, l) in enumerate(zip(outputs, gather_masks)):
        # Get tokens
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[idx])
        
        # Build gather mapping
        gather_ids_list = []
        for token_idx, token in enumerate(all_tokens):
            if not token.startswith("##") and token not in ['[PAD]', '[SEP]']:
                if token != '[CLS]':
                    gather_ids_list.append(token_idx)
        
        gather_to_word = {}
        word_idx = 0
        for gather_pos, token_pos in enumerate(gather_ids_list):
            gather_to_word[gather_pos] = word_idx
            word_idx += 1
        
        # Get sentence
        sentence = original_texts[idx]
        words = sentence.split()
        
        # Extract entities
        entities = []
        for i in range(int(l)):
            for j in range(int(l)):
                if output[i][j] >= 0 and output[i][j] < len(processor.labels):
                    if i in gather_to_word and j in gather_to_word:
                        word_start = gather_to_word[i]
                        word_end = gather_to_word[j]
                        label_id = int(output[i][j])
                        label_name = processor.labels[label_id]
                        
                        # Get confidence
                        confidence = 1.0
                        if log_potentials is not None:
                            confidence = float(probs[idx, i, j, label_id])
                        
                        # Extract entity text
                        entity_text = ' '.join(words[word_start:word_end + 1]) if word_end + 1 <= len(words) else ''
                        
                        entities.append({
                            'start': word_start,
                            'end': word_end + 1,
                            'label': label_name,
                            'confidence': confidence,
                            'text': entity_text
                        })
        
        all_predictions.append({
            'sentence': sentence,
            'entities': entities
        })
    
    return all_predictions


def predict(model, tokenizer, processor, input_file, device, batch_size=8):
    """
    Make predictions on new text
    Args:
        model: trained model
        tokenizer: BERT tokenizer
        processor: Processor instance
        input_file: path to input text file
        device: torch device
        output_file: path to save predictions
        batch_size: batch size for inference
    Returns:
        predictions: list of dicts with sentence and entities
    """
    model.eval()
    
    # Process input
    inference_processor = InferenceProcessor(tokenizer, processor)
    sentences = inference_processor.process_text_file(input_file)
    features, original_texts = inference_processor.sentences_to_features(sentences)
    
    logger.info(f"Processed {len(features)} sentences")
    
    # Convert to dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_gather_ids = torch.tensor([f.gather_ids for f in features], dtype=torch.long)
    all_gather_masks = torch.tensor([f.gather_masks for f in features], dtype=torch.long)
    all_partial_masks = torch.tensor([f.partial_masks for f in features], dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                           all_gather_ids, all_gather_masks, all_partial_masks)
    
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    all_predictions = []
    
    batch_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Making predictions"):
            batch = tuple(t.to(device) for t in batch)
            
            # Get batch texts
            batch_size_actual = batch[0].size(0)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size_actual
            batch_texts = original_texts[start_idx:end_idx]
            batch_idx += 1
            
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'gather_ids': batch[3],
                'gather_masks': batch[4]
            }
            
            outputs = model.infer(**inputs)
            trees = outputs[0]
            log_potentials = outputs[1] if len(outputs) > 1 else None
            
            # Extract entities
            batch_predictions = extract_entities_from_outputs(
                trees, batch[0], batch[4], log_potentials,
                tokenizer, processor, batch_texts
            )
            
            all_predictions.extend(batch_predictions)
            
            # Write to file if provided
            #if out_f:
                #for pred in batch_predictions:
                    #out_f.write(pred['sentence'] + '\n')
                    #entity_strs = []
                    #for entity in pred['entities']:
                        #entity_strs.append(
                            #f"{entity['start']},{entity['end']} {entity['label']} {entity['confidence']:.4f}"
                        #)
                    #out_f.write('|'.join(entity_strs) + '\n')
                    #out_f.write('\n')
    
    #if out_f:
        #out_f.close()
        #logger.info(f"Predictions saved to {output_file}")
    
    return all_predictions


def build_knowledge_graph(predictions, output_file='knowledge_graph.json'):
    kg = {
        'entities': [],
        'entity_count_by_type': {},
        'entity_index': {},
        'sentences': []
    }
    
    entity_id = 0
    entity_map = {}
    
    for pred in predictions:
        sentence_entities = []
        
        for entity in pred['entities']:
            entity_text = entity['text'].strip()
            entity_type = entity['label']
            entity_key = (entity_text, entity_type)
            
            # Add entity if not exists
            if entity_key not in entity_map:
                entity_map[entity_key] = entity_id
                kg['entities'].append({
                    'id': entity_id,
                    'text': entity_text,
                    'type': entity_type,
                    'confidence': entity['confidence']
                })
                
                # Track entity types
                if entity_type not in kg['entity_count_by_type']:
                    kg['entity_count_by_type'][entity_type] = 0
                kg['entity_count_by_type'][entity_type] += 1
                
                # Build index
                if entity_type not in kg['entity_index']:
                    kg['entity_index'][entity_type] = []
                kg['entity_index'][entity_type].append({
                    'id': entity_id,
                    'text': entity_text
                })
                
                entity_id += 1
            
            sentence_entities.append({
                'entity_id': entity_map[entity_key],
                'start': entity['start'],
                'end': entity['end'],
                'text': entity_text,
                'type': entity_type
            })
        
        kg['sentences'].append({
            'text': pred['sentence'],
            'entities': sentence_entities
        })
    
    # Save to file
    #with open(output_file, 'w', encoding='utf-8') as f:
        #json.dump(kg, f, indent=2, ensure_ascii=False)
    
    #logger.info(f"Entities saved to {output_file}")
    logger.info(f"Total unique entities: {len(kg['entities'])}")
    logger.info(f"Entity types: {list(kg['entity_count_by_type'].keys())}")
    for entity_type, count in kg['entity_count_by_type'].items():
        logger.info(f"  {entity_type}: {count}")
    
    return kg

def process_paper_text(paper_text, output_folder, model, tokenizer, processor, device, 
                       grobid_url, batch_size=8, save_txt=True):
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PDF files
    #pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    #logger.info(f"Found {len(pdf_files)} PDF files in {pdf_folder}")
    
    #if not pdf_files:
        #logger.warning("No PDF files found!")
        #return
    
    #for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        #pdf_name = Path(pdf_file).stem
        #logger.info(f"\n{'='*60}")
        #logger.info(f"Processing: {pdf_name}")
        #logger.info(f"{'='*60}")
        
        #logger.info("Converting PDF to text using GROBID...")
    text = paper_text
        
    if text is None:
        logger.error(f"Failed to process file ...")
        #continue
        
        #if save_txt:
            #txt_file = os.path.join(output_folder, f"{pdf_name}.txt")
            #with open(txt_file, 'w', encoding='utf-8') as f:
                #f.write(text)
        #logger.info(f"Saved extracted text to {txt_file}")
        
        # Make predictions
    logger.info("Making predictions...")
    #pred_file = os.path.join(output_folder, f"{pdf_name}_predictions.txt")
    predictions = predict(model, tokenizer, processor, text, device, batch_size)
        
    #kg_file = os.path.join(output_folder, f"{pdf_name}_entites.json")
    kg = build_knowledge_graph(predictions)
        
    #logger.info(f"Completed processing {pdf_name}")
    #logger.info(f"  - Entities: {kg_file}")
    logger.info(f"  - Total entities: {len(kg['entities'])}")


def generate_annotations(paper_text):
    pdf_folder = 'papers'
    output_folder = 'annotations'
    grobid_url = ''
    batch_size = 8
    save_txt = True
    dataset = 'pp'
    latent_size = 1
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', 
        do_lower_case=False, 
        do_basic_tokenize=False
    )
    
    # Load processor
    logger.info("Loading processor...")
    processor = Processor(logger, dataset=dataset, latent_size=1)
    logger.info(f"Labels: {processor.labels}")
    
    model_path = 'trainedmodel'
    logger.info(f"Loading model from {model_path}...")
    config = BertConfig.from_pretrained(model_path)
    
    # Set config parameters
    config.label_size = len(processor.labels) + latent_size
    config.observed_label_size = len(processor.labels)
    config.latent_label_size = config.label_size - config.observed_label_size
    
    # Load model
    model = PartialPCFG.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")

    process_paper_text(
        pdf_folder, 
        output_folder, 
        model, 
        tokenizer, 
        processor, 
        device,
        grobid_url,
        batch_size,
        save_txt
    )