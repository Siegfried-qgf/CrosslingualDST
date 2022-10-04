import os
import math
from venv import create
import spacy
import random
import difflib
from tqdm import tqdm
from difflib import get_close_matches
from itertools import chain
from collections import OrderedDict, defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer,AutoTokenizer
from torch.utils.data import Dataset

from utils import definitions
from utils.io_utils import load_json, load_pickle, save_pickle, get_or_create_logger,save_json
logger = get_or_create_logger(__name__)

class MultiWOZDataset(Dataset):
    def __init__(self, reader, l_type ,data_type,context_size):
        super().__init__()
        self.reader = reader
        self.l_type=l_type
        if self.l_type=="en":
            self.dials = self.reader.data_s[data_type]
        else:
            self.dials=self.reader.data_t[data_type]
        self.context_size=context_size
        self.create_turn_batch()

    def create_turn_batch(self):
        logger.info("Creating turn batches...")
        self.turn_encoder_input_ids = []
        self.turn_belief_label_ids = []
        for dial in tqdm(self.dials, desc='Creating turn batches'):
            dial_history = []
            for turn in dial:
                context = self.flatten_dial_history(dial_history, len(turn['user']),self.context_size)
                encoder_input_ids = context + turn['user'] + [self.reader.eos_token_id]
                bspn = turn['bspn']
                bspn_label = bspn
                belief_label_ids = bspn_label + [self.reader.eos_token_id]
                self.turn_encoder_input_ids.append(encoder_input_ids)
                self.turn_belief_label_ids.append(belief_label_ids)
                turn_text = turn['user'] + bspn +  turn['resp']
                dial_history.append(turn_text)
        print("turn_num:{}".format(len(self.turn_encoder_input_ids)))

    def flatten_dial_history(self, dial_history,  len_postfix, context_size):
        if context_size > 0:
            context_size -= 1

        if context_size == 0:
            windowed_context = []
        elif context_size > 0:
            windowed_context = dial_history[-context_size:]
        else:
            windowed_context = dial_history

        ctx_len = sum([len(c) for c in windowed_context])

        # consider eos_token
        spare_len = self.reader.max_seq_len - len_postfix - 1
        while ctx_len >= spare_len:
            ctx_len -= len(windowed_context[0])
            windowed_context.pop(0)

        context = list(chain(*windowed_context))

        return context

    def __len__(self):
        return len(self.turn_encoder_input_ids)

    def __getitem__(self, index):
        return self.turn_encoder_input_ids[index],  self.turn_belief_label_ids[index]

class BaseReader(object):
    def __init__(self, s_type,t_type):
        self.s_type = s_type
        self.t_type = t_type
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = self.init_tokenizer()
        self.s_data_dir,self.t_data_dir = self.get_data_dir()
        encoded_data_path_s = os.path.join(self.s_data_dir, "encoded_data.pkl")
        encoded_data_path_t = os.path.join(self.t_data_dir, "encoded_data.pkl")

        if os.path.exists(encoded_data_path_s):
            logger.info("Load encoded english data from {}".format(encoded_data_path_s))
            self.data_s = load_pickle(encoded_data_path_s)
        else:
            logger.info("Encode english data and save to {}".format(encoded_data_path_s))
            train,_ = self.encode_data("train")
            dev_s,dev_t = self.encode_data("dev")
            test_s,test_t = self.encode_data("test")
            self.data_s = {"train": train, "dev": dev_s, "test": test_s}
            save_pickle(self.data_s, encoded_data_path_s)

        if os.path.exists(encoded_data_path_t):
            logger.info("Load target encoded data from {}".format(encoded_data_path_t))
            self.data_t = load_pickle(encoded_data_path_t)
        else:
            logger.info("Encode target data and save to {}".format(encoded_data_path_t))
            dev_s, dev_t = self.encode_data("dev")
            test_s, test_t = self.encode_data("test")
            self.data_t = {"dev": dev_t, "test": test_t}
            save_pickle(self.data_t, encoded_data_path_t)



        span_tokens = [self.pad_token, "O"]
        for slot in definitions.EXTRACTIVE_SLOT:
            #span_tokens.append("B-{}".format(slot))
            #span_tokens.append("I-{}".format(slot))
            span_tokens.append(slot)

        self.span_tokens = span_tokens

    def get_data_dir(self):
        raise NotImplementedError

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
        save_json(tokenizer.vocab,"vocab.json")
        special_tokens = []

        # add domains
        domains = definitions.ALL_DOMAINS + ["general"]
        for domain in sorted(domains):
            token = "[" + domain + "]"
            special_tokens.append(token)

        # add intents
        intents = list(set(chain(*definitions.DIALOG_ACTS.values())))
        for intent in sorted(intents):
            token = "[" + intent + "]"
            special_tokens.append(token)

        # add slots
        slots = list(set(definitions.ALL_INFSLOT + definitions.ALL_REQSLOT))

        for slot in sorted(slots):
            token = "[value_" + slot + "]"
            special_tokens.append(token)

        special_tokens.extend(definitions.SPECIAL_TOKENS)

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def max_seq_len(self):
        self.tokenizer.model_max_length = 1024
        return self.tokenizer.model_max_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def encode_text(self, text, bos_token=None, eos_token=None):
        tokens = text.split() if isinstance(text, str) else text

        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]

            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]

            tokens = tokens + eos_token

        encoded_text = self.tokenizer.encode(" ".join(tokens))

        # except eos token
        if encoded_text[-1] == self.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def encode_data(self, data_type):
        raise NotImplementedError

class MultiWOZReader(BaseReader):
    def __init__(self,  s_type,t_type):
        super(MultiWOZReader, self).__init__(s_type,t_type)
    #返回源语言和目标语言的data文件夹位置
    def get_data_dir(self):
        return os.path.join("dataset/"+self.s_type+"/processed"),os.path.join("dataset/"+self.t_type+"/processed")

    def encode_data(self, data_type):
        if data_type == "train":
            s_data = load_json(os.path.join(self.s_data_dir, "{}_data_{}.json".format(data_type, self.s_type)))
            datalist = [s_data]
        else:
            s_data = load_json(os.path.join(self.s_data_dir, "{}_data_{}.json".format(data_type,self.s_type)))
            t_data = load_json(os.path.join(self.t_data_dir, "{}_data_{}.json".format(data_type,self.t_type)))
            datalist = [s_data,t_data]
        encoded_data_s=[]
        encoded_data_t = []

        #fn为对话编号 dial为对话内容DICT(goal，log) 对每个item：
        for n,data in enumerate(datalist):
            if data_type=="train" and n==1:
                break
            for fn, dial in tqdm(data.items(), desc=data_type, total=len(data)):
                encoded_dial = []
                accum_constraint_dict = {}
                #对每轮对话：
                for t in dial["log"]:
                    turn_constrain_dict = self.bspn_to_constraint_dict(t["constraint"])
                    for domain, sv_dict in turn_constrain_dict.items():
                        if domain not in accum_constraint_dict:
                            accum_constraint_dict[domain] = {}

                        for s, v in sv_dict.items():
                            if s not in accum_constraint_dict[domain]:
                                accum_constraint_dict[domain][s] = []
                            accum_constraint_dict[domain][s].append(v)

                for idx, t in enumerate(dial["log"]):
                    enc = {}
                    enc["dial_id"] = fn
                    enc["turn_num"] = t["turn_num"]
                    enc["turn_domain"] = t["turn_domain"].split()
                    target_domain = enc["turn_domain"][0] if len(enc["turn_domain"]) == 1 else enc["turn_domain"][1]
                    target_domain = target_domain[1:-1]
                    user_ids = self.encode_text(t["user"],
                                                bos_token=definitions.BOS_USER_TOKEN,
                                                eos_token=definitions.EOS_USER_TOKEN)
                    enc["user"] = user_ids
                    resp_ids = self.encode_text(t["resp"],
                                                bos_token=definitions.BOS_RESP_TOKEN,
                                                eos_token=definitions.EOS_RESP_TOKEN)
                    enc["resp"] = resp_ids
                    constraint_dict = self.bspn_to_constraint_dict(t["constraint"])
                    ordered_constraint_dict = OrderedDict()
                    for domain, slots in definitions.INFORMABLE_SLOTS.items():
                        if domain not in constraint_dict:
                            continue
                        ordered_constraint_dict[domain] = OrderedDict()
                        for slot in slots:
                            if slot not in constraint_dict[domain]:
                                continue
                            value = constraint_dict[domain][slot]
                            ordered_constraint_dict[domain][slot] = value
                    ordered_bspn = self.constraint_dict_to_bspn(ordered_constraint_dict)

                    bspn_ids = self.encode_text(ordered_bspn,
                                                bos_token=definitions.BOS_BELIEF_TOKEN,
                                                eos_token=definitions.EOS_BELIEF_TOKEN)
                    enc["bspn"] = bspn_ids

                    if (len(enc["user"]) == 0 or len(enc["resp"]) == 0 or
                             len(enc["bspn"]) == 0 ):
                        raise ValueError(fn, idx)
                    # NOTE: if curr_constraint_dict does not include span[domain][slot], remove span[domain][slot] ??

                    '''
                    curr_constraint_dict = self.bspn_to_constraint_dict(t["constraint"])
        
                    # e2e: overwrite span token w.r.t user span information only
                    e2e_constraint_dict = copy.deepcopy(curr_constraint_dict)
                    for domain, sv_dict in e2e_constraint_dict.items():
                        for s, v in sv_dict.items():
                            if domain in user_span and s in user_span[domain]:
                                e2e_constraint_dict[domain][s] = definitions.BELIEF_COPY_TOKEN
        
                            # maintaining copy token if previous value had been copied
                            if (domain in prev_constraint_dict and
                                    s in prev_constraint_dict[domain] and
                                    v == prev_constraint_dict[domain][s]):
                                prev_enc = encoded_dial[-1]
                                prev_e2e_constraint_dict = self.bspn_to_constraint_dict(
                                    self.tokenizer.decode(prev_enc["bspn_e2e"]))
        
                                if prev_e2e_constraint_dict[domain][s] == definitions.BELIEF_COPY_TOKEN:
                                    e2e_constraint_dict[domain][s] = definitions.BELIEF_COPY_TOKEN
        
                    e2e_constraint = self.constraint_dict_to_bspn(e2e_constraint_dict)
        
                    e2e_bspn_ids = self.encode_text(e2e_constraint,
                                                    bos_token=definitions.BOS_BELIEF_TOKEN,
                                                    eos_token=definitions.EOS_BELIEF_TOKEN)
        
                    enc["bspn_e2e"] = e2e_bspn_ids
        
                    # dst: overwirte span token w.r.t user/resp span information
                    dst_constraint_dict = copy.deepcopy(e2e_constraint_dict)
                    for domain, sv_dict in dst_constraint_dict.items():
                        for s, v in sv_dict.items():
                            if domain in resp_span and s in resp_span[domain]:
                                dst_constraint_dict[domain][s] = definitions.BELIEF_COPY_TOKEN
        
                            # maintaining copy token if previous value had been copied
                            if (domain in prev_constraint_dict and
                                  s in prev_constraint_dict[domain] and
                                  v == prev_constraint_dict[domain][s]):
                                prev_enc = encoded_dial[-1]
                                prev_dst_constraint_dict = self.bspn_to_constraint_dict(
                                    self.tokenizer.decode(prev_enc["bspn_dst"]))
        
                                if prev_dst_constraint_dict[domain][s] == definitions.BELIEF_COPY_TOKEN:
                                    dst_constraint_dict[domain][s] = definitions.BELIEF_COPY_TOKEN
        
                    dst_constraint = self.constraint_dict_to_bspn(dst_constraint_dict)
        
                    dst_bspn_ids = self.encode_text(dst_constraint,
                                                    bos_token=definitions.BOS_BELIEF_TOKEN,
                                                    eos_token=definitions.EOS_BELIEF_TOKEN)
        
                    enc["bspn_dst"] = dst_bspn_ids
                    '''
                    encoded_dial.append(enc)

                    prev_bspn = t["constraint"]
                    #prev_constraint_dict = curr_constraint_dict
                    '''
                    print("dial_id |", enc["dial_id"])
                    print("user | ", self.tokenizer.decode(enc["user"]))
                    print("resp | ", self.tokenizer.decode(enc["resp"]))
                    print("redx | ", self.tokenizer.decode(enc["redx"]))
                    print("bspn | ", self.tokenizer.decode(enc["bspn"]))
                    print("aspn | ", self.tokenizer.decode(enc["aspn"]))
                    print("dbpn | ", self.tokenizer.decode(enc["dbpn"]))
                    print("bspn_e2e | ", self.tokenizer.decode(enc["bspn_e2e"]))
                    print("bspn_dst | ", self.tokenizer.decode(enc["bspn_dst"]))
        
                    for domain, ss_dict in enc["user_span"].items():
                        print("user_span | ", domain)
                        for s, span in ss_dict.items():
                            print("       {}: {}".format(
                                s, self.tokenizer.decode(enc["user"][span[0]:span[1]])))
        
                    for domain, ss_dict in enc["resp_span"].items():
                        print("resp_span | ", domain)
                        for s, span in ss_dict.items():
                            print("       {}: {}".format(
                                s, self.tokenizer.decode(enc["resp"][span[0]:span[1]])))
        
                    input()
                    '''
                if n==0:
                    encoded_data_s.append(encoded_dial)
                else:
                    encoded_data_t.append(encoded_dial)
        return encoded_data_s,encoded_data_t
    #把constraint字段转换成字典   constraint_dict DICT{domain:{slot:value}}
    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn
        constraint_dict = OrderedDict()#有序字典
        domain, slot = None, None
        for token in bspn:
            if token == definitions.EOS_BELIEF_TOKEN:#"<eos_belief>"
                break

            if token.startswith("["):
                token = token[1:-1]

                if token in definitions.ALL_DOMAINS:
                    domain = token

                if token.startswith("value_"):
                    if domain is None:
                        continue

                    if domain not in constraint_dict:
                        constraint_dict[domain] = OrderedDict()

                    slot = token.split("_")[1]

                    constraint_dict[domain][slot] = []

            else:
                try:
                    if domain is not None and slot is not None:
                        constraint_dict[domain][slot].append(token)
                except KeyError:
                    continue

        for domain, sv_dict in constraint_dict.items():
            for s, value_tokens in sv_dict.items():
                constraint_dict[domain][s] = " ".join(value_tokens)

        return constraint_dict

    def constraint_dict_to_bspn(self, constraint_dict):
        tokens = []
        for domain, sv_dict in constraint_dict.items():
            tokens.append("[" + domain + "]")
            for s, v in sv_dict.items():
                tokens.append("[value_" + s + "]")
                tokens.extend(v.split())

        return " ".join(tokens)

class CollatorTrain(object):
    def __init__(self, pad_token_id, tokenizer):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_encoder_input_ids = []
        batch_belief_label_ids = []
        batch_size = len(batch)

        for i in range(batch_size):
            encoder_input_ids,  belief_label_ids = batch[i]
            batch_encoder_input_ids.append(torch.tensor(encoder_input_ids, dtype=torch.long))
            batch_belief_label_ids.append(torch.tensor(belief_label_ids, dtype=torch.long))

        batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_belief_label_ids = pad_sequence(batch_belief_label_ids, batch_first=True, padding_value=self.pad_token_id)
        return batch_encoder_input_ids, batch_belief_label_ids

class BaseIterator(object):
    def __init__(self, reader):
        self.reader = reader

    def bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def construct_mini_batch(self, data, batch_size, num_gpus):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if (len(batch) % num_gpus) != 0:
            batch = batch[:-(len(batch) % num_gpus)]
        if len(batch) > 0.5 * batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def transpose_batch(self, dial_batch):
        turn_batch = []
        turn_num = len(dial_batch[0])
        for turn in range(turn_num):
            turn_l = []
            for dial in dial_batch:
                this_turn = dial[turn]
                turn_l.append(this_turn)
            turn_batch.append(turn_l)
        return turn_batch

    def get_batches(self, data_type, batch_size, num_gpus):
        print(self.reader.data_s.keys())
        print(self.reader.data_t.keys())
        dial = self.reader.data_t[data_type]
        turn_bucket = self.bucket_by_turn(dial)
        all_batches = []
        num_training_steps = 0
        num_turns = 0
        num_dials = 0
        for k in turn_bucket:
            if data_type != "test" and (k == 1 or k >= 17):
                continue
            batches = self.construct_mini_batch(
                turn_bucket[k], batch_size, num_gpus)
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        return all_batches, num_training_steps, num_dials, num_turns

    def flatten_dial_history(self, dial_history, len_postfix, context_size):
        if context_size > 0:
            context_size -= 1
        if context_size == 0:
            windowed_context = []
        elif context_size > 0:
            windowed_context = dial_history[-context_size:]
        else:
            windowed_context = dial_history
        ctx_len = sum([len(c) for c in windowed_context])

        # consider eos_token
        spare_len = self.reader.max_seq_len - len_postfix - 1
        while ctx_len >= spare_len:
            ctx_len -= len(windowed_context[0])
            windowed_context.pop(0)
        context = list(chain(*windowed_context))

        return context

    def tensorize(self, ids):
        return torch.tensor(ids, dtype=torch.long)

    def get_data_iterator(self, all_batches, task, ururu, add_auxiliary_task=False, context_size=-1):
        raise NotImplementedError

class MultiWOZIterator(BaseIterator):
    def __init__(self, reader):
        super(MultiWOZIterator, self).__init__(reader)

    def get_readable_batch(self, dial_batch):
        dialogs = {}

        decoded_keys = ["user", "resp" ,"bspn",
                        "bspn_gen"]
        for dial in dial_batch:
            dial_id = dial[0]["dial_id"]

            dialogs[dial_id] = []

            for turn in dial:
                readable_turn = {}

                for k, v in turn.items():
                    if k == "dial_id":
                        continue
                    elif k in decoded_keys:
                        v = self.reader.tokenizer.decode(
                            v, clean_up_tokenization_spaces=False)

                    readable_turn[k] = v

                dialogs[dial_id].append(readable_turn)

        return dialogs

    def get_data_iterator(self, all_batches, context_size=-1):
        for dial_batch in all_batches:
            batch_encoder_input_ids = []
            batch_belief_label_ids = []

            for dial in dial_batch:
                dial_encoder_input_ids = []
                dial_belief_label_ids = []

                dial_history = []
                for turn in dial:
                    context = self.flatten_dial_history(
                        dial_history, len(turn["user"]), context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    # add current span of user utterance
                    bspn = turn["bspn"]
                    bspn_label = bspn
                    belief_label_ids = bspn_label + [self.reader.eos_token_id]

                    dial_encoder_input_ids.append(encoder_input_ids)
                    dial_belief_label_ids.append(belief_label_ids)



                    turn_text = turn["user"] + bspn + turn["resp"]


                    dial_history.append(turn_text)

                batch_encoder_input_ids.append(dial_encoder_input_ids)
                batch_belief_label_ids.append(dial_belief_label_ids)


            # turn first
            batch_encoder_input_ids = self.transpose_batch(batch_encoder_input_ids)
            batch_belief_label_ids = self.transpose_batch(batch_belief_label_ids)

            num_turns = len(batch_encoder_input_ids)

            tensor_encoder_input_ids = []
            tensor_belief_label_ids = []
            for t in range(num_turns):
                tensor_encoder_input_ids = [
                    self.tensorize(b) for b in batch_encoder_input_ids[t]]
                tensor_belief_label_ids = [
                    self.tensorize(b) for b in batch_belief_label_ids[t]]

                tensor_encoder_input_ids = pad_sequence(tensor_encoder_input_ids,
                                                        batch_first=True,
                                                        padding_value=self.reader.pad_token_id)


                tensor_belief_label_ids = pad_sequence(tensor_belief_label_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)


                yield tensor_encoder_input_ids, tensor_belief_label_ids

