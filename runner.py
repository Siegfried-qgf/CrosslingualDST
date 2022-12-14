from functools import reduce
import os
import re
import copy
import math
import time
import glob
import shutil
from abc import *
from tracemalloc import start
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_outputs import BaseModelOutput
from tensorboardX import SummaryWriter
from transformers import MT5ForConditionalGeneration
from reader import  MultiWOZReader, MultiWOZDataset, CollatorTrain,MultiWOZIterator
from utils import definitions
from utils.io_utils import get_or_create_logger, load_json, save_json
from utils.ddp_utils import reduce_mean, reduce_sum
from evaluator import MultiWozEvaluator
from model import myMT5
logger = get_or_create_logger(__name__)

class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))
        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0
        self.belief_loss = 0.0
        self.belief_correct = 0.0
        self.belief_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.belief_loss += step_outputs["belief"]["loss"]
        self.belief_correct += step_outputs["belief"]["correct"]
        self.belief_count += step_outputs["belief"]["count"]

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step)

    def info_stats(self, data_type, global_step, do_span_stats=False, do_resp_stats=False):
        avg_step_time = self.step_time / self.log_frequency

        belief_ppl = math.exp(self.belief_loss / self.belief_count)
        belief_acc = (self.belief_correct / self.belief_count) * 100

        self.summary_writer.add_scalar(
            "{}/belief_loss".format(data_type), self.belief_loss, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_ppl".format(data_type), belief_ppl, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_acc".format(data_type), belief_acc, global_step=global_step)

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.belief_loss, belief_ppl, belief_acc)

        logger.info(
            " ".join([common_info, belief_info]))

        self.init_stats()

class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader
        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
        else:
            model_path = self.cfg.backbone
        logger.info("Load model from {}".format(model_path))
        model = myMT5.from_pretrained(model_path)
        #model = MT5ForConditionalGeneration.from_pretrained(model_path)
        model.resize_token_embeddings(self.reader.vocab_size)
        model.to(self.cfg.device)
        if self.cfg.num_gpus > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank)
        return model

    def save_model(self, epoch):
        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)
        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model
        model.save_pretrained(save_path)
        if not self.cfg.save_best_model:
            # keep chekpoint up to maximum
            checkpoints = sorted(
                glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
                key=os.path.getmtime,
                reverse=True)
            checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]
            for ckpt in checkpoints_to_be_deleted:
                shutil.rmtree(ckpt)
        return latest_ckpt

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, train_batch_size):
        num_train_steps = (num_traininig_steps_per_epoch *
            self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)
        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        pred = pred.view(-1)
        label = label.view(-1)
        num_count = label.ne(pad_id).long().sum()
        num_correct = torch.eq(pred, label).long().sum()
        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)
        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()
        return num_correct, num_count

    # ????????????????????????????????????
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):

        reader = MultiWOZReader(cfg.s_type, cfg.t_type)
        self.iterator = MultiWOZIterator(reader)
        super(MultiWOZRunner, self).__init__(cfg, reader)


    def step_fn(self, inputs,  belief_labels):
        inputs = inputs.to(self.cfg.device)
        belief_labels = belief_labels.to(self.cfg.device)

        attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)
        belief_outputs = self.model(input_ids=inputs,
                                    attention_mask=attention_mask,
                                    labels=belief_labels,
                                    return_dict=False,
                                    )
        belief_loss = belief_outputs[0]
        belief_pred = belief_outputs[1]
        belief_pred=torch.argmax(belief_pred, dim=-1)
        num_belief_correct, num_belief_count = self.count_tokens(
            belief_pred, belief_labels, pad_id=self.reader.pad_token_id)

        loss = belief_loss

        step_outputs = {"belief": {"loss": belief_loss.item(),
                                   "correct": num_belief_correct.item(),
                                   "count": num_belief_count.item()}}

        return loss, step_outputs

    def reduce_ddp_stepoutpus(self, step_outputs):
        step_outputs_all = {"belief": {"loss": reduce_mean(step_outputs['belief']['loss']),
                                       "correct": reduce_sum(step_outputs['belief']['correct']),
                                       "count": reduce_sum(step_outputs['belief']['count'])}}

        if self.cfg.add_auxiliary_task:
            step_outputs_all['span'] = {
                'loss': reduce_mean(step_outputs['span']['loss']),
                "correct": reduce_sum(step_outputs['span']['correct']),
                "count": reduce_sum(step_outputs['span']['count'])
            }

        if self.cfg.task == "e2e":
            step_outputs_all["resp"] = {
                'loss': reduce_mean(step_outputs['resp']['loss']),
                "correct": reduce_sum(step_outputs['resp']['correct']),
                "count": reduce_sum(step_outputs['resp']['count'])
            }

        return step_outputs_all

    def train_epoch(self, data_loader, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()
        epoch_step, train_loss = 0, 0.
        for batch in tqdm(data_loader):
            start_time = time.time()
            inputs, belief_labels = batch

            loss, step_outputs = self.step_fn(inputs, belief_labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (epoch_step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)

            epoch_step += 1

        return train_loss

    def train(self):
        train_dataset = MultiWOZDataset(self.reader, self.cfg.s_type, "train",self.cfg.context_size)
        if self.cfg.num_gpus > 1:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = sampler.RandomSampler(train_dataset)
        '''
        ckpt_para = torch.load("./ckpt/test/ckpt-epoch1/pytorch_model.bin",
                                   map_location=torch.device('cpu'))
        if  self.cfg.num_gpus>1:
            self.model.module.load_state_dict(ckpt_para,strict=False)
        else:
            self.model.load_state_dict(ckpt_para, strict=False)
        '''
        train_collator = CollatorTrain(self.reader.pad_token_id, self.reader.tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.cfg.batch_size_per_gpu,
                                      collate_fn=train_collator)

        num_training_steps_per_epoch = len(train_dataloader) // self.cfg.grad_accum_steps
        logger.info("len(train_dataloader) is {}".format(num_training_steps_per_epoch))
        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size_per_gpu)


        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)

        max_score = 0.0
        for epoch in range(1, self.cfg.epochs + 1):
            if self.cfg.num_gpus > 1:
                train_dataloader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_dataloader, optimizer, scheduler, reporter)

            logger.info("done {}/{} epoch, train loss is:{:f}".format(epoch, self.cfg.epochs, train_loss))

            if self.cfg.save_best_model:
                if self.cfg.local_rank == 0:
                    current_score = self.predict()
                    if max_score < current_score:
                        max_score = current_score
                        self.save_model(epoch)
            else:
                if self.cfg.local_rank in [0, -1]:
                    self.save_model(epoch)

            if self.cfg.num_gpus > 1:
                torch.distributed.barrier()

    def validation(self, global_step):
        self.model.eval()

        eval_dataset = MultiWOZDataset(self.reader, 'dev', self.cfg.task, self.cfg.ururu,
                                       context_size=self.cfg.context_size, excluded_domains=self.cfg.excluded_domains)
        eval_sampler = SequentialDistributedSampler(eval_dataset, self.cfg.batch_size_per_gpu_eval)
        eval_collator = CollatorTrain(self.reader.pad_token_id)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.cfg.batch_size_per_gpu_eval,
                                     collate_fn=eval_collator)

        reporter = Reporter(1000000, self.cfg.model_dir)

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Validaction"):
                start_time = time.time()

                inputs, labels = batch

                _, step_outputs = self.step_fn(inputs, *labels)

                torch.distributed.barrier()

                step_outputs_all = self.reduce_ddp_stepoutpus(step_outputs)

                if self.cfg.local_rank == 0:
                    reporter.step(start_time, lr=None, step_outputs=step_outputs_all, is_train=False)

            do_span_stats = True if "span" in step_outputs else False
            do_resp_stats = True if "resp" in step_outputs else False

            reporter.info_stats("dev", global_step, do_span_stats, do_resp_stats)

    def finalize_bspn(self, belief_outputs):
        eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}

            decoded["bspn_gen"] = bspn

            batch_decoded.append(decoded)

        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                # logger.warn("bos/eos action token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                # logger.warn("bos/eos resp token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded

    def predict(self):
        self.model.eval()
        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model
        print(model)
        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size_per_gpu_eval,
             1)

        early_stopping = True if self.cfg.beam_size > 1 else False

        eval_dial_list = None

        results = {}
        evaluator = MultiWozEvaluator(self.reader, self.cfg.t_type,self.cfg.pred_data_type)
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)
            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]
            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context = self.iterator.flatten_dial_history(
                        dial_history[t], len(turn["user"]), self.cfg.context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)

                attention_mask = torch.where(
                    batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                # belief tracking
                with torch.no_grad():
                    #print(self.reader.tokenizer.decode(batch_encoder_input_ids[0]))
                    belief_outputs = model.generate(input_ids=batch_encoder_input_ids,
                                                    attention_mask=attention_mask,
                                                    eos_token_id=self.reader.eos_token_id,
                                                    max_length=200,
                                                    do_sample=self.cfg.do_sample,
                                                    num_beams=self.cfg.beam_size,
                                                    early_stopping=early_stopping,
                                                    temperature=self.cfg.temperature,
                                                    top_k=self.cfg.top_k,
                                                    top_p=self.cfg.top_p,
                                                    )

                belief_outputs = belief_outputs.cpu().numpy().tolist()

                print(self.reader.tokenizer.decode(belief_outputs[0]))
                print(self.reader.tokenizer.decode(turn_batch[0]["bspn"]))

                decoded_belief_outputs = self.finalize_bspn(
                    belief_outputs)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])
                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])

                    if self.cfg.use_true_prev_bspn:
                        pv_bspn = turn["bspn"]
                    else:
                        pv_bspn = turn["bspn_gen"]

                    pv_resp = turn["resp"]



                    pv_text += (pv_bspn + pv_resp)

                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)
            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(
                results, add_auxiliary_task=self.cfg.add_auxiliary_task)

            logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))
        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))



        joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(
                results, add_auxiliary_task=self.cfg.add_auxiliary_task)

        logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))

        for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100

                logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))
        return joint_goal