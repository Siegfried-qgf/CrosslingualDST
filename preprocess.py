'''
preprocess Multi2Woz
'''
import os
import re
import copy
import argparse
from collections import OrderedDict
import spacy
from tqdm import tqdm
from utils import definitions
from utils.io_utils import load_json, save_json, load_text
from utils.clean_dataset import clean_text, clean_slot_values


class Preprocessor(object):
    def __init__(self, l_type):
        self.nlp = spacy.load("en_core_web_sm")
        self.l_type=l_type
        self.data_dir=os.path.join("dataset/"+l_type)
        self.save_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(self.save_dir, exist_ok=True)

        if l_type=="en":
            data_name = "data.json"
            self.dev_list = load_text(os.path.join(self.data_dir, "valListFile.txt"))
            self.test_list = load_text(os.path.join(self.data_dir, "testListFile.txt"))
            self.do_tokenize_text = True
            self.data = load_json(os.path.join(self.data_dir, data_name))
        else:
            test_name="test_full_{}.json".format(l_type)
            dev_name="val_full_{}.json".format(l_type)
            self.test_data = load_json(os.path.join(self.data_dir, test_name))
            self.dev_data = load_json(os.path.join(self.data_dir, dev_name))

        self.mapping_pair = self.load_mapping_pair()

    def load_mapping_pair(self):
        mapping_pair = []
        curr_dir = os.path.dirname(__file__)
        with open(os.path.join(curr_dir, './utils/mapping.pair'), 'r') as fin:
            for line in fin.readlines():
                fromx, tox = line.replace('\n', '').split('\t')
                mapping_pair.append((fromx, tox))
        return mapping_pair

    def preprocess(self):
        if self.l_type=="en":
            train_data, dev_data, test_data = {}, {}, {}
            count = 0
            self.unique_da = {}
            ordered_sysact_dict = {}
            for fn, raw_dial in tqdm(list(self.data.items())):
                if ".json" not in fn:
                    fn += ".json"
                if fn in ['pmul4707.json', 'pmul2245.json', 'pmul4776.json', 'pmul3872.json', 'pmul4859.json']:
                    continue
                count += 1
                #if count==10:
                #    break

                dial_domains, dial_reqs = [], []
                for dom, g in raw_dial['goal'].items():
                    if dom != 'topic' and dom != 'message' and g:
                        if g.get('reqt'):  # request info. eg. postcode/address/phone
                            # normalize request slots
                            for i, req_slot in enumerate(g['reqt']):
                                if definitions.NORMALIZE_SLOT_NAMES.get(req_slot):
                                    g['reqt'][i] = definitions.NORMALIZE_SLOT_NAMES[req_slot]
                                    dial_reqs.append(g['reqt'][i])
                        if dom in definitions.ALL_DOMAINS:
                            dial_domains.append(dom)

                dial_reqs = list(set(dial_reqs))

                dial = {'log': []}
                single_turn = {}
                constraint_dict = OrderedDict()
                prev_constraint_dict = {}
                prev_turn_domain = ['general']
                ordered_sysact_dict[fn] = {}

                for turn_num, dial_turn in enumerate(raw_dial['log']):
                    # for user turn, have text
                    # sys turn: text, belief states(metadata), dialog_act, span_info
                    dial_state = dial_turn['metadata']

                    if self.do_tokenize_text:
                        dial_turn['text'] = ' '.join([t.text for t in self.nlp(dial_turn['text'])])

                        dial_turn["text"] = ' '.join(
                            dial_turn["text"].replace(".", " . ").split())

                    if not dial_state:   # user
                        u = ' '.join(clean_text(dial_turn['text']).split())
                        single_turn['user'] = u

                    else:   # system
                        single_turn['resp'] = ' '.join(clean_text(dial_turn['text']).split())
                        for domain in dial_domains:
                            if not constraint_dict.get(domain):
                                constraint_dict[domain] = OrderedDict()
                            info_sv = dial_state[domain]['semi']
                            for s, v in info_sv.items():
                                s, v = clean_slot_values(domain, s, v)
                                if len(v.split()) > 1:
                                    v = ' '.join(
                                        [token.text for token in self.nlp(v)]).strip()
                                if v != '':
                                    constraint_dict[domain][s] = v
                            book_sv = dial_state[domain]['book']
                            for s, v in book_sv.items():
                                if s == 'booked':
                                    continue
                                s, v = clean_slot_values(domain, s, v)
                                if len(v.split()) > 1:
                                    v = ' '.join(
                                        [token.text for token in self.nlp(v)]).strip()
                                if v != '':
                                    constraint_dict[domain][s] = v

                        constraints = []  # list in format of [domain] slot value
                        cons_delex = []
                        turn_dom_bs = []
                        for domain, info_slots in constraint_dict.items():
                            if info_slots:
                                constraints.append('['+domain+']')
                                cons_delex.append('['+domain+']')
                                for slot, value in info_slots.items():
                                    constraints.append('[value_' + slot + ']')
                                    constraints.extend(value.split())
                                    cons_delex.append('[value_' + slot + ']')
                                if domain not in prev_constraint_dict:
                                    turn_dom_bs.append(domain)
                                elif prev_constraint_dict[domain] != constraint_dict[domain]:
                                    turn_dom_bs.append(domain)


                        turn_dom_da = set()
                        # get turn domain
                        turn_domain = turn_dom_bs
                        for dom in turn_dom_da:
                            if dom != 'booking' and dom not in turn_domain:
                                turn_domain.append(dom)
                        if not turn_domain:
                            turn_domain = prev_turn_domain
                        if len(turn_domain) == 2 and 'general' in turn_domain:
                            turn_domain.remove('general')
                        if len(turn_domain) == 2:
                            if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                                turn_domain = turn_domain[::-1]



                        single_turn['constraint'] = ' '.join(constraints)
                        single_turn['cons_delex'] = ' '.join(cons_delex)
                        single_turn['turn_num'] = len(dial['log'])
                        single_turn['turn_domain'] = ' '.join(
                            ['['+d+']' for d in turn_domain])

                        prev_turn_domain = copy.deepcopy(turn_domain)
                        prev_constraint_dict = copy.deepcopy(constraint_dict)

                        if 'user' in single_turn:
                            dial['log'].append(single_turn)

                        single_turn = {}

                if fn in self.dev_list:
                    dev_data[fn] = dial
                elif fn in self.test_list:
                    test_data[fn] = dial
                else:
                    train_data[fn] = dial

            print("Save preprocessed data to {} (#train: {}, #dev: {}, #test: {})"
                    .format(self.save_dir, len(train_data), len(dev_data), len(test_data)))

            save_json(train_data, os.path.join(self.save_dir, "train_data.json"))
            save_json(dev_data, os.path.join(self.save_dir, "dev_data.json"))
            save_json(test_data, os.path.join(self.save_dir, "test_data.json"))
        else:
            datalist=[self.dev_data,self.test_data]
            for num,dataset in enumerate(datalist):
                count=0
                dataout = {}
                for fn, raw_dial in tqdm(dataset.items()):
                    ordered_sysact_dict = {}
                    if ".json" not in fn:
                        fn += ".json"
                    count+=1
                    #if count==10:
                    #    break

                    dial = {'log': []}
                    single_turn = {}
                    constraint_dict = OrderedDict()
                    prev_constraint_dict = {}
                    prev_turn_domain = ['general']
                    ordered_sysact_dict[fn] = {}
                    log_name="log-"+self.l_type
                    for turn_num, dial_turn in enumerate(raw_dial[log_name]):
                        # for user turn, have text
                        # sys turn: text, belief states(metadata), dialog_act, span_info
                        dial_state = dial_turn['metadata']

                        if not dial_state:  # user
                            u = ' '.join(clean_text(dial_turn['text']).split())
                            single_turn['user'] = u

                        else:  # system
                            single_turn['resp'] = ' '.join(clean_text(dial_turn['text']).split())
                            dial_domains=self.get_dial_domain(dial_state)
                            for domain in dial_domains:
                                if not constraint_dict.get(domain):
                                    constraint_dict[domain] = OrderedDict()
                                info_sv = dial_state[domain]['semi']
                                for s, v in info_sv.items():
                                    s, v = clean_slot_values(domain, s, v)
                                    if v != '':
                                        constraint_dict[domain][s] = v
                                book_sv = dial_state[domain]['book']
                                for s, v in book_sv.items():
                                    if s == 'booked':
                                        continue
                                    s, v = clean_slot_values(domain, s, v)
                                    if v != '':
                                        constraint_dict[domain][s] = v

                            constraints = []  # list in format of [domain] slot value
                            cons_delex = []
                            turn_dom_bs = []
                            for domain, info_slots in constraint_dict.items():
                                if info_slots:
                                    constraints.append('[' + domain + ']')
                                    cons_delex.append('[' + domain + ']')
                                    for slot, value in info_slots.items():
                                        constraints.append('[value_' + slot + ']')
                                        constraints.extend(value.split())
                                        cons_delex.append('[value_' + slot + ']')
                                    if domain not in prev_constraint_dict:
                                        turn_dom_bs.append(domain)
                                    elif prev_constraint_dict[domain] != constraint_dict[domain]:
                                        turn_dom_bs.append(domain)

                            turn_dom_da = set()
                            # get turn domain
                            turn_domain = turn_dom_bs
                            for dom in turn_dom_da:
                                if dom != 'booking' and dom not in turn_domain:
                                    turn_domain.append(dom)
                            if not turn_domain:
                                turn_domain = prev_turn_domain
                            if len(turn_domain) == 2 and 'general' in turn_domain:
                                turn_domain.remove('general')
                            if len(turn_domain) == 2:
                                if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                                    turn_domain = turn_domain[::-1]

                            # get system action

                            single_turn['constraint'] = ' '.join(constraints)
                            single_turn['cons_delex'] = ' '.join(cons_delex)
                            single_turn['turn_num'] = len(dial['log'])
                            single_turn['turn_domain'] = ' '.join(
                                ['[' + d + ']' for d in turn_domain])

                            prev_turn_domain = copy.deepcopy(turn_domain)
                            prev_constraint_dict = copy.deepcopy(constraint_dict)

                            if 'user' in single_turn:
                                dial['log'].append(single_turn)

                            single_turn = {}

                    dataout[fn] = dial
                if num==0:
                    print("Save preprocessed {} data to {} "
                          .format("val_"+self.l_type ,self.save_dir))
                    save_json(dataout, os.path.join(self.save_dir, "dev_data_{}.json".format(self.l_type)))
                else:
                    print("Save preprocessed {} data to {} "
                          .format("test_" + self.l_type, self.save_dir))
                    save_json(dataout, os.path.join(self.save_dir, "test_data_{}.json".format(self.l_type)))

    def get_dial_domain(self,dict):
        domainlist=[]
        for domain,context in dict.items():
            for type,t in context.items():
                for subtype,subt in t.items():
                    if(subt):
                        domainlist.append(domain)
                        break
                else:
                    continue
                break
        return domainlist

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Argument for preprocessing")
    #parser.add_argument("-l_type", type=str, default="en", choices=["en", "ar","de","ru","zh"])
    #args = parser.parse_args()
    '''
    preprocessor = Preprocessor("cn")
    preprocessor.preprocess()

    preprocessor = Preprocessor("ar")
    preprocessor.preprocess()

    preprocessor = Preprocessor("de")
    preprocessor.preprocess()

    preprocessor = Preprocessor("ru")
    preprocessor.preprocess()
    
    preprocessor = Preprocessor("en")
    preprocessor.preprocess()
    '''
    preprocessor = Preprocessor("de")
    preprocessor.preprocess()