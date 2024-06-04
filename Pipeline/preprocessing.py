from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizerFast
from utils import FormatLabel, Frames


class Preprocess:
    
    def __init__(self,
                 path: str,
                 tokenizer: BertTokenizerFast,
                 pad_size: int,
                 out_label='O',
                 subword_label='X',
                 label2idx=None,
                 idx2labels=None,
                 pad=False,
                 only_numbered_args=True):
        
        """Preprocesses and loads the data of a CoNLL 2009-file.
        
        The data is in field data: a dict with the tokens and labels for the
        given language. Keys are of the format '{lang}_token' or
        '{lang}_label'."""

        self.path           = path
        self.frames         = Frames(path)
        self.format_label   = FormatLabel(only_numbered_args)
        self.lang           = self.frames.lang
        self.tokenizer      = tokenizer
        self.data           = defaultdict(list)
        self.label2idx      = defaultdict(int) if not label2idx else label2idx
        self.idx2label      = dict() if not idx2labels else idx2labels
        self.out_label      = out_label
        self.subword_label  = subword_label
        self.pad_size       = pad_size
        self.pad            = pad
        
        if not label2idx:
            self._init_labels()
        
        # Helps with indexing of labels.
        self._current_lbl_idx = max(self.label2idx.values()) + 1

    def __call__(self):
        """Preprocesses the data.
        
        For each sentence, gathers the tokens and the labels.
        The labels are p[predicate_i]_[arg_i] for predicates (frames) and
        [arg_i] for the arguments (semantic roles),
        and O for tokens that are neither predicates or arguments
        and X for subtokens.
        """
        
        def end_of_sentence():
            return line == '\n'

        sentence        = defaultdict(dict)
        predicates      = defaultdict(dict)
        correct_pred_i  = 0
        predicate_i     = -1
        
        with open(self.path, 'r', encoding='utf-8') as fhand:
            for line in tqdm(fhand, desc='Line'):
                
                if end_of_sentence():
                    
                    sentence_with_labels = self._get_labels(sentence, predicates)
                    encoded_sentence, labels = self._tokenize(sentence_with_labels)
                    
                    self.data[f'tokens'].append(encoded_sentence)
                    self.data[f'labels'].append(labels)
                    
                    sentence        = defaultdict(dict)
                    predicates      = defaultdict(dict)
                    correct_pred_i  = 0
                    predicate_i     = -1
                     
                else:
                    line = line.rstrip('\n').split()
                    
                    id              = line[0]
                    token           = line[1]
                    lemma           = line[2]
                    pos             = line[4]
                    is_predicate    = line[12]
                    unparsed_args   = line[14:]
                    
                    sentence[id]['token'] = token
                    sentence[id]['args']  = unparsed_args
                    
                    # Keep track of the total predicates
                    if is_predicate == 'Y':
                        predicate_i += 1
                        
                    if self._is_correct_predicate(lemma=lemma,
                                                  is_predicate=is_predicate,
                                                  pos=pos
                                                  ):
                        
                        # Keep track of the indices of total predicates and of
                        # correct predicates.
                        # One dict sorted after i predicate and id.
                        predicates['predicate_i'][predicate_i] = correct_pred_i
                        predicates['id'][id]                   = correct_pred_i
                        
                        correct_pred_i += 1
                        
    def _init_labels(self):
        self.label2idx['[PAD]']            = -1
        self.label2idx['[CLS]']            = -1
        self.label2idx['[SEP]']            = -1
        self.label2idx[self.subword_label] = -1
        self.label2idx[self.out_label]     = 0
            
    def _is_correct_predicate(self,
                              lemma: str,
                              is_predicate: str,
                              pos: str):
        
        """Checks if the current token is frame of the correct type.
        Uses lemmas and part-of-speech tags to determine if a frame is correct.
        """
        
        pos_tags = set([
                        'v',  # Catalan and Spanish
                        'VC', 'VE', 'VA', 'VV',  # Chinese
                        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # English
                        # German
                        'VVFIN', 'VVIMP', 'VVINF', 'VVIZU',
                        'VVPP', 'VAFIN', 'VAIMP', 'VAINF',
                        'VAPP', 'VMFIN', 'VMINF', 'VMPP'
                        ])
        
        return is_predicate == 'Y' and pos in pos_tags and lemma in self.frames
            
    def _get_labels(self,
                    sentence: defaultdict,
                    predicates: defaultdict):

        def get_pred_label():
            if predicates['id'].get(id, None) is not None:
                
                if predicates['id'][id] < 5 or 'development' in self.path:
                     labels.append(f"p{predicates['id'][id]}")
                # labels.append(f"p{predicates['id'][id]}")

        def get_arg_labels():
            for i, arg in enumerate(sentence[id]['args']):
                
                label = self.format_label(arg, self.lang)
                
                if predicates['predicate_i'].get(i, None) is not None \
                   and label is not None:

                    p = predicates['predicate_i'][i]
                    
                    if p > 5:
                        breakpoint
                    
                    if p < 5 or 'development' in self.path:
                        labels.append(f"p{p}_{label}")
                    # labels.append(f"p{p}_{label}")

        for id in sentence:
            labels = []

            get_pred_label()
            
            get_arg_labels()
            
            sentence[id]['labels'] = labels if len(labels) > 0 \
                                            else [self.out_label]
            
        return sentence
    
    def _index_labels(self, 
                      labels: list):
        """Maps each unique label to a unique index."""

        for label in labels:
            if label not in self.label2idx:

                self.label2idx[label] = self._current_lbl_idx

                self._current_lbl_idx += 1
                
            self._label_indices(label, self.label2idx[label])
                
    def _label_indices(self,
                       label: str,
                       idx: int):
        
        # Map CLS-args
        if '[CLS]' in label:
            self.idx2label[idx] = '[CLS]'
        
        # Map SEP-args:
        elif '[SEP]' in label:
            self.idx2label[idx] = '[SEP]'
        
        # Map O
        elif 'O' in label:
            self.idx2label[idx] = 'O'
        
        # Map X
        elif 'X' in label:
            self.idx2label[idx] = 'X'
        
        # Map AM-args
        elif 'AM' in label:
            self.idx2label[idx] = 'AM'
        
        # Map predicates
        elif '_' not in label:
            self.idx2label[idx] = label
            
        # Map numbered args
        elif label[-1].isdigit() and 'C' not in label and 'R' not in label:
            self.idx2label[idx] = label[label.find('_')+1:]
        
        # Map C-args
        elif 'C' in label:
            self.idx2label[idx] = 'C'
            
        # Map R-args
        elif 'R' in label:
            self.idx2label[idx] = 'R'
        
        else:
            self.idx2label[idx] = label[label.find('_')+1:]
            breakpoint

    def _tokenize(self,
                  sentence: defaultdict):
        encoded = self.tokenizer([sentence[id]['token'] for id in sentence],
                                 is_split_into_words=True,
                                 padding=self.pad,
                                 max_length=self.pad_size,
                                 truncation=True,
                                 return_tensors='pt')
        
        labels = []
        previous_word_id = -1
        
        for word_id in encoded.word_ids():
            
            if word_id is None:
                if previous_word_id == -1:
                    labels.append(['[CLS]'])
                elif previous_word_id is not None:
                    labels.append(['[SEP]'])
                else:
                    labels.append(['[PAD]'])
            
            elif word_id == previous_word_id:
                labels.append([self.subword_label])
                
            else:
                self._index_labels(sentence[str(word_id+1)]['labels'])
                labels.append(sentence[str(word_id+1)]['labels'])

            previous_word_id = word_id
            
        return encoded, labels
