from collections import defaultdict
import glob
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from torch.utils.data import Dataset
import torch


class Data(Dataset):
    
    def __init__(self,
                 x: list[dict],
                 y: list[list[list[None, str]]],
                 transform=None) -> None:
        self.x, self.y = x, y
        self.transform  = transform
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'x': self.x[idx], 'y': self.y[idx]}
            
        if self.transform:
            sample = self.transform(sample)
            
        if torch.cuda.is_available():
            sample['x']['input_ids'] = sample['x']['input_ids'].cuda()
            sample['x']['attention_mask'] = sample['x']['attention_mask'].cuda()
            sample['y'] = sample['y'].cuda()
            
        return sample
    
    
class EncodeLabels:
    
    def __init__(self,
                 label2idx: defaultdict,
                 n_labels: int,
                 dtype=torch.int64,
                 special_labels={'[CLS]', '[SEP]', '[PAD]', 'X'}):
        
        print(label2idx)
        
        self.label2idx = label2idx
        self.dtype     = dtype
        self.special_labels = special_labels
        self.n_labels  = n_labels
        
    def __call__(self,
                 sample: dict):
        labels = sample['y']
        n_tokens = len(labels)
        label_tens = torch.zeros((n_tokens, self.n_labels - len(self.special_labels)))

        for n, label in enumerate(labels):
            for lbl in label:
                if lbl not in self.special_labels:
                    label_tens[n, self.label2idx[lbl]] = 1

        sample['y'] = label_tens

        return sample


class FormatLabel:

    def __init__(self, only_numbered_args=True):
        self.only_numbered_args = only_numbered_args
        #self.only_numbered_args = False

    def __call__(self, label, lang):
        
        if label == '_':
            label = None
        else:
            label = self._translate_label(label, lang)

            if self.only_numbered_args:
                if self._AM_label(label) or self._C_R_label(label):
                    label = None
        
        return label
    
    def _chinese_special_label(self, label):
        chinese_special_labels = {'TMP', 'DIS', 'ADV', 'BNF',
                                  'EXT', 'QTY', 'ASP', 'T',
                                  'PN', 'PSE', 'DIR', 'LOC',
                                  'PRP', 'MNR', 'TPC', 'CND',
                                  'CRD', 'FRQ', 'DGR', 'VOC',
                                  'PRD', 'PSR'}
        return label in chinese_special_labels
    
    def _AM_label(self, label):
        return 'AM' in label or 'AL' in label
    
    def _C_R_label(self, label):
        return 'C' in label or 'R' in label
    
    def _translate_label(self,
                         label: str,
                         lang: str) -> str:

        if lang in {'ca', 'es'}:
            if label[3].isnumeric():
                label = 'A' + label[3]
            elif label[3] in {'M', 'L'}:
                label = 'A' + label[3] + label[4:].upper()

        elif lang == 'zh':
            if self._chinese_special_label(label):
                label = 'AM-' + label

        return label
    

class Frames:

    def __init__(self, path):
        
        self.lang = self._get_lang(path)

        paths = self._get_paths(path, self.lang)
                
        if self.lang in {'de', 'en'}:
            self.lemmas = self._get_lemmas_from_logical_xml_files(paths)
        elif self.lang in {'es', 'ca'}:
            self.lemmas = self._get_lemmas_from_annoying_txt_files(paths)
        elif self.lang == 'zh':
            self.lemmas = self._get_lemmas_from_illogical_xml_files(paths)
        elif self.lang == 'cz':
            self.lemmas = self._get_lemmas_from_a_weird_file(f'{path}/Czech.vallex')

    def __contains__(self, x):
        return x in self.lemmas

    def _get_lang(self, path):
        if 'Catalan' in path:
            lang = 'ca'
        elif 'Czech' in path:
            lang = 'cz'
        elif 'German' in path:
            lang = 'de'
        elif 'Spanish' in path:
            lang = 'es'
        elif 'Chinese' in path:
            lang = 'zh'
        elif 'English' in path:
            lang = 'en'
        return lang    

    def _get_paths(self, path, lang):
        stub = ''.join(reversed(path)).find('/')
        
        if lang == 'en':
            dir = 'pb_frames'
            file_type = 'xml'
        elif lang in {'de', 'zh'}:
            dir = 'frames'
            file_type = 'xml'
        elif lang in {'es', 'ca'}:
            dir = 'entries'
            file_type = 'txt'
                
        return glob.glob(f'{path[:-stub-1]}/{dir}/*.{file_type}',
                         recursive=True)

    def _get_lemmas_from_logical_xml_files(self, paths):
        return set([ET.parse(path).getroot().find('predicate').attrib['lemma']
                    for path in paths])
    
    def _get_lemmas_from_illogical_xml_files(self, paths):
        
        errors = 0
        parsed = []
        for path in paths:
            
            try:
                parsed.append(ET.parse(path).getroot().find('id').text.strip())
            except ParseError:
                errors += 1
        
        print(f'N errors: {errors}')
        
        return set(parsed)
        
    def _get_lemmas_from_annoying_txt_files(self, paths):
        lemmas = set()
        for path in paths:
            with open(path, 'r', encoding='utf-8') as fhand:
                line = fhand.readline()
                lemmas.add(line[:line.find(' ')])
        return lemmas
    
    def _get_lemmas_from_a_weird_file(self, path):
        with open(path, 'r', encoding='utf-8') as fhand:
            lemmas = [line[:line.find(' ')] for line in fhand]
        return set(lemmas)