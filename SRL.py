"""Module for creating heatmaps of the average attention weight for each attention head."""

import torch
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from preprocessing import Preprocess
from prettytable import PrettyTable


def heatmap(data, row_labels, col_labels, ax=None, x_label='x', y_label='y',
            cbar_kw=None, cbarlabel="", vmin=0.0, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Stolen from some pyplot tutorial somewhere on the web.
    
    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    ax.set_xlabel(x_label)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(y_label, rotation=90)
    # Plot the heatmap
    im = ax.imshow(data, vmin=vmin, **kwargs)

    im.cmap.set_under('white')
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, extend='min')
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def plot_attn(weights, row_labels, col_labels, x_label, y_label):
    """Plot heatmaps of attention weights."""
    
    fig, ax = plt.subplots()
    im, cbar = heatmap(data=weights, row_labels=row_labels,
                       col_labels=col_labels, x_label=x_label, 
                       y_label=y_label, ax=ax, cmap="YlGn", cbarlabel="Attention weight")
    im.set
    fig.tight_layout()
    plt.show()


def save_attn(weights, row_labels, col_labels, x_label, y_label, fname):
    """Save heatmaps of attention weights."""
    
    fig, ax = plt.subplots()
    im, cbar = heatmap(data=weights, row_labels=row_labels,
                       col_labels=col_labels, x_label=x_label, 
                       y_label=y_label, ax=ax, cmap="YlGn", cbarlabel="Attention weight")
    fig.tight_layout()
    plt.savefig(f'{fname}.png')
    plt.close()


def get_data(data_path, tokenizer, pad_size, label_dict, idx2labels, pad=False):
    preprocess = Preprocess(path=data_path,
                            tokenizer=tokenizer,
                            pad_size=pad_size,
                            label2idx=label_dict,
                            idx2labels=idx2labels,
                            pad=pad)

    preprocess()

    data = {'x': preprocess.data[f'tokens'],
            'y': preprocess.data[f'labels']}
    return data


def find_links(data):

    bad_labels = {'[CLS]', '[SEP]', 'O', 'X'}

    links = defaultdict(list)
    for i, y in enumerate(data['y']):
        preds, args = {}, defaultdict(dict)

        for j, labels in enumerate(y):
            
            for k, lbl in enumerate(labels):
                
                if lbl not in bad_labels:
                
                    if '_' not in lbl:
                        preds[lbl] = j
                    
                    elif '_' in lbl:
                        args[lbl[:2]][lbl] = j
        
        for pred in args:
            pred_idx = preds[pred]
            
            for arg_idx in args[pred].values():
                if abs(pred_idx - arg_idx) > 1:
                    links[i].append((pred_idx, arg_idx))

    return links


class Attentions:
    """Class for extracting the attention maps of a given sequence."""
    
    def __init__(self, model, tokenizer):
        """Inits class Attentions:
        
        Fields:
            model: A pre-trained BERT model.
            tokenizer: A pre-trained BERT tokenizer.
            attention_weights: A defaultdict with lists of the attention
            weights of coreference pairs."""
            
        self.model = model.cuda()
        self.tokenizer = tokenizer
        self.attention_weights = defaultdict(list)
        
    def extract_attention_weights(self,
                                  inp,
                                  links,
                                  n_layers=12,
                                  n_heads=12,
                                  return_head=None):
        
        """Extracts the max attention weight of the coreference pairs 
        for every head in the model.
        
        Saves weight in dict Attentions.attention_weights.
        Weights must be averaged after with Attentions.average_weights.
        
        Args:
            inp: A BERT batch encoding of a sequence.
            pron: The position of the pronoun in the tokenized sequence.
            chars: The position of the characters in the tokenized sequence."""
        
        model_output = self.model(inp['input_ids'].cuda(),
                                  inp['attention_mask'].cuda(),
                                  output_attentions=True)
        
        attns = model_output.attentions

        if return_head:
            return self._get_attention_head(attns,
                                            n_layer=return_head[0],
                                            n_head=return_head[1]).cpu().numpy()

        for l in range(1, n_layers+1):
            for h in range(1, n_heads+1):
                attn_head = self._get_attention_head(attns,
                                                    n_layer=l,
                                                    n_head=h)
                
                #attn_head = self._convert_to_word_attention(attn_head,
                #                                            inp)
                
                weight = self._get_max_attn_weight(attn_head, links)
                
                self.attention_weights[f'l{l}h{h}'].append(weight)
        
    def _get_attention_head(self, attentions, n_layer, n_head):
        """Gets the attentions of one head.
        
        Argument attentions is a tuple of length n layers in model.
        Each element in the tuple is the attention weights of the input,
        for that layer, after softmax, but before the value weighting.
        Each layer has the shape n_batches * n_heads * seq * seq.
        
        Args:
            attentions: A tuple with the attention heads in the model.
            n_layer: The current layer.
            n_head: The current head in the layer."""
        
        # squeeze to get rid of the batch dimension.
        layer_attention = attentions[n_layer-1].squeeze()
        attention_head = layer_attention[n_head-1]
        return attention_head
        
    def _convert_to_word_attention(self, attn, inp):
        """Converts subword level attention maps to word level attention maps.
        
        Follows the method in Clark et al. (2019) for pooling the subword level
        attention to word level: sums all subwords' attention weights for 
        attention *to* a split word; takes the mean of the subwords' attention
        weights *from* a split word.
        
        Args:
            attn: A pytorch tensor of an attention matrix for a head.
            inp: The batch encoding of a sequence.
        Returns:
            attn: A numpy array of the word level attention matrix."""
        
        def get_subword_indices(inp):
            """Helper function for converting subwords to words.
            
            Args:
                inp: The batch encoding of a sequence.
            Returns:
                splitword_indices: A dict with the index of the first subtoken
                of a subtokenized word as key and the indices of all subtokens
                in the word.
                subword_indices: A list with all subtokens in the sequence
                except for the first subtoken of every word."""
            
            splitword_indices = {}
            subword_indices = []
            for id in set(inp.word_ids()[1:-1]):
                start, end = inp.word_to_tokens(id)
                if start != end - 1:
                    splitword_indices[start] = list(range(start, end))
                    subword_indices.extend(list(range(start+1, end)))
            return splitword_indices, subword_indices
        
        def sum_subwords(attn, indices, subword_indices):
            """Sums the attention weights to a split word."""
            
            for word, subwords in indices.items():
                attn[:, word] = np.sum(attn[:, subwords], axis=1)
            attn = np.delete(attn, subword_indices, axis=1)
            return attn
        
        def mean_subwords(attn, indices, subword_indices):
            """Takes the mean of the attention weights from a split word."""
            
            for word, subwords in indices.items():
                attn[word, :] = np.mean(attn[subwords, :], axis=0)
            attn = np.delete(attn, subword_indices, axis=0)
            return attn
        
        attn = attn.cpu().numpy()
        splitword_indices, subword_indices = get_subword_indices(inp)
        
        if splitword_indices:
            attn = sum_subwords(attn, splitword_indices, subword_indices)
            attn = mean_subwords(attn, splitword_indices, subword_indices)
            
        return attn

    def _get_max_attn_weight(self, attn, links):
        """Returns the index of the max weight of all coreference pairs in the
        sequence."""
        
        return max([attn[arg, pred] for pred, arg in links])

    def average_heads(self):
        """Averages the attention weights for all heads in the defaultdict
        Attentions.attention_weights."""
        
        for head, weights in self.attention_weights.items():
            self.attention_weights[head] = sum(weights) / len(weights)
            
            
if __name__ == '__main__':
    lang = 'Chinese'
    part = 2
    
    path = f'/home/hantan/Kurser/Uppsats/Data/CoNLL-2009/2009_conll_p{part}/data/CoNLL2009-ST-{lang}/CoNLL2009-ST-{lang}-development.txt'    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    data = get_data(path, tokenizer, 1, None, None, pad=False)
    links = find_links(data)

    # Extract the attention weights for all valid coreference pairs in the dataset.
    with torch.no_grad():
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        attentions = Attentions(model, tokenizer)

        for i in tqdm(range(len(data['x'][:10000])), desc='Sentence'):
            if links[i]:
                attentions.extract_attention_weights(data['x'][i], links[i])

        # Average the attention weights for the heads.
        attentions.average_heads()

    # Put the average attention weights in a numpy array.
    attn = np.zeros((12,12))
    for l in range(12):
        for h in range(12):
            attn[l, h] = attentions.attention_weights[f'l{l+1}h{h+1}']

    layer_attn = np.zeros(12)
    for l in range(12):
        layer_attn[l] = round(np.sum(attn[l, :]), 4)

    table = PrettyTable([f'Layer {i}' for i in range(1,13)])
    table.add_row(layer_attn)
    print(table)
    
    # Save heatmap
    save_attn(attn, [i for i in range(1,13)],
        [i for i in range(1,13)], x_label='Head', y_label='Layer', fname=f'Heatmaps/{lang}')
    
    