"""Module for running the pipeline (preprocessing, training, evaluation)."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from transformers import BertTokenizerFast, BertModel
from model import Enlabeler
from preprocessing import Preprocess
from utils import Data, EncodeLabels
from modeling import Modeling
from prettytable import PrettyTable
import random
from pathlib import Path
import os
import csv


@torch.autocast(device_type='cuda', dtype=torch.float16)
def get_data(data_path: str,
             pad_size: int,
             pad: str,
             tokenizer,
             label_dict=None,
             idx2labels=None):
    
    """Reads the data from file and loads it to a Dataloader object.

    Returns:
        dataloader: A Dataloader object with the data.
        len(dataset): The total number of examples in the data."""

    preprocess = Preprocess(path=data_path,
                            tokenizer=tokenizer,
                            pad_size=pad_size,
                            label2idx=label_dict,
                            idx2labels=idx2labels,
                            pad=pad)

    preprocess()

    data = {'x': preprocess.data[f'tokens'],
            'y': preprocess.data[f'labels']}

    return data, preprocess.label2idx, preprocess.idx2label


def load_data(data, label_dict, batch_size, data_amount):
    
    random.shuffle(list(data['x']) if isinstance(data['x'], tuple) else data['x'])
    random.shuffle(list(data['y']) if isinstance(data['y'], tuple) else data['y'])
    
    data['x'] = data['x'][:int(len(data['x']) * data_amount)]
    data['y'] = data['y'][:int(len(data['y']) * data_amount)]
        
    dataset = Data(x=data['x'],
                   y=data['y'],
                   transform=transforms.Compose(
                   [EncodeLabels(label2idx=label_dict,
                                 n_labels=len(label_dict))]))
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True)
    
    return dataset, dataloader


def write_results_to_file(args, table, tot_p, tot_r, tot_f, lbl_predictions, final_epoch):
    Path(f'Results/{args.lang}/{args.data_amount}/{args.freeze_layer}').mkdir(parents=True, exist_ok=True)
    
    results_path = f'Results/{args.lang}/{args.data_amount}/{args.freeze_layer}/{args.save_dir}-epochs_{final_epoch+1}-batchsize_{args.batch_size}-lr_{args.lr}-dropout_{args.dropout}-wd_{args.weight_decay}.txt'
    predictions_path = f'Results/{args.lang}/{args.data_amount}/{args.freeze_layer}/predictions-epochs_{final_epoch+1}-batchsize_{args.batch_size}-lr_{args.lr}-dropout_{args.dropout}-wd_{args.weight_decay}.csv'

    with open(results_path, 'w', encoding='utf-8') as fhand:
        fhand.write(f'Train path: {args.train_path}\n')
        fhand.write(f'Test path: {args.test_path}\n')
        fhand.write(f'Learning rate: {args.lr}\n')
        fhand.write(f'Dropout: {args.dropout}\n')
        fhand.write(f'WD: {args.weight_decay}\n')
        fhand.write(f'Batch size: {args.batch_size}\n')
        fhand.write(f'Num epochs: {args.epochs}\n')
        fhand.write(f'Early stopping: {args.early_stop}\n')
        fhand.write(f'Language: {args.lang}\n')
        fhand.write(f'Freeze layer: {args.freeze_layer}\n')
        fhand.write(f'Model name: {args.pt_model_name}\n')
        fhand.write(f'Patience: {args.patience}\n')
        fhand.write(f'Data_amount: {args.data_amount}\n')
        fhand.write('\n--------------------------------------\n\n')
        fhand.write(f'GLOBAL PRECISION: {round(tot_p, 2)}\n')
        fhand.write(f'GLOBAL RECALL: {round(tot_r, 2)}\n')
        fhand.write(f'GLOBAL FSCORE: {round(tot_f, 2)}\n')
        fhand.write(table.get_string())
        
    with open(predictions_path, 'w', encoding='utf-8') as fhand:
        writer = csv.writer(fhand, delimiter='\t')
        writer.writerow(['Token', 'Correct', 'Predicted'])
        for sentence in lbl_predictions:
            for token in sentence:
                writer.writerow(token)
            fhand.write('\n')

        
def print_results(lbl_p, lbl_r, lbl_f, tot_p, tot_r, tot_f, args):
    print('\n\n----------------------------- MODEL INFO ---------------------------')
    print_info(args)
    print('\n\n----------------------------- RESULTS -----------------------------')

    print(f'GLOBAL PRECISION: {round(tot_p, 2)}')
    print(f'GLOBAL RECALL: {round(tot_r, 2)}')
    print(f'GLOBAL FSCORE: {round(tot_f, 2)}')

    print(f'\nLABEL-WISE SCORES:')

    table = PrettyTable(field_names=['Value'] + list(lbl_p.keys()))
    
    table.add_row(['Precision'] + [round(val, 2) for val in lbl_p.values()])
    table.add_row(['Recall']    + [round(val, 2) for val in lbl_r.values()])
    table.add_row(['F-score']   + [round(val, 2) for val in lbl_f.values()])
    
    print(table)
    
    return table
    
    
def print_info(args):
    print(f'Train path: {args.train_path}')
    print(f'Test path: {args.test_path}')
    print(f'Learning rate: {args.lr}')
    print(f'Dropout: {args.dropout}')
    print(f'WD: {args.weight_decay}')
    print(f'Batch size: {args.batch_size}')
    print(f'Num epochs: {args.epochs}')
    print(f'Early stopping: {args.early_stop}')
    print(f'Language: {args.lang}')
    print(f'Freezing layer: {args.freeze_layer}')
    print(f'Model name: {args.pt_model_name}')
    print(f'Patience: {args.patience}')
    print(f'Data_amount: {args.data_amount}')
    print(f'Save_dir: {args.save_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_path', default='Data/CoNLL-2009/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt')
    ap.add_argument('--test_path', default='Data/CoNLL-2009/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.txt')
    ap.add_argument('--train', default=True, type=bool)
    ap.add_argument('--eval', default=True, type=bool)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', default=300, type=int)
    ap.add_argument('--batch_size', default=32, type=int)
    ap.add_argument('--pad_size', default=256, type=int)
    ap.add_argument('--embedding_size', default=768, type=int)
    ap.add_argument('--lr', default=1e-05, type=float)
    ap.add_argument('--dropout', default=0.5, type=float)
    ap.add_argument('--weight_decay', default=0.0005, type=float)
    ap.add_argument('--pt_model_name', default='bert-base-multilingual-cased')
    ap.add_argument('--early_stop', default=True, type=bool)
    ap.add_argument('--lang', type=str, default='English-Spanish')
    ap.add_argument('--freeze_layer', type=int, default=0, nargs='+')
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--save_dir', type=int, default=1)
    ap.add_argument('--data_amount', type=float, default=1.0)

    args = ap.parse_args()
    
    print(f'Loading pre-trained model {args.pt_model_name}')
    pt_model = BertModel.from_pretrained(args.pt_model_name,
                                         torch_dtype='auto')
    tokenizer = BertTokenizerFast.from_pretrained(args.pt_model_name)

    print('Starting training')

    print_info(args)

    print(f'Preprocessing training data')
    train_data, label_dict, idx2labels = \
        get_data(data_path=args.train_path,
                 pad_size=args.pad_size,
                 pad='max_length',
                 tokenizer=tokenizer)
    
    print(f'Preprocessing test data')
    test_data, label_dict, idx2labels = \
        get_data(data_path=args.test_path,
                 pad_size=args.pad_size,
                 pad=False,
                 label_dict=label_dict,
                 idx2labels=idx2labels,
                 tokenizer=tokenizer)

    n_special_labels = 4
    model = Enlabeler(pt_model=pt_model,
                      in_dim=args.embedding_size,
                      out_dim=len(label_dict) - n_special_labels,
                      dropout=args.dropout)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    modeling = Modeling(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        label_mapping=idx2labels,
                        all_labels={idx:lbl for lbl, idx in label_dict.items()},
                        patience=args.patience,
                        tokenizer=tokenizer)

    test, val = train_test_split([(x,y) for x, y in zip(test_data['x'],
                                                        test_data['y'])])
    
    test_data['x'], test_data['y'] = zip(*test)

    if args.train:
        
        train_dataset, train_dataloader = load_data(train_data,
                                                    label_dict,
                                                    args.batch_size,
                                                    args.data_amount)
        
        val_data = dict()
        val_data['x'], val_data['y'] = zip(*val)
        
        val_dataset, val_dataloader = load_data(val_data,
                                                label_dict,
                                                1,
                                                args.data_amount)
        
        print('Training model')
        
        final_epoch = modeling.train(train_data=train_dataloader,
                                     val_data=val_dataloader,
                                     train_data_size=len(train_dataset) // args.batch_size,
                                     val_data_size=len(val_dataset) // 1,
                                     n_epochs=args.epochs,
                                     early_stopping=args.early_stop,
                                     freeze_layers=args.freeze_layer)
 
    if args.eval:
        
        print('Evaluating model')
        
        eval_batch_size = 1
        
        test_dataset, test_dataloader = load_data(test_data,
                                                  label_dict,
                                                  eval_batch_size,
                                                  args.data_amount)
        
        (lbl_p, lbl_r, lbl_f), (tot_p, tot_r, tot_f), lbl_predictions = \
                                modeling.evaluate(data=test_dataloader,
                                data_size=len(test_dataset) // eval_batch_size)

        table = print_results(lbl_p, lbl_r, lbl_f, tot_p, tot_r, tot_f, args)

        write_results_to_file(args, table, tot_p, tot_r, tot_f, lbl_predictions, final_epoch)

if __name__ == '__main__':
    main()