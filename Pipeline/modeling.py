"""Module for executing the model training loop and evaluation of the model."""

from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


class Modeling:

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 label_mapping,
                 all_labels,
                 patience: int,
                 tokenizer):
        
        self.model         = model
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.label_mapping = label_mapping
        self.all_labels    = all_labels
        self.patience      = patience
        
        self.correct   = []
        self.incorrect = []
        self.losses    = defaultdict(list)
        self.ave_loss  = []
        self.val_loss  = []
        self.val_losses = []

        self.tokenizer = tokenizer

    def _fit(self,
             data: DataLoader,
             data_size: int,
             epoch: int):
        """Fits model for one epoch"""

        with tqdm(data,
                  total=data_size,
                  unit='batch',
                  desc=f'Epoch {epoch+1}') as batches:

                for batch in batches:
                    with torch.autocast(device_type='cuda',
                                        dtype=torch.float16):

                        batch_size, seq_len, _ = batch['y'].shape

                        y = batch['y'].reshape(batch_size*seq_len, -1)
                        
                        loss_mask = torch.any(y != 0, dim=1)

                        output = self._forward(batch['x'])
                    
                    loss = self._backward(output, y, loss_mask)

                    batches.set_postfix(loss=loss)

                    if self.model.training:
                        self.losses[epoch].append(loss)
                    elif not self.model.training:
                        self.val_loss.append(loss)

    def _backward(self,
                  output: torch.Tensor,
                  y: torch.Tensor,
                  pad_mask: torch.Tensor):
        """Runs one backward pass."""
        
        og_loss = self.criterion(output, y)
                
        masked_loss = og_loss.where(pad_mask.unsqueeze(1), 0)
        
        loss = masked_loss.sum() / pad_mask.sum()

        if self.model.training:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return loss.item()

    def _forward(self,
                 x: dict):
        """Runs one forward pass"""

        self.model.zero_grad()

        output = self.model(x)

        return output

    def _early_stopping(self):
        
        return all([round(self.val_losses[-(i+1)], 2) >= round(self.val_losses[-(i+2)], 2)
                    for i in range(self.patience-1)])
        
    def _freeze_parameters(self, n_layers):
        if isinstance(n_layers, int):
            n_layers = [n_layers]
        
        for n_layer in n_layers:
            
            print(f'Freezing parameter {n_layer}')
            
            for parameter in self.model.pt_model.encoder.layer[n_layer-1].parameters():
                parameter.requires_grad = False

    def train(self,
              train_data: DataLoader,
              val_data: DataLoader,
              train_data_size: int,
              val_data_size: int,
              n_epochs: int,
              freeze_layers,
              early_stopping=None):
        """Trains model."""
        
        if freeze_layers:
            self._freeze_parameters(freeze_layers)
            
        for epoch in range(n_epochs):
            
            ##### TRAINING #####
            self.model.train()
            
            self._fit(train_data, train_data_size, epoch=epoch)

            self.ave_loss.append(sum(self.losses[epoch]) \
                                     / len(self.losses[epoch]))
            
            print(f'Average train loss for epoch {epoch+1}: {self.ave_loss[-1]}')


            ##### VALIDATION #####
            self.model.eval()

            print('VALIDATING')
            with torch.no_grad():
                self._fit(val_data, val_data_size, epoch=epoch)

            print(f'Average validation loss for epoch \
                   {epoch+1}: {sum(self.val_loss) / len(self.val_loss)}')

            self.val_losses.append(sum(self.val_loss) / len(self.val_loss))
            self.val_loss = []

            if early_stopping:
               if epoch >= self.patience and self._early_stopping():
                   print('Stopping early!!')
                   break

        return epoch

    def predict(self,
                sentence: dict[torch.Tensor]):
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            batch_size, seq_len, _ = sentence['y'].shape

            y = sentence['y'].reshape(batch_size*seq_len, -1)

            sentence['x']['input_ids'] = \
                sentence['x']['input_ids'].squeeze(0)

            output = self._forward(sentence['x'])

            y_hat = torch.where(torch.sigmoid(output)>0.5,
                                1.0,
                                0.0)
        
        return y_hat, y

    def _fmeasure(self, ground_truth, predicted, average=None):
        return precision_recall_fscore_support(ground_truth,
                                               predicted,
                                               average=average,
                                               zero_division=0.0)

    @torch.no_grad()
    def evaluate(self,
                 data: DataLoader,
                 data_size: int):
        
        self.model.eval()
        
        predicted = []
        ground_truth = []
        lbl_predictions = []

        with tqdm(data,
                  total=data_size,
                  unit='Sentence') as sentences:

            for sentence in sentences:

                y_hat, y = self.predict(sentence)

                lbl_predictions.append(self._generated_labels(sentence['x'], y, y_hat))

                mask = torch.any(y != 0, dim=1)

                ground_truth.extend(y[mask].tolist())
                predicted.extend(y_hat[mask].tolist())

        lbl_precision, lbl_recall, lbl_fscore, _ = self._fmeasure(ground_truth,
                                                                  predicted)

        lbl_precision = self._map_scores(lbl_precision)
        lbl_recall = self._map_scores(lbl_recall)
        lbl_fscore = self._map_scores(lbl_fscore)

        total_precision, total_recall, total_fscore, _ = \
            self._fmeasure(ground_truth, predicted, average='micro')
        
        lblwise_metrics = lbl_precision, lbl_recall, lbl_fscore
        global_metrics = total_precision, total_recall, total_fscore
                
        return lblwise_metrics, global_metrics, lbl_predictions
    
    def _map_scores(self, scores):
        scores_per_label_type = defaultdict(list)

        average = lambda x: sum(x) / len(x)

        for idx, label in self.label_mapping.items():
            scores_per_label_type[label].append(scores[idx])

        for key, val in scores_per_label_type.items():
            scores_per_label_type[key] = average(val)

        return scores_per_label_type

    def _generated_labels(self, x, y, y_hat):
        def pool(inp):
            """Pools subtokens back to word-level."""
            
            ids = inp['input_ids'].squeeze().tolist()
            tokenized_seq = self.tokenizer.convert_ids_to_tokens(ids)
            pooled = []
            for token in tokenized_seq:
                if token[:2] == '##':
                    pooled[-1] = ''.join([pooled[-1], token[2:]])
                pooled.append(token)
            return pooled
        
        sentence = pool(x)
        
        output = []
        
        for i, token in enumerate(sentence):
            correct = [self.all_labels[i.item()]
                       for lbls in torch.nonzero(y[i, :]) for i in lbls]
            predicted = [self.all_labels[i.item()]
                       for lbls in torch.nonzero(y_hat[i, :]) for i in lbls]
            if correct:
                output.append((token, ', '.join(correct), ', '.join(predicted)))
            
        return output