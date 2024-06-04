import torch
import torch.nn as nn


class Enlabeler(nn.Module):

    def __init__(self,
                 pt_model,
                 in_dim: int,
                 out_dim: int,
                 get_logits_from='last_hidden_state',
                 output_hidden_states=False,
                 output_attentions=False,
                 dropout=0.5):
        
        super(Enlabeler, self).__init__()
        
        self.pt_model = pt_model
        self.linear   = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(in_dim, out_dim))
        
        self.get_logits_from    = get_logits_from
        self.outp_hdn_states    = output_hidden_states
        self.output_attentions  = output_attentions
        
        if torch.cuda.is_available():
            self.cuda()
        else:
            print('CUDA not available!!')
            exit()

    def forward(self,
                x: dict):
        
        if len(x['input_ids'].shape) == 3:  # If batched
            x['input_ids'] = x['input_ids'].squeeze(1)
            x['attention_mask'] = x['attention_mask'].squeeze(1)
            
        model_output = self.pt_model(input_ids=x['input_ids'],
                                     attention_mask=x['attention_mask'],
                                     output_hidden_states=self.outp_hdn_states,
                                     output_attentions=self.output_attentions)

        logits = model_output['last_hidden_state']
        batch_size, seq_len, _ = logits.shape
        reshaped = logits.view(batch_size*seq_len, -1)
        output = self.linear(reshaped)


        #return output, x['attention_mask'].flatten() != 0
        return output