import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer

from transformers import BertForSequenceClassification
import torch.utils.data as Data
from transformers import AdamW, get_linear_schedule_with_warmup

import text_classification

import json
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score



if __name__ == '__main__':
    setting_path = 'etc.json'
    with open(setting_path,"r") as load_f:
        setting = json.load(load_f)
    rejected_words = setting['rejected_words']
    not_related_words = setting['not_related_words']
    validation_portion = setting['validation_portion']
    out_json_path = 'record_dict.json'

    if os.path.exists(out_json_path):
        with open(out_json_path,"r") as load_f:
            record_dict = json.load(load_f)
            print("loading...")
    else:
        record_dict = {"cv":{},"tfidf":{}}
        print ('creating...')

    df = pd.read_csv('df.csv',encoding='utf-8')
    df = text_classification.DropRep(df)
    
    df['tweet_text'] = text_classification.TransformMerge([text_classification.text_prepare(x,not_related_words,rejected_words) for x in df['tweet_text']],rejected_words)

    print (df.head())
    print (df['label'].value_counts())
    possible_labels = df.label.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    df['label_id'] = df.label.replace(label_dict)
    

    X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
        df.label_id.values, 
        test_size=validation_portion, 
        # random_state=42, 
        stratify=df.label_id.values)

    df['data_type'] = ['not_set']*df.shape[0]

    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    print (df.groupby(['label', 'label_id', 'data_type']).count())

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
        do_lower_case=True)
    
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].tweet_text.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'].tweet_text.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )


    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label_id.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label_id.values)

    dataset_train = Data.TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = Data.TensorDataset(input_ids_val, attention_masks_val, labels_val)
    
    print (len(dataset_train), len(dataset_val))

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False)
    
    

    batch_size = 10

    dataloader_train =  Data.DataLoader(dataset_train, 
        sampler= Data.RandomSampler(dataset_train), 
        batch_size=batch_size)

    dataloader_validation =  Data.DataLoader(dataset_val, 
        sampler= Data.SequentialSampler(dataset_val), 
        batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(),
        lr=1e-5, 
        eps=1e-8)
    
    epochs = 5

    scheduler = get_linear_schedule_with_warmup(optimizer,  
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train)*epochs)
    
    def f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(preds, labels):
        label_dict_inverse = {v: k for k, v in label_dict.items()}
        
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
    
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model.to(device)

    print(device)

    def evaluate(dataloader_val):
    
        model.eval()
        
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in dataloader_val:
            
            batch = tuple(b for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }

            with torch.no_grad():        
                outputs = model(**inputs)
                
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        
        loss_val_avg = loss_val_total/len(dataloader_val) 
        
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
                
        return loss_val_avg, predictions, true_vals
    
    for epoch in tqdm(range(1, epochs+1)):
        
        model.train()
        
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()
            
            batch = tuple(b for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
            
        torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
            
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False)
    
    model.load_state_dict(torch.load('finetuned_BERT_epoch_1.model'))

    _, predictions, true_vals = evaluate(dataloader_validation)

    accuracy_per_class(predictions, true_vals)

        
