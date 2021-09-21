import torch
from transformers import *
from fastai.text.all import *
from blurr.data.all import *
from blurr.modeling.all import *
import unidecode
import wordninja
import pandas as pd
import numpy as np

df=pd.read_csv('/Users/brl.314/Downloads/SPAM M:L/India/Large Dataset/Training August 2021/Train_Aug2.csv')

df2=pd.read_csv('/Users/brl.314/Downloads/SPAM M:L/India/Large Dataset/Training August 2021/Test_Aug2.csv')

import re
def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text

df['content'] = df['content'].apply(clean_text).astype('str')
df2['content'] = df2['content'].apply(clean_text).astype('str')

df['type'].replace({'Spam':0,'Not Spam':1},inplace=True)
df2['type'].replace({'Spam':0,'Not Spam':1},inplace=True)

train_df = df

test_df = df2

#Training model
n_labels = len(train_df['type'].unique())

model_cls = AutoModelForSequenceClassification

pretrained_model_name = "bert-base-multilingual-cased"

config = AutoConfig.from_pretrained(pretrained_model_name)

config.num_labels = n_labels

hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, model_cls=model_cls, config=config)

# single input
blocks = (HF_TextBlock(hf_arch, hf_config, hf_tokenizer, hf_model), CategoryBlock)

dblock = DataBlock(blocks=blocks,  get_x=ColReader('content'), get_y=ColReader('type'), splitter=ColSplitter())

## Create DataLoader using Blurr and FastAI Data loaders
blocks = (HF_TextBlock(hf_arch, hf_config, hf_tokenizer, hf_model), CategoryBlock)
dblock = DataBlock(
    blocks=blocks,  
    get_x=ColReader('content'), 
    get_y=ColReader('type'), 
    splitter=RandomSplitter(0.2, seed=42))
dls = dblock.dataloaders(train_df, bs=16)

#slow
model = HF_BaseModelWrapper(hf_model)

learn = Learner(dls, 
                model,
                opt_func=partial(Adam, decouple_wd=True),
                loss_func=CrossEntropyLossFlat(),
                metrics=[accuracy],
                cbs=[HF_BaseModelCallback],
                splitter=hf_splitter)

learn.freeze()

learn.fit_one_cycle(5, lr_max=1e-3)

#export_fname = 'fastai_indiaspam_export'
learn.export('/Users/brl.314/Downloads/SPAM M:L/India/Large Dataset/Training August 2021/Train_Aug2_distillbert-multilingual.pkl')

inf_learn = load_learner('/Users/brl.314/Downloads/SPAM M:L/India/Large Dataset/Training August 2021/Train_Aug2_distillbert-multilingual.pkl')

inf_learn.blurr_predict("g.i.r.l.s.j.o.i.n.t.h.e.m.e.e.t.i.n.g bdy-adqu-byo")