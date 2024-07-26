#!/usr/bin/env python
# coding: utf-8

# # GPT-2 Fine-Tuning

# #### This is the code I wrote at the company, but I think it would be nice to share it here, so I post it.
# 
# #### With this data, we will fine tune GPT-2 to make a sentence generation model. 
# 
# #### This code is for AI beginners.

# ## Step 1. Data preprocessing

# #### the data contains unnecessary newlines, tags, and URLs it will be necessary to remove them before preprocessing.

# In[1]:


import pandas as pd
import numpy as np
import re


# ## Step 2. Model Training

# In[2]:



# In[3]:


from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

# In[4]:


def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
        
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator


def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):

  print("Start loading model")
  tokenizer = GPT2Tokenizer.from_pretrained(model_name)
  print("Start loading dataset")
  train_dataset = load_dataset(train_file_path, tokenizer)
  print("Start loading collator")
  data_collator = load_data_collator(tokenizer)
  print("Start saving tokenizer")
  # tokenizer.save_pretrained(output_dir)
  print("Start loading tokenizer")
  model = GPT2LMHeadModel.from_pretrained(model_name)
  print("Start saving model")
  # model.save_pretrained(output_dir)

  print("Finish loading")
  training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
      )

  trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
  )
      
  trainer.train()
  trainer.save_model()


# In[ ]:


# you need to set parameters 
train_file_path = "/home/aigc_ehr/cache/output.txt"
model_name = 'gpt2'
output_dir = '/home/aigc_ehr/cache/'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 5.0
save_steps = 500


# In[ ]:

print("Start training")

# It takes about 30 minutes to train in colab.
train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)



# The following process may be a little more complicated or tedious because you have to write the code one by one, and it takes a long time if you don't have a personal GPU.
# 
# Then, how about use Ainize's Teachable NLP? Teachable NLP provides an API to use the model so when data is input it will automatically learn quickly.
# 
# Teachable NLP : [https://ainize.ai/teachable-nlp](https://link.ainize.ai/3tJVRD1)
# 
# Teachable NLP Tutorial : [https://forum.ainetwork.ai/t/teachable-nlp-how-to-use-teachable-nlp/65](https://link.ainize.ai/3tATaUh)
