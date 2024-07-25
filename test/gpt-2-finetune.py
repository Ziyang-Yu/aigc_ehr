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

# In[ ]:


import pandas as pd
import numpy as np
import re


# In[ ]:


def cleaning(s):
    s = str(s)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\w*"," ")
    return s


# In[ ]:


df = pd.read_csv("Articles.csv", encoding="ISO-8859-1") 
df = df.dropna()
text_data = open('Articles.txt', 'w')
for idx, item in df.iterrows():
  article = cleaning(item["Article"])
  text_data.write(article)
text_data.close()


# ## Step 2. Model Training

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


# In[ ]:


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
  tokenizer = GPT2Tokenizer.from_pretrained(model_name)
  train_dataset = load_dataset(train_file_path, tokenizer)
  data_collator = load_data_collator(tokenizer)

  tokenizer.save_pretrained(output_dir)
      
  model = GPT2LMHeadModel.from_pretrained(model_name)

  model.save_pretrained(output_dir)

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
train_file_path = "/content/drive/MyDrive/Articles.txt"
model_name = 'gpt2'
output_dir = '/content/drive/MyDrive/result'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 5.0
save_steps = 500


# In[ ]:


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


# ## Step 3. Inference

# In[ ]:


from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer


# In[ ]:


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(sequence, max_length):
    model_path = "/content/drive/MyDrive/result"
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))


# In[ ]:


sequence = input() # oil price
max_len = int(input()) # 20
generate_text(sequence, max_len) # oil price for July June which had been low at as low as was originally stated Prices have since resumed


# The following process may be a little more complicated or tedious because you have to write the code one by one, and it takes a long time if you don't have a personal GPU.
# 
# Then, how about use Ainize's Teachable NLP? Teachable NLP provides an API to use the model so when data is input it will automatically learn quickly.
# 
# Teachable NLP : [https://ainize.ai/teachable-nlp](https://link.ainize.ai/3tJVRD1)
# 
# Teachable NLP Tutorial : [https://forum.ainetwork.ai/t/teachable-nlp-how-to-use-teachable-nlp/65](https://link.ainize.ai/3tATaUh)
