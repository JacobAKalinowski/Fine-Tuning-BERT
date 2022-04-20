import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, load_metric

#Loading Tokenizer and sst2 dataset from Huggingface

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset("glue", "sst2")

dataset

#Splitting dataset
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

#Function to pass into dataset
def tokenize_help(data):
  return tokenizer(data["sentence"], truncation = True)

tokenized_train_data = train_data.map(tokenize_help, batched = True)
tokenized_validation_data = validation_data.map(tokenize_help, batched = True)
tokenized_test_data = test_data.map(tokenize_help, batched = True)

print(tokenized_train_data.features)

#Removing columns that BERT does not expect
tokenized_train_data = tokenized_train_data.remove_columns(["sentence", "idx"])
tokenized_train_data = tokenized_train_data.rename_column("label", "labels")

tokenized_validation_data = tokenized_validation_data.remove_columns(["sentence", "idx"])
tokenized_validation_data = tokenized_validation_data.rename_column("label", "labels")

tokenized_test_data = tokenized_test_data.remove_columns(["sentence", "idx"])
tokenized_test_data = tokenized_test_data.rename_column("label", "labels")

#Formatting to torch tensors
tokenized_train_data.set_format("torch")
tokenized_validation_data.set_format("torch")
tokenized_test_data.set_format("torch")

from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

train_iterator = DataLoader(
    tokenized_train_data, shuffle = True, batch_size = 4, collate_fn = DataCollatorWithPadding(tokenizer = tokenizer)
)

validation_iterator = DataLoader(
    tokenized_validation_data, shuffle = True, batch_size = 4, collate_fn = DataCollatorWithPadding(tokenizer = tokenizer)
)

#See what the data looks like in each batch
for batch in train_iterator:
    break
{k: v.shape for k, v in batch.items()}

#Setting Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#Loading the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2)
model.to(device)

#Loading the model after fine tuning
#model.load_state_dict(torch.load("fine_tuned_bert"))

from torch import optim
from transformers import get_scheduler


#Creating optimizer and scheduler
epochs = 2
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
steps = epochs * len(train_iterator)
scheduler = get_scheduler(
    "linear", 
    optimizer = 
    optimizer, 
    num_warmup_steps = 0, 
    num_training_steps = steps
)

from tqdm.auto import tqdm

#Setting a progress bar for training
progress = tqdm(range(steps))

for epoch in range(epochs):
  for batch in train_iterator:
      batch = {k: v.to(device) for k, v in batch.items()}
      output = model(**batch)

      loss = output.loss
      loss.backward()

      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      progress.update(1)

#Using metric given by SST2 
metric = load_metric("glue", "sst2")
model.eval
for batch in validation_iterator:
    batch = {k: v.to(device) for k, v in batch.items()}
    #Evaluating Model
    with torch.no_grad():
        out = model(**batch)
        
    logits = out.logits
    predictions = torch.argmax(logits, dim = -1)
    metric.add_batch(predictions = predictions, references = batch["labels"])
    
metric.compute()

#Saving Model

torch.save(model.state_dict(), "fine_tuned_bert")

