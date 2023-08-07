import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import BertTokenizer, BertModel,BertForSequenceClassification , AutoModelForSequenceClassification
from transformers import InputExample, InputFeatures
import torch

def train_batch(model, train_input_ids, train_attention_masks, train_labels, optimizer, scheduler, batch_size, num_epochs):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        for i in range(0, len(train_input_ids), batch_size):
            batch_input_ids = train_input_ids[i : i + batch_size].to(device)
            batch_attention_masks = train_attention_masks[i : i + batch_size].to(device)
            batch_labels = train_labels[i : i + batch_size].to(device)

            outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)

            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
