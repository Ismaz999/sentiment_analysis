import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel,BertForSequenceClassification , AutoModelForSequenceClassification
from transformers import InputExample, InputFeatures


from batch_processing import train_batch
from data_processing import load_dataset, clean_review, prep_train_test

###### Chargement des données #####
data = load_dataset('IMDB Dataset.csv')

num_reviews = 50
sampled_data = data.sample(n=num_reviews, random_state=42)

###### Nettoyage des données  #####
sampled_data['review'] = sampled_data['review'].apply(clean_review)


sampled_data['sentiment'] = sampled_data['sentiment'].map({'positive': 1, 'negative': 0})

###### préparation des données pour l'entrainement ######
x_train, x_test, y_train, y_test = prep_train_test(sampled_data)

###### Chargement du tokenizer et du modèle #####
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

###### Configuration de l'appareil (GPU si disponible, sinon CPU) #####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Tokenization des trains inputs ######
train_encodings = tokenizer(list(x_train), truncation=True, padding=True, return_tensors='pt')
train_input_ids = train_encodings['input_ids']
train_attention_masks = train_encodings['attention_mask']
train_labels = torch.tensor(y_train.tolist())

###### Tokenization des test inputs ######
test_encodings = tokenizer(list(x_test), truncation=True, padding=True, return_tensors='pt')
test_input_ids = test_encodings['input_ids']
test_attention_masks = test_encodings['attention_mask']
test_labels = torch.tensor(y_test.tolist())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)

###### Mettre le modele en mode training ###### On place notre modele en mode train afin d'éviter un overfitting et met a jour ses poids afin de minimiser la fonction de perte
model.train()

# Define optimizer and learning rate scheduler (you can adjust these as needed)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

###### Transfert des données au modèle par batch ######
predicted_sentiments = []

###### Définition de la taille du batch ######
batch_size = 12
num_epochs = 10

###### Entrainement du réseau ######
train_batch(model, train_input_ids, train_attention_masks, train_labels, optimizer, scheduler, batch_size, num_epochs)

###### Modèle mis en mode évaluation ######
model.eval()

###### Augmentation des ressources en arretant le gradient descent ######
torch.set_grad_enabled(False)

###### Génération des prédictions ######
with torch.no_grad():
    test_outputs = model(test_input_ids.to(device), attention_mask=test_attention_masks.to(device))
    test_predictions = torch.argmax(test_outputs.logits, dim=1)

###### Evaluation du modèle (accuracy, matrice de confusion) ######
accuracy = accuracy_score(test_labels.tolist(), test_predictions.tolist())
confusion_matrix = confusion_matrix(test_labels.tolist(), test_predictions.tolist())

print("Accuracy:", accuracy)
print("Confusion Matrix:", confusion_matrix)