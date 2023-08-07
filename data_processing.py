import pandas as pd
from sklearn.model_selection import train_test_split
import re

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_review(review):
    review = re.sub('<.*?>', '', review)
    review = review.lower()
    review = re.sub('[^\w\s]', '', review)
    review = re.sub('[^a-zA-Z\s]', '', review)
    return review

def prep_train_test(data, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test