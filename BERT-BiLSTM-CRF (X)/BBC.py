import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import Model
import torch
from torchcrf import CRF


# Load data
with open('data/crf_train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('data/crf_test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

# Preprocessing function to tokenize and encode the sentences and labels
def preprocess_data(data, tokenizer, label_encoder, max_len):
    sentences = ["".join([t[0] for t in sentence]) for sentence in data]
    labels = [[t[1] for t in sentence] for sentence in data]
    
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=max_len, return_tensors="tf", is_split_into_words=True)
    input_ids, attention_masks = tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]
    
    all_labels = [label for sentence_labels in labels for label in sentence_labels]
    label_encoder.fit(all_labels)
    encoded_labels = [label_encoder.transform(sentence_labels) for sentence_labels in labels]
    encoded_labels_padded = pad_sequences(encoded_labels, maxlen=max_len, padding='post', value=-1)
    
    return input_ids, attention_masks, encoded_labels_padded

max_len = 128

# Initialize label encoder
label_encoder = LabelEncoder()

# Prepare the training data
train_input_ids, train_attention_masks, train_labels = preprocess_data(train_data, tokenizer, label_encoder, max_len)

# Prepare the test data
test_input_ids, test_attention_masks, test_labels = preprocess_data(test_data, tokenizer, label_encoder, max_len)

# Convert labels to one-hot encoding
num_labels = len(label_encoder.classes_)
train_labels_one_hot = [to_categorical(i, num_classes=num_labels+1) for i in train_labels]
test_labels_one_hot = [to_categorical(i, num_classes=num_labels+1) for i in test_labels]

# Define the model
input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

# BERT model
bert_model = TFBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
sequence_output = bert_model(input_ids, attention_mask=attention_mask)[0]

# BiLSTM layer
bi_lstm = Bidirectional(LSTM(units=64, return_sequences=True))(sequence_output)
dropout = Dropout(0.5)(bi_lstm)
dense = TimeDistributed(Dense(num_labels, activation="relu"))(dropout)

# CRF layer
crf = CRF(num_labels)
logits = tf.convert_to_tensor(dense)
output = crf.decode(logits)

model = Model(inputs=[input_ids, attention_mask], outputs=output)

# Define loss and metrics
def crf_loss(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.int32)
    log_likelihood, _ = crf(y_pred, y_true)
    return -log_likelihood

def crf_accuracy(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    y_pred = tf.cast(y_pred, dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true[mask], y_pred[mask]), dtype=tf.float32))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=crf_loss, metrics=[crf_accuracy])

model.summary()

# Convert the one-hot encoded labels to numpy arrays
train_labels_one_hot = np.array(train_labels_one_hot)
test_labels_one_hot = np.array(test_labels_one_hot)

# Train the model
history = model.fit(
    [train_input_ids, train_attention_masks],
    train_labels_one_hot,
    validation_data=([test_input_ids, test_attention_masks], test_labels_one_hot),
    batch_size=16,
    epochs=5
)

# Predict on the test data
test_pred = model.predict([test_input_ids, test_attention_masks])
test_pred_labels = np.argmax(test_pred, axis=-1)

# Convert the labels back to their original form
true_labels = np.concatenate(test_labels)
pred_labels = np.concatenate(test_pred_labels)

# Remove padding (-1)
mask = true_labels != -1
true_labels = true_labels[mask]
pred_labels = pred_labels[mask]

# Inverse transform the labels
true_labels = label_encoder.inverse_transform(true_labels)
pred_labels = label_encoder.inverse_transform(pred_labels)

# Print classification report
print(classification_report(true_labels, pred_labels))
