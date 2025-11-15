#Import libraries and load model
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

#Load LSTM model
model = load_model('seq_words_predictor.h5')

#Load tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

#Function to predict the next word
def predict_next_word(model,tokenizer,text,max_seq_len):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list)>=max_seq_len:
    token_list = token_list[-(max_seq_len-1):] #Ensure list is same length as x input (i.e., maxseqlen-1)
  else:
    token_list = pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')
  prediction = model.predict(token_list)
  predicted_word_index = np.argmax(prediction,axis=1) #argmax finds index of highest prob value in softmax output, axis=1 means find max across each column (here we have 1 row only due to 1 input sentence)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None

#Streamlit UI
st.title('Next word prediction with LSTM and EarlyStopping')
input_text=st.text_input("Enter the sequence of words","To be or not to be")
if st.button('Predict next word'):
  max_seq_len = model.input_shape[1]+1
  next_word = predict_next_word(model,tokenizer,input_text,max_seq_len)
  st.write(f'Predicted next word: {next_word}')