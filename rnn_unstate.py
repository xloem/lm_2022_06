#!/usr/bin/env python3
import sys, pickle
from src.model import RWKV_RNN
import torch

if len(sys.argv) != 3:
    print('Continues a prompt from a state file using RWKV_RNN')
    print(f'Usage: {sys.argv[0]} model_name state_name')
    sys.exit(-1)
MODEL_NAME=sys.argv[1]
STATE_NAME=sys.argv[2]

with open(STATE_NAME, 'rb') as file:
    state = pickle.load(file)

rnnmodel = RWKV_RNN(MODEL_NAME=MODEL_NAME)
rnnmodel.load(state)
tokenizer = rnnmodel.tokenizer

token_id = torch.argmax(state.next_logits,dim=-1).view(-1)[0]

def process(token_id):
  # output token
  token = tokenizer.decode(token_id)
  sys.stdout.write(token)
  sys.stdout.flush()
  # pass through the model
  logits_list = rnnmodel.run([token_id])
  # use max(range,key) as an argmax for python list to do greedy sampling
  token_id = max(range(tokenizer.vocab_size), key=lambda token_id: logits_list[token_id])
  return token_id

while True:
  token_id = process(token_id)
