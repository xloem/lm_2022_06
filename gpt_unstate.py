#!/usr/bin/env python3
import sys, pickle
import src.model
from src.model import RWKV_GPT
from src.model_train import GPT as GPT_Train, GPTConfig
import torch
src.
if len(sys.argv) != 3:
    print('Continues a prompt from a state file using RWKV_GPT')
    print(f'Usage: {sys.argv[0]} model_name state_name')
    sys.exit(-1)
MODEL_NAME=sys.argv[1]
STATE_NAME=sys.argv[2]

with open(STATE_NAME + '.pickle', 'rb') as file:
    state = pickle.load(file)

gptmodel = RWKV_GPT(MODEL_NAME=MODEL_NAME)

tokenizer = gptmodel.tokenizer

token_id = torch.argmax(state.next_logits,dim=-1).view(-1)[0]

token_ids = []

def process(token_id):
  # output token
  token = tokenizer.decode(token_id)
  sys.stdout.write(token)
  sys.stdout.flush()
  # pass through the model
  token_ids.append(token_id)
    # load state
  gptmodel.load(state)
  logits = gptmodel(torch.tensor([token_ids]), recur=True)[0,-1,:]
  # use max(range,key) as an argmax for python list to do greedy sampling
  token_id = torch.argmax(logits)
  return token_id

while True:
    token_id = process(token_id)
