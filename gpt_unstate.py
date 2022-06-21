#!/usr/bin/env python3
import sys, pickle
import src.model
from src.model import RWKV_GPT
import torch

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

# a long context is filled to simplify comparison with gpt_cuda
token_ids = torch.zeros(1,src.model.ctx_len - 1,dtype=int)
token_id_ct = 0

def process(token_id):
  global token_id_ct
  # output token
  token = tokenizer.decode(token_id)
  sys.stdout.write(token)
  sys.stdout.flush()
  # pass through the model
  token_ids[0,token_id_ct] = token_id
  token_id_ct += 1
    # load state
  gptmodel.load(state)
  logits = gptmodel(token_ids, recur=True)[0,token_id_ct-1,:]
  # use max(range,key) as an argmax for python list to do greedy sampling
  token_id = torch.argmax(logits)
  return token_id

while True:
    token_id = process(token_id)
