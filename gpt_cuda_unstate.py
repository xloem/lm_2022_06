#!/usr/bin/env python3
import sys, pickle
import src.model
from src.model_train import GPT, GPTConfig, T_MAX, B_GROUP_FORWARD
import torch

if len(sys.argv) != 3:
    print('Continues a prompt from a state file using src.model_train.GPT')
    print(f'Usage: {sys.argv[0]} model_name state_name')
    sys.exit(-1)
MODEL_NAME=sys.argv[1]
STATE_NAME=sys.argv[2]

with open(STATE_NAME + '.pickle', 'rb') as file:
    state = pickle.load(file)

w = torch.load(MODEL_NAME + '.pth', map_location=src.model.RUN_DEVICE)
vocab_size, n_embd = w['emb.weight'].shape
n_layer = len([key for key in w.keys() if key.endswith('att.key.weight')])
ctx_len = src.model.ctx_len - 1
#ctx_len = T_MAX - 1

gptmodel = GPT(GPTConfig(vocab_size, ctx_len, n_embd=n_embd, model_type='RWKV', n_layer=n_layer)).cuda()
gptmodel.load_state_dict(w)

tokenizer = src.model.PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

token_id = torch.argmax(state.next_logits,dim=-1).view(-1)[0]

token_ids = torch.zeros(1,ctx_len,dtype=int,device='cuda')
token_id_ct = 0

def process(token_id):
  global token_id_ct
  # output token
  token = tokenizer.decode(token_id)
  sys.stdout.write(token)
  sys.stdout.flush()
  # pass through the model
    # load state
  gptmodel.load(state)
  with torch.no_grad():
      logits = gptmodel(token_ids.expand(B_GROUP_FORWARD,ctx_len), recur=True)[0][0,token_id_ct,:]
  # use max(range,key) as an argmax for python list to do greedy sampling
  token_id = torch.argmax(logits)
  token_ids[0,token_id_ct] = token_id
  token_id_ct += 1
  return token_id

while True:
    token_id = process(token_id)
