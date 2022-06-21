#!/usr/bin/env python3
import sys, pickle, types
import torch
from src.model import RWKV_GPT

if len(sys.argv) != 3:
    print('Condenses a prompt to a state file using RWKV_GPT')
    print(f'Usage: {sys.argv[0]} model_name prompt_file state_name')
    sys.exit(-1)
MODEL_NAME=sys.argv[1]
PROMPT_FILE=sys.argv[2]
STATE_NAME=sys.argv[3]
gptmodel = RWKV_GPT(MODEL_NAME=MODEL_NAME)
tokenizer = gptmodel.tokenizer

print('reading ...')
with open(PROMPT_FILE) as file:
    token_ids = tokenizer.encode(file.read())
print('processing ...')
with torch.no_grad():
    gptoutput = gptmodel(torch.tensor([token_ids]))
print('extracting ...')

state = types.SimpleNamespace()
state.next_logits = gptoutput[...,-1,:]
gptmodel.save(state)
print('writing ...')
with open(STATE_NAME + '.pickle', 'wb') as file:
    pickle.dump(state, file)
print('written')

