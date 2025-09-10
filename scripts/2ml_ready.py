import os
import json
import torch
from transformers import AutoTokenizer

json_path = "./yProv4MLProvenanceDataset/data/"
dst_path = "./yProv4MLProvenanceDataset/ml_ready/"
desc_path = "./yProv4MLProvenanceDataset/descriptions/"

files = os.listdir(json_path)
json_files = [json_path + file for file in files]
desc_files = [desc_path + file.replace(".json", ".txt") for file in files]

model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "EleutherAI/pythia-1.4b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 


def preprocess(data): 
    final_data = []
    prefix = json.dumps(data["prefix"], indent=2)
    final_data.append(prefix)

    entities = data["entity"].keys()
    for ent in entities: 
        final_data.append(json.dumps({ent: data["entity"][ent]}, indent=2))
    activities = data["activity"].keys()
    for act in activities: 
        final_data.append(json.dumps({act: data["activity"][act]}, indent=2))

    return "\n".join(final_data)


for index in range(len(json_files)): 
    json_file = json.load(open(json_files[index], "r"))
    json_data = preprocess(json_file)
    desc_file = open(desc_files[index]).read()

    inp_ids = tokenizer(json_data, truncation=True, padding="max_length", max_length=256)["input_ids"]
    out_ids = tokenizer(desc_file, truncation=True, padding="max_length", max_length=256)["input_ids"]
    
    torch.save(torch.tensor(inp_ids), f"{dst_path}/input_ids_{index}.pt")
    torch.save(torch.tensor(out_ids), f"{dst_path}/labels_{index}.pt")

