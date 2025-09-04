
import argparse
import json
import os
import datetime

CAPTION_PATH = "/Users/gabrielepadovani/Desktop/Università/PhD/yProv4MLProvenanceDataset/descriptions/"
DATA_PATH = "/Users/gabrielepadovani/Desktop/Università/PhD/yProv4MLProvenanceDataset/data/"

def handle_v0(data, version): 
    namespace = f'creating a user namespace at {data["prefix"]["default"]}'
    following_w3c = ("not " if "context" in data["prefix"].keys() else "") + f"following the W3C prov schema ({data['prefix']['xsd_1']})"
    experiment = data["entity"][list(data["entity"].keys())[0]][0]
    exp_name = list(data["entity"].keys())[0]
    user = experiment["prov-ml:user_id"]
    pyversion = experiment["prov-ml:python_version"]
    run_id = f"with run_id: {experiment['prov-ml:run_id']}"
    starttime = float(data["entity"]["execution_start_time"]['prov-ml:parameter_value'])
    st = datetime.datetime.fromtimestamp(starttime).strftime('%Y-%m-%d %H:%M:%S')
    endtime = float(data["entity"]["execution_end_time"]['prov-ml:parameter_value'])
    et = datetime.datetime.fromtimestamp(endtime).strftime('%Y-%m-%d %H:%M:%S')

    source_code = None
    env = None
    if "source_code" in data["entity"]["source_code"].keys(): 
        source_code = data["entity"]["source_code"]["prov-ml:source_name"]
        env = data["entity"]["source_code"]["prov-ml:runtime_type"]
    repo = None

    max_epochs = len(data["activity"].keys())
    max_epochs = max_epochs if max_epochs > 0 else None
    
    list_params = []
    datasets = []
    artifacts = []
    for ent in data["entity"].keys(): 
        if not isinstance(data["entity"][ent], list): 
            if "prov-ml:type" in data["entity"][ent].keys(): 
                if data["entity"][ent]["prov-ml:type"] == "Parameter": 
                    list_params.append(ent)
                if data["entity"][ent]["prov-ml:type"] == "Dataset": 
                    datasets.append(ent)
    list_params = None if len(list_params) == 0 else ", ".join(list_params)
    datasets = None if len(datasets) == 0 else datasets
    
    num_samples = []
    num_steps = []
    batchsize = []
    if datasets: 
        for d in datasets: 
            try: 
                bs = data["entity"][d]["prov-ml:size"] 
                steps = data["entity"][d]["prov-ml:steps"]
                samples = data["entity"][d]["prov-ml:samples"]
                num_samples.append(samples)
                num_steps.append(steps)
                batchsize.append(bs)
            except: 
                datasets.remove(d)

    list_checkpoints = [k for k in list(data["entity"].keys()) if k.endswith(".pt") or k.endswith(".pth")]
    num_checkpoints = len(list_checkpoints)
    list_checkpoints = None if len(list_checkpoints) == 0 else ", ".join(list_checkpoints)
    
    final_model = data["entity"]["model_name"]['prov-ml:parameter_value'] if "model_name" in data["entity"].keys() else None
    model_params = data["entity"]["total_params"]['prov-ml:parameter_value'] if "total_params" in data["entity"].keys() else None
    memory_of_model = data["entity"]["memory_of_model"]['prov-ml:parameter_value'] if "memory_of_model" in data["entity"].keys() else None

    list_artifacts = None

    metrics = ", ".join([k for k in data["entity"].keys() if "_Context." in k])

    contexts = set([k.split("_Context.")[-1].replace("_Context.", "") for k in data["entity"].keys() if "_Context." in k])
    contexts = None if len(contexts) == 0 else ", ".join(list(contexts))

    description = \
        f"A provenance file created using the yProv4ML library (version {version}), {following_w3c}, and {namespace}. " +\
        f"The file contains a machine learning process {run_id}, started by user {user}, with python {pyversion}, and named {exp_name}. \n" + \
        f"The entire experiment was run in a {env} environment, " + \
        (f"and the user also attached a source code file, as the local file {source_code}. " if source_code else "") + \
        (f"The code is available at the github repository {repo}. " if repo else "") + \
        f"The experiment was scheduled at {st} and it finished at {et}, for a total of {round(endtime-starttime, 2)} seconds. \n" + \
        f"The model was trained for {max_epochs} epochs" + ("," if num_samples else ". ") + \
        (f"each epoch consisting in the passthrough of {num_samples[0]} samples, with batch size {batchsize[0]}. " if num_samples else "") + \
        (f"The model was saved {num_checkpoints} times, in the checkpoints {list_checkpoints}" + ". " if list_checkpoints else ", ") + \
        (f"with the final version being {final_model}. " if final_model else "") + \
        (f"The final model has a total of {model_params} parameters" if model_params else "") + \
        (f", and the footprint on the memory is of {round(float(memory_of_model), 2)} Mb. " if memory_of_model else ". ") + \
        (f"The datasets used for the training of the model are: " if datasets else "") + \
        ("; ".join([f"{d} with {s} samples, {bs} batch size and {steps} steps" for d, s, bs, steps in zip(datasets, num_samples, batchsize, num_steps)]) + ". \n" if datasets else "") + \
        (f"Additional artifacts saved are: {list_artifacts}." if list_artifacts else "") + \
        (f"The user saved a set of parameters from the process, including: {list_params}" if list_params else "") + \
        (f"A set of metrics have been collected from the process, in particular in the contexts {contexts}, " if contexts else "") +\
        (f"these metrics are {metrics}. " if contexts else "")
    
    return description

def handle_v1(data, version): 
    namespace = data["prefix"]["default"]
    following_w3c = ("not " if "context" in data["prefix"].keys() else "") + f"following the W3C prov schema ({data['prefix']['xsd_1']})"
    experiment = data["activity"]["context:" + namespace]
    exp_name = list(data["entity"].keys())[0]
    user = list(data["agent"].keys())[0]
    pyversion = experiment["yProv4ML:python_version"]
    run_id = f"with run_id {experiment['yProv4ML:run_id']['$']}"
    starttime = float(experiment["execution_start_time"])
    st = datetime.datetime.fromtimestamp(starttime).strftime('%Y-%m-%d %H:%M:%S')
    endtime = float(experiment["execution_end_time"])
    et = datetime.datetime.fromtimestamp(endtime).strftime('%Y-%m-%d %H:%M:%S')

    source_code = data["entity"]["source_code"]["yProv4ML:path"]
    env = "single core" if experiment["prov-ml:execution_command"].startswith("python ") else "cluster"
    repo = None 

    max_epochs = len(data["activity"].keys())
    max_epochs = max_epochs if max_epochs > 0 else None
    
    list_params = []
    datasets = []
    artifacts = []
    checkpoints = []
    metrics = []
    for ent in data["entity"].keys(): 
        if "_dataset" in ent: 
            datasets.append(ent)
            continue
        if isinstance(data["entity"][ent], list): 
            for subent in data["entity"][ent]: 
                if "yProv4ML:path" in subent.keys(): 
                    path = subent["yProv4ML:path"]
                    if path.endswith(".pth") or path.endswith(".pt"): 
                        checkpoints.append(path)
                    elif path.endswith(".zarr") or path.endswith(".csv") or path.endswith(".tsv") or path.endswith(".nc"): 
                        metrics.append(path) 
                    else:
                        artifacts.append(path)   
                    continue
        else: 
            if "yProv4ML:path" in data["entity"][ent].keys(): 
                path = data["entity"][ent]["yProv4ML:path"]
                if path.endswith(".pth") or path.endswith(".pt"): 
                    checkpoints.append(ent)
                elif path.endswith(".zarr") or path.endswith(".csv") or path.endswith(".tsv") or path.endswith(".nc"): 
                    metrics.append(path) 
                else:
                    artifacts.append(path)   
                continue

        
    list_params = None if len(list_params) == 0 else ", ".join(list_params)
    datasets = None if len(datasets) == 0 else datasets
    
    num_samples = []
    num_steps = []
    shuffle = []
    batchsize = []
    if datasets: 
        for d in datasets: 
            bs = data["entity"][d][f"{d}_stat_batch_size"]['$']
            steps = data["entity"][d][f"{d}_stat_total_steps"]['$']
            samples = data["entity"][d][f"{d}_stat_total_samples"]['$']
            sh = "shuffled" if bool(data["entity"][d][f"{d}_stat_shuffle"]) else "not shuffled"
            
            shuffle.append(sh)
            num_samples.append(samples)
            num_steps.append(steps)
            batchsize.append(bs)

    num_checkpoints = len(checkpoints)
    list_checkpoints = None if len(checkpoints) == 0 else ", ".join(checkpoints)
    num_artifacts = len(artifacts)
    list_artifacts = None if len(artifacts) == 0 else ", ".join(artifacts)
    
    model_params = None
    memory_of_model = None
    for model in checkpoints: 
        if "total_params" in data["entity"][model].keys(): 
            final_model = model
            model_params = data["entity"][model]["total_params"]["$"]
            memory_of_model = data["entity"][model]["memory_of_model"]["$"]


    contexts = set([k.split("context:Context.")[-1].replace("context:Context.", "") for k in data["entity"].keys() if "context:Context." in k])
    contexts = None if len(contexts) == 0 else ", ".join(list(contexts))

    description = \
        f"A provenance file created using the yProv4ML library (version {version}), {following_w3c}, namespace {namespace}. " +\
        f"The file contains a machine learning process {run_id}, started by user {user}, with python {pyversion}, and named {exp_name}. \n" + \
        f"The entire experiment was run in a {env} environment, " + \
        (f"and the user also attached a source code file, as the local file {source_code}. " if source_code else "") + \
        (f"The code is available at the github repository {repo}. " if repo else "") + \
        f"The experiment was scheduled at {st} and it finished at {et}, for a total of {round(endtime-starttime, 2)} seconds. \n" + \
        f"The model was trained for {max_epochs} epochs" + ("," if num_samples else ". ") + \
        (f"each epoch consisting in the passthrough of {num_samples[0]} samples, with batch size {batchsize[0]}. " if num_samples else "") + \
        (f"The model was saved {num_checkpoints} times, in the checkpoints {list_checkpoints}" + ". " if list_checkpoints else ", ") + \
        (f"with the final version being {final_model}. " if final_model else "") + \
        (f"The final model has a total of {model_params} parameters" if model_params else "") + \
        (f", and the footprint on the memory is of {round(float(memory_of_model), 2)} Mb. " if memory_of_model else "") + \
        (f"The datasets used for the training of the model are: " if datasets else "") + \
        ("; ".join([f"{d} with {s} {sh} samples, {bs} batch size and {steps} steps" for d, s, bs, steps, sh in zip(datasets, num_samples, batchsize, num_steps, shuffle)]) + ". \n" if datasets else "") + \
        (f"Additionally, {num_artifacts} artifacts are saved: {list_artifacts}." if list_artifacts else "") + \
        (f"The user saved a set of parameters from the process, including: {list_params}" if list_params else "") + \
        (f"A set of metrics have been collected from the process, in particular in the contexts {contexts}, " if contexts else "") +\
        (f"these metrics are {metrics}. " if contexts else "")
    
    return description

def main(json_path): 
    data = json.load(open(json_path))
    version = 1 if "yProv4ML" in data["prefix"].keys() else 0

    if version == 0: 
        description = handle_v0(data, version)
    else: 
        description = handle_v1(data, version)

    description = description.replace(" ,", "").replace(" .", "")

    with open(CAPTION_PATH + json_path.split("/")[-1].replace(".json", "") + ".txt", "w+") as file: 
        file.write(description)

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', default=None)  
    args = parser.parse_args()

    if args.json: 
        main(args.json)
    else: 
        files = [DATA_PATH + f for f in os.listdir(DATA_PATH)]
        for file in files: 
            main(file)