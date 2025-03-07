# yProv4MLProvenanceDataset

## Script Folder

If it's convenient for you, we have developed some python functions to help in extracting data from provenance files. 
The "script" folder contains helper functions which I usually use to get data. 

```python
import json
from scripts.prov_getters import *

data = json.load(open(path_to_prov_json)) 
loss = get_metric(data, "Loss_Context.TRAINING")
```

This function will return a pandas DataFrame with three columns: 
- epoch: the metric_epoch_list in the provenance graph
- timestamp: the metric_timestamp_list in the provenance graph
- value: the actual metric value you chose

There are other parameters which can be passed to the get_metric() function, the doc is [here](https://hpci-lab.github.io/yProv4ML.github.io/prov_getters.html). 

## Important

The issue with duplicated epochs ("epoch_1.0" and "epoch_1") is just visual when converting the json in graph format, but when analyzing the JSON everything should be correct. 

I'm also uploading some runs which were run on multi-process environments, so you may see epoch skips (e.g. provgraph_3DGAN where we go from 1, 5, 9, 13...), this is normal, some other processes will have handled the missing epochs, but we kept track of just one. 
If this is a problem, I can avoid uploading these kind of provenance files, just let me know. 

### Variables Description

- Loss: loss function over time, which represents the error between predicted and actual values in training

- gpu_memory_usage: Amount of GPU memory (VRAM) currently being used
- gpu_usage: GPU utilization percentage, indicating how much of the GPU’s processing power is in use
- gpu_temperature: Temperature of the GPU in degrees Celsius
- gpu_power: The current power consumption of the GPU in watts
- gpu_energy: The total energy consumed by the GPU over time (measured in joules or watt-hours)

- cpu_memory_usage: Amount of RAM being used by the CPU
- cpu_usage: Percentage of CPU utilization (same as gpu)
- cpu_temperature: Temperature of the CPU in degrees Celsius
- cpu_power: The current power consumption of the CPU in watts
- cpu_energy: The total energy consumed by the CPU over time (measured in joules or watt-hours)

- memory_usage: Total system memory (RAM) usage
- ram_power: Power consumption of the RAM in watts
- disc_usage: Storage space currently in use on the disk

- emissions: The total amount of CO₂ emissions (measured in Co2eq) generated during model execution
- emissions_rate: The rate of CO₂ emissions over time
- energy_consumed: The total energy consumed by the system during execution (CPU + GPU + mem etc.)

### Parameters Description

- total_params: The total number of trainable and non-trainable parameters in the model
- memory_of_model: The amount of memory required to store the model parameters
- total_memory_load_of_model: The overall memory footprint of the model, including activations and intermediate computations
- optimizer: The optimization algorithm used to update model weights (e.g., Adam, SGD)
- loss_fn: The loss function used to measure prediction error (e.g., CrossEntropy, MSE)
- execution_start_time: The timestamp when model execution started
- execution_end_time: The timestamp when model execution ended

- dataset_batch_size: The number of samples processed per batch during training/inference (grouped samples)
- dataset_total_samples: The total number of data samples in the dataset
- dataset_total_steps: The total number of training/inference steps taken

### Difference between epochs / samples / batches

- Epoch means one pass over the full training set, you'll often have several epochs in a training execution
- You can think of a sample as a single image of a cat or dog
- In ML you often pass several samples together in the model, this grouping is called batch