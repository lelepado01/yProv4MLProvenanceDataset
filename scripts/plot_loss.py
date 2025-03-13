
from prov_getters import * 
import json
import matplotlib.pyplot as plt

data = json.load(open("./data/provgraph_transformer_finetuning.json"))

# print(get_metrics(data))

gpu = get_metric(data, "gpu_power_Context.TRAINING")["value"]
loss = get_metric(data, "Loss_Context.TRAINING")["value"]

plt.plot(loss)
# plt.plot(gpu)
plt.show()