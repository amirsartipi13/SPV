import matplotlib.pyplot as plt

def visualize(results, step, net):
  epochs = list(results.keys())

  num_metrics = len(results[1])
  num_plots_per_row = num_metrics // 2 + num_metrics % 2

  fig, axs = plt.subplots(2, num_plots_per_row, figsize=(15, 10))

  axs = axs.flatten()

  for i, metric in enumerate(results[1].keys()):
      metric_values = [result[metric] for result in results.values()]
      axs[i].plot(epochs, metric_values, marker='o')
      axs[i].set_xlabel('Epoch')
      axs[i].set_ylabel(metric.capitalize())
      axs[i].set_title(f'{metric.capitalize()} values during epochs', fontsize=8)
      axs[i].grid(True)

  if num_metrics % 2 != 0:
      axs[-1].axis('off')

  plt.tight_layout()
  plt.savefig(os.path.join(step, net))
  plt.show()


import math

def aggregate_dict(dictionary, group_size):
    result = {}
    keys = list(dictionary.keys())
    num_groups = math.ceil(len(keys) / group_size)

    for i in range(num_groups):
        start_index = i * group_size
        end_index = start_index + group_size
        group_keys = keys[start_index:end_index]
        group_values = [dictionary[key] for key in group_keys]
        aggregated_value = aggregate_values(group_values)
        result[i + 1] = aggregated_value

    return result


def aggregate_values(values):
    aggregated_value = {}
    for value in values:
        for key, sub_value in value.items():
            if key not in aggregated_value:
                aggregated_value[key] = 0
            aggregated_value[key] += sub_value
    return {key: value / len(values) for key, value in aggregated_value.items()}

