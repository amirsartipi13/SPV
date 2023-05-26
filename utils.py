import matplotlib.pyplot as plt

def visualize(results):
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

  plt.show()