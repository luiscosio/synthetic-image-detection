import matplotlib.pyplot as plt

# Detector accuracies for each dataset
datasets = ['DALLE2', 'DALLE3', 'Midjourney', 'MSCOCO2014_filtered_val', 'VQGAN']
accuracies = {
    'CNNDetector_p0.1_crop': [0.24, 0.0063723608445297505, 0.065, 0.988, 0.096],
    'CNNDetector_p0.1': [0.24, 0.0063723608445297505, 0.065, 0.988, 0.096],
    'CNNDetector_p0.5_crop': [0.057, 0.004145873320537428, 0.013, 0.998, 0.032],
    'CNNDetector_p0.5': [0.057, 0.004145873320537428, 0.013, 0.998, 0.032],
    'EnsembleDetector': [0.001, 0.0014587332053742803, 0.013, 0.988, 0.442],
    'EnsembleDetector_crop': [0.001, 0.0014587332053742803, 0.013, 0.988, 0.442],
    'CLIPDetector_crop': [0.655, 0.06694817658349328, 0.157, 0.973, 0.789]
}

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(datasets))  # the label locations
width = 0.1  # the width of the bars

for i, (detector, acc) in enumerate(accuracies.items()):
    ax.bar([x_ + (i-3)*width for x_ in x], acc, width, label=detector)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Datasets')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Detectors across Datasets')
ax.set_xticks([x + width for x in x])
ax.set_xticklabels(datasets)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
