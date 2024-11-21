from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

results = [
    {"learning_rate": 1e-2, "batch_size": 16, "repeat_dataset": 1, "accuracy": 0.825},
    {"learning_rate": 1e-3, "batch_size": 16, "repeat_dataset": 1, "accuracy": 0.907},
    {"learning_rate": 1e-4, "batch_size": 16, "repeat_dataset": 1, "accuracy": 0.863},
    {"learning_rate": 1e-2, "batch_size": 16, "repeat_dataset": 2, "accuracy": 0.920},
    {"learning_rate": 1e-3, "batch_size": 16, "repeat_dataset": 2, "accuracy": 0.890},
    {"learning_rate": 1e-4, "batch_size": 16, "repeat_dataset": 2, "accuracy": 0.915},
    {"learning_rate": 1e-2, "batch_size": 16, "repeat_dataset": 3, "accuracy": 0.833},
    {"learning_rate": 1e-3, "batch_size": 16, "repeat_dataset": 3, "accuracy": 0.935},
    {"learning_rate": 1e-4, "batch_size": 16, "repeat_dataset": 3, "accuracy": 0.860},
    {"learning_rate": 1e-2, "batch_size": 32, "repeat_dataset": 1, "accuracy": 0.932},
    {"learning_rate": 1e-3, "batch_size": 32, "repeat_dataset": 1, "accuracy": 0.877},
    {"learning_rate": 1e-4, "batch_size": 32, "repeat_dataset": 1, "accuracy": 0.907},
    {"learning_rate": 1e-2, "batch_size": 32, "repeat_dataset": 2, "accuracy": 0.870},
    {"learning_rate": 1e-3, "batch_size": 32, "repeat_dataset": 2, "accuracy": 0.900},
    {"learning_rate": 1e-4, "batch_size": 32, "repeat_dataset": 2, "accuracy": 0.907},
    {"learning_rate": 1e-2, "batch_size": 32, "repeat_dataset": 3, "accuracy": 0.865},
    {"learning_rate": 1e-3, "batch_size": 32, "repeat_dataset": 3, "accuracy": 0.932},
    {"learning_rate": 1e-4, "batch_size": 32, "repeat_dataset": 3, "accuracy": 0.830},
]

# Extract hyperparams in order
learning_rates = [r['learning_rate'] for r in results]
batch_sizes = [r['batch_size'] for r in results]
repeat_datasets = [r['repeat_dataset'] for r in results]
accuracies = [r['accuracy'] for r in results]

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(learning_rates, batch_sizes, repeat_datasets, c=accuracies, cmap="viridis", s=100)

top_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:1]

for result in top_results:
    learning_rate = result['learning_rate']
    batch_size = result['batch_size']
    repeat_dataset = result['repeat_dataset']
    accuracy = result['accuracy']
    label = f"(LR:{learning_rate}, BS:{batch_size}, RD:{repeat_dataset}, Acc:{accuracy})"
    ax.text(learning_rate, batch_size, repeat_dataset, label, fontsize=8)

# Add labels and title
ax.set_title("Hyperparameter Grid Search Validation Accuracy")
ax.set_xlabel("Learning Rate (LR)")
ax.set_ylabel("Batch Size (BS)")
ax.set_zlabel("Repeat Dataset (RD)")
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label("Accuracy (Acc)")

plt.savefig("grid_search_visualization.png", dpi=300, bbox_inches="tight")
plt.show()