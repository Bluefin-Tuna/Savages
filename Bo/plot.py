import numpy as np
import matplotlib.pyplot as plt

# Define the data for each title and the associated accuracies at each number of options
data = {
    "Phaedo": {2: [0.91, 0.88], 6: [0.85, 0.80], 10: [0.85, 0.75]},
    "Only Natural- Gender, Knowledge, and Humankind, by Louise Antony": {2: [0.75, 0.81], 6: [0.53, 0.48], 10: [0.47, 0.23]},
    "The World as Will and Idea (Vol. 3 of 3)": {2: [0.91, 0.94], 6: [0.70, 0.59], 10: [0.62, 0.41]},
    "Kant's Critique of Judgement": {2: [0.95, 0.87], 6: [0.77, 0.62], 10: [0.66, 0.54]},
    "The roots and routes of the epistemology of ignorance": {2: [1.0, 0.81], 6: [0.90, 0.51], 10: [0.67, 0.32]},
    "The Birth of Tragedy; or, Hellenism and Pessimism": {2: [0.84, 0.81], 6: [0.71, 0.58], 10: [0.54, 0.43]},
    "Commentary on Mark Richard, Meanings as Species": {2: [0.96, 0.94], 6: [0.92, 0.83], 10: [0.88, 0.77]},
    "The Essence of Christianity": {2: [0.89, 0.84], 6: [0.66, 0.48], 10: [0.56, 0.35]},
    "Aristotle on the art of poetry": {2: [0.83, 0.74], 6: [0.71, 0.65], 10: [0.63, 0.50]},
    "The Logic of Hegel": {2: [0.86, 0.80], 6: [0.66, 0.50], 10: [0.52, 0.39]},
    "I've been thinking": {2: [0.96, 0.86], 6: [0.83, 0.64], 10: [0.76, 0.53]}
}

# Define hex colors for the titles
hex_colors = {
    "Phaedo": "#1e76b4",
    "Only Natural- Gender, Knowledge, and Humankind, by Louise Antony": "#ff7f0e",
    "The World as Will and Idea (Vol. 3 of 3)": "#2da02d",
    "Kant's Critique of Judgement": "#d62728",
    "The roots and routes of the epistemology of ignorance": "#9566bd",
    "The Birth of Tragedy; or, Hellenism and Pessimism": "#8d574b",
    "Commentary on Mark Richard, Meanings as Species": "#e377c2",
    "The Essence of Christianity": "#7e7f7e",
    "Aristotle on the art of poetry": "#bcbc23",
    "The Logic of Hegel": "#16bece",
    "I've been thinking": "#1e76b4"
}
num_choices_of_interest = [2, 6, 10]

# Calculate mean and standard deviation for each title and number of options
accuracy_stats = {}
for title, records in data.items():
    stats = {}
    for num_choice in num_choices_of_interest:
        accuracies = records.get(num_choice, [])
        if accuracies:
            stats[num_choice] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            }
    accuracy_stats[title] = stats

# Plotting setup
plt.figure(figsize=(10, 6))
for title, stats in accuracy_stats.items():
    x = num_choices_of_interest
    y = [stats[nc]['mean'] for nc in num_choices_of_interest if nc in stats]
    yerr = [stats[nc]['std'] for nc in num_choices_of_interest if nc in stats]
    plt.errorbar(x, y, yerr=yerr, label=title, color=hex_colors[title], fmt='-o', capsize=5)

# Customizing the plot
plt.xticks(range(2, 11))  # Ticks from 2 to 10
plt.xlabel('Number of Options')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Options on GPT3.5')
plt.grid(False)  # No grid lines
plt.box(True)  # Keep frame

# Save and show the plot
plt.tight_layout()
plt.savefig('accuracy_vs_num_options.png')
plt.show()
