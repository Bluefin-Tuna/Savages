import re
from collections import defaultdict
import math
import matplotlib.pyplot as plt

results = defaultdict(lambda: defaultdict(list))
current_num_options = None

output_path = "output.log" # results of llama2-7b
output_path = 'output_13b.log' # results of llama2-13b
# output_path = 'output_llama3.log' # results of llama3-8b

with open(output_path, "r") as file:
    for line in file:
        num_options_match = re.search(r"num_options': (\d+)", line)
        if num_options_match:
            current_num_options = int(num_options_match.group(1))
        else:
            match = re.match(r"Author: (.*?), Title: (.*?), Accuracy: (.*?)\n", line)
            if match and current_num_options in [2,6,10]:
                author, title, accuracy = match.groups()
                accuracy = float(accuracy)
                random_guess_accuracy = 1 / current_num_options
                adjusted_accuracy = accuracy - random_guess_accuracy
                results[current_num_options][title].append(accuracy)

for num_options, book_results in results.items():
    print(f"num_options: {num_options}")
    for title, adjusted_accuracies in book_results.items():
        avg_adjusted_accuracy = sum(adjusted_accuracies) / len(adjusted_accuracies)
        std_dev = math.sqrt(sum((x - avg_adjusted_accuracy) ** 2 for x in adjusted_accuracies) / len(adjusted_accuracies))
        print(f"Title: {title}, Average Adjusted Accuracy: {avg_adjusted_accuracy:.2f}, Standard Deviation: {std_dev:.2f}")
    print()
    
# Data for plotting
num_options_list = sorted(results.keys())
data_for_plotting = {title: ([], []) for title in results[next(iter(results))].keys()}  # Initialize with titles

# Calculate averages and standard deviations for each title and num_options
for num_options in num_options_list:
    for title, accuracies in results[num_options].items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        std_dev = math.sqrt(sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies))
        data_for_plotting[title][0].append(avg_accuracy)
        data_for_plotting[title][1].append(std_dev)

# Plotting the main figure without legend
fig1, ax1 = plt.subplots()
for title, (averages, std_devs) in data_for_plotting.items():
    ax1.errorbar(num_options_list, averages, yerr=std_devs, label=title, capsize=5)
ax1.set_xlabel('Number of Options')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs. Number of Options on Llama2-13b')
plt.show()
plt.savefig('accuracy_vs_num_options.png')
plt.close()

# Creating a separate figure for the legend
fig2, ax2 = plt.subplots()
handles, labels = ax1.get_legend_handles_labels()
figlegend = plt.figure(figsize=(6, 6))
plt.figlegend(handles, labels, loc='upper left')
plt.show()
plt.savefig('legend.png')
plt.close()
