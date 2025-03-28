from flower_beams.paper_plots_flowers_ML.plots_functions_general import *
import itertools

def calculate_accuracy(cm):
    # Sum of diagonal elements (correct predictions)
    correct_predictions = np.trace(cm)
    # Total number of samples
    total_predictions = np.sum(cm)
    # Accuracy calculation
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy
# List of files and their corresponding labels
files = [
    {"path": "data\\foils_dots_part_005.npz", "label": f"Shape-based CNN. $\\sigma_R^2=0.05$."},
    {"path": "data\\foils_dots_part_015.npz", "label": f"Shape-based CNN. $\\sigma_R^2=0.15$."},
    {"path": "data\\foils_dots_part_025.npz", "label": f"Shape-based CNN. $\\sigma_R^2=0.25$."},
    {"path": "data\\foils_dots_part_all.npz", "label": f"Shape-based CNN."},

]
# files = [
# {"path": "data\\foils_dots_part_all.npz", "label": f"Shape-based CNN."},
# ]
# Class labels
class_labels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'T1', 'T2']
class_labels = list(itertools.product(range(3), repeat=4))
class_labels = [''.join(map(str, label)) for label in class_labels]
# Loop through files and plot confusion matrices
for file in files:
    data = np.load(file["path"])
    cm = data['cm']
    accuracy = calculate_accuracy(cm)
    label = file["label"] + f" Total Accuracy={accuracy * 100:0.3}%"
    plot_confusion_matrix_big(cm, class_labels, label=label)