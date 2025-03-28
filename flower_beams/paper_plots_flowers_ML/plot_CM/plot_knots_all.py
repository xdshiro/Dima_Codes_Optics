from flower_beams.paper_plots_flowers_ML.plots_functions_general import *


def calculate_accuracy(cm):
    # Sum of diagonal elements (correct predictions)
    correct_predictions = np.trace(cm)
    # Total number of samples
    total_predictions = np.sum(cm)
    # Accuracy calculation
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy
# List of files and their corresponding labels
files_dots = [
    {"path": "data\\hopfs_dots_005.npz", "label": f"Shape-based CNN. $\\sigma_R^2=0.05$."},
    {"path": "data\\hopfs_dots_015.npz", "label": f"Shape-based CNN. $\\sigma_R^2=0.15$."},
    {"path": "data\\hopfs_dots_025.npz", "label": f"Shape-based CNN. $\\sigma_R^2=0.25$."},
    # {"path": "data\\hopfs_dots_all.npz", "label": f"Shape-based CNN, Knots, Combined,"},

]
files_spectre_FCNN = [
    {"path": "data\\hopfs_spectr_fc_005.npz", "label": f"Spectrum-based FCNN. $\\sigma_R^2=0.05$."},
    {"path": "data\\hopfs_spectr_fc_015.npz", "label": f"Spectrum-based FCNN. $\\sigma_R^2=0.15$."},
    {"path": "data\\hopfs_spectr_fc_025.npz", "label": f"Spectrum-based FCNN. $\\sigma_R^2=0.25$."},
    # {"path": "data\\hopfs_spectr_fc_all.npz", "label": f"Spectre-based FCNN, Knots, Combined,"},

]
files_spectre_CNN = [
    # {"path": "data\\hopfs_spectr_cnn_005.npz", "label": f"Spectre-based CNN. $\\sigma_R^2=0.05$."},
    # {"path": "data\\hopfs_spectr_cnn_015.npz", "label": f"Spectre-based CNN. $\\sigma_R^2=0.15$."},
    # {"path": "data\\hopfs_spectr_cnn_025.npz", "label": f"Spectre-based CNN. $\\sigma_R^2=0.25$."},
    {"path": "data\\hopfs_spectr_cnn_all.npz", "label": f"Spectrum-based CNN."},

]
files_main = [
    {"path": "data\\hopfs_dots_all.npz", "label": f"Shape-based CNN."},
    {"path": "data\\hopfs_spectr_fc_all.npz", "label": f"Spectrum-based FCNN."},
]
files = files_dots
files = files_main
files = files_spectre_FCNN
files = files_spectre_CNN
# Class labels
class_labels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'T1', 'T2']

# Loop through files and plot confusion matrices
for file in files:
    data = np.load(file["path"])
    cm = data['cm']
    accuracy = calculate_accuracy(cm)
    label = file["label"] + f" Total Accuracy={accuracy * 100:0.3}%"
    plot_confusion_matrix(cm, class_labels, label=label)