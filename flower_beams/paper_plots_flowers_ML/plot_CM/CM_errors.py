import torch


def calculate_95_confidence_interval_torch(num_classes, samples_per_class, accuracy_percent):
    """
    Calculate the 95% confidence interval error (margin of error) for a binomial proportion
    using PyTorch operations. The inputs can be scalars or tensors, and the operations will be
    applied element-wise.

    Parameters:
      num_classes (int, float, or torch.Tensor): Number of classes.
      samples_per_class (int, float, or torch.Tensor): Number of samples per class.
      accuracy_percent (int, float, or torch.Tensor): Overall accuracy in percent (e.g., 70 for 70%).

    Returns:
      margin_of_error_percent (torch.Tensor): The margin of error (in percentage points).
      total_samples (torch.Tensor): Total number of test samples.
    """
    # Ensure inputs are torch tensors of type float32 for consistency
    if not torch.is_tensor(num_classes):
        num_classes = torch.tensor(num_classes, dtype=torch.float32)
    else:
        num_classes = num_classes.float()

    if not torch.is_tensor(samples_per_class):
        samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)
    else:
        samples_per_class = samples_per_class.float()

    if not torch.is_tensor(accuracy_percent):
        accuracy_percent = torch.tensor(accuracy_percent, dtype=torch.float32)
    else:
        accuracy_percent = accuracy_percent.float()

    # Convert accuracy from percentage to a proportion (0 to 1)
    p = accuracy_percent / 100.0

    # Compute the total number of samples (element-wise multiplication if arrays)
    total_samples = num_classes * samples_per_class

    # Compute the standard error for the binomial proportion:
    #   SE = sqrt( p * (1 - p) / total_samples )
    se = torch.sqrt(p * (1 - p) / total_samples)

    # For a 95% confidence interval, the margin of error is:
    #   margin_error = 1.96 * SE
    margin_of_error = 1.96 * se

    # Convert the margin of error back to percentage points
    margin_of_error_percent = margin_of_error * 100

    return margin_of_error_percent, total_samples


# === Example Usage ===
# Suppose you have three experiments with the following parameters:
num_classes = [11, 11, 11, 11,
               11, 11, 11, 11,
               11, 11, 11, 11,
               11, 11, 11, 11,
               11, 11, 11, 11,
               81, 81, 81, 81,
               81, 81, 81, 81, ]  # For each experiment, 11 classes.
samples_per_class = [100, 100, 100, 300,
                     100, 100, 100, 300,
                     100, 100, 100, 300,
                     100, 100, 100, 300,
                     100, 100, 100, 300,
                     50, 50, 50, 150,
                     10, 10, 10, 30

                     ]  # For each experiment, 300 samples per class.
accuracy_percent = [50.3, 32.4, 21.6, 34.8,
                    72.1, 42.8, 30.6, 48.5,
                    79.6, 59.1, 49.1, 62.6,
                    79.7, 62.0, 53.8, 65.2,
                    97.7, 87.1, 75.4, 86.7,
                    92.7, 73.6, 59.0, 75.1,
                    91.2, 60.7, 34.6, 62.2
                    ]  # Different overall accuracies for each experiment.

margin_error_percent, total_samples = calculate_95_confidence_interval_torch(
    num_classes, samples_per_class, accuracy_percent
)

print("Total samples:", total_samples)
print("95% confidence interval errors (±% points):", margin_error_percent)
