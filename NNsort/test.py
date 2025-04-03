import torch
import time
import numpy as np
from NNsort import RecursiveSortNet, RandomDataset, DataLoader

def calculate_accuracy(pred, label):
    """Calculate the percentage of correctly ordered elements."""
    pred_rounded = torch.round(pred)
    correct = torch.sum(pred_rounded == label)
    accuracy = correct.item() / label.numel()
    return accuracy

def test_model():
    input_size = 1024
    recursion_depth = 2
    model = RecursiveSortNet(input_size, recursion_depth)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()  # Set the model to evaluation mode

    # Create a dataloader for test data
    test_dataset = RandomDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Benchmarking variables
    num_samples = 100
    nn_times = []
    python_sort_times = []
    numpy_sort_times = []
    accuracies = []

    # Iterate over test data and make predictions
    with torch.no_grad():
        for i, (input, label) in enumerate(test_dataloader):
            if i >= num_samples:
                break

            # Benchmark Neural Network sorting time
            start_time = time.time()
            output = model(input)
            nn_times.append(time.time() - start_time)

            # Calculate accuracy
            accuracy = calculate_accuracy(output, label)
            accuracies.append(accuracy)

            # Convert input to a list for Python's built-in sort
            input_list = input.numpy().tolist()

            # Benchmark Python's built-in sorting time
            start_time = time.time()
            sorted_list = sorted(input_list[0])
            python_sort_times.append(time.time() - start_time)

            # Benchmark NumPy's sorting time
            start_time = time.time()
            sorted_array = np.sort(input.numpy())
            numpy_sort_times.append(time.time() - start_time)

            # Print the results for one example
            if i == 0:
                print('Input: ', input)
                print('Label: ', label)
                print('Prediction: ', output)
                print('Rounded Prediction: ', torch.round(output))
                print('Python Sorted: ', sorted_list)
                print('NumPy Sorted: ', sorted_array)
                print(f'Accuracy: {accuracy * 100:.2f}%')

    # Calculate average times and accuracy
    avg_nn_time = sum(nn_times) / len(nn_times)
    avg_python_sort_time = sum(python_sort_times) / len(python_sort_times)
    avg_numpy_sort_time = sum(numpy_sort_times) / len(numpy_sort_times)
    avg_accuracy = sum(accuracies) / len(accuracies)

    # Print benchmarking results
    print(f'Average Neural Network sorting time: {avg_nn_time:.6f} seconds')
    print(f'Average Python built-in sorting time: {avg_python_sort_time:.6f} seconds')
    print(f'Average NumPy sorting time: {avg_numpy_sort_time:.6f} seconds')
    print(f'Average Accuracy: {avg_accuracy * 100:.2f}%')

if __name__ == '__main__':
    test_model()
