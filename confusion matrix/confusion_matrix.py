import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



# ShallowCNN_lfcc_I
# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\ShallowCNN_lfcc_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for ShallowCNN_LFCC_I')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\ShallowCNN_LFCC_I.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()



# ShallowCNN_lfcc_O
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\ShallowCNN_lfcc_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for ShallowCNN_LFCC_O')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\ShallowCNN_LFCC_O.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()


# ShallowCNN_mfcc_I
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\ShallowCNN_mfcc_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for ShallowCNN_MFCC_I')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
        
save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\ShallowCNN_MFCC_I.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()


# SimpleLSTM_lfcc_I
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\SimpleLSTM_lfcc_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SimpleLSTM_lfcc_I')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\SimpleLSTM_lfcc_I.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()


# SimpleLSTM_lfcc_O
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\SimpleLSTM_lfcc_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SimpleLSTM_LFCC_O')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\SimpleLSTM_LFCC_O.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()



# SimpleLSTM_mfcc_I
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\SimpleLSTM_mfcc_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SimpleLSTM_MFCC_I')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\SimpleLSTM_MFCC_I.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()



# TSSD_wave_I
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\TSSD_wave_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for TSSD_wave_I')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\TSSD_wave_I.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()


# TSSD_wave_O
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\TSSD_wave_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for TSSD_wave_O')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\TSSD_wave_O.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()



# WaveLSTM_wave_I
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\WaveLSTM_wave_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for WaveLSTM_wave_I')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\WaveLSTM_wave_I.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()



# WaveLSTM_wave_O
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\WaveLSTM_wave_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for WaveLSTM_wave_O')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\WaveLSTM_wave_O.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()


# WaveRNN_wave_I
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\WaveRNN_wave_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for WaveRNN_wave_I')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\WaveRNN_wave_I.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()


# WaveRNN_wave_O
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\WaveRNN_wave_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
predicted_labels = np.array(data['y_pred'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for WaveRNN_wave_O')
plt.colorbar()
plt.xticks(np.arange(2), ['Real', 'Fake'])
plt.yticks(np.arange(2), ['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

save_path = r"E:\AI Generated Audio Detection using Machine Learning\confusion matrix\WaveRNN_wave_O.png" 
plt.savefig(save_path)

plt.tight_layout()
plt.show()