import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.special import expit as sigmoid  # Sigmoid function

# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\ShallowCNN_lfcc_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for ShallowCNN_lfcc_I')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\ShallowCNN_lfcc_I.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\ShallowCNN_lfcc_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for ShallowCNN_lfcc_O')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\ShallowCNN_lfcc_O.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\ShallowCNN_mfcc_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for ShallowCNN_mfcc_I')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\ShallowCNN_mfcc_I.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\SimpleLSTM_lfcc_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SimpleLSTM_lfcc_I')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\SimpleLSTM_lfcc_I.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\SimpleLSTM_lfcc_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SimpleLSTM_lfcc_O')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\SimpleLSTM_lfcc_O.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\SimpleLSTM_mfcc_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SimpleLSTM_mfcc_I')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\SimpleLSTM_mfcc_I.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\TSSD_wave_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for TSSD_wave_I')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\TSSD_wave_I.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\TSSD_wave_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for TSSD_wave_O')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\TSSD_wave_O.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\WaveLSTM_wave_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for WaveLSTM_wave_I')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\WaveLSTM_wave_I.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\WaveLSTM_wave_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for WaveLSTM_wave_O')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\WaveLSTM_wave_O.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\WaveRNN_wave_I\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for WaveRNN_wave_I')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\WaveRNN_wave_I.png" 
plt.savefig(save_path)
plt.show()







# Load predictions from best_pred.json file
file_path = r"E:\AI Generated Audio Detection using Machine Learning\saved\WaveRNN_wave_O\best_pred.json"
with open(file_path, 'r') as file:
    data = json.load(file)

test_labels = np.array(data['y_true'])
model_scores = np.array(data['y_pred'])  # Assuming you have model output scores

# Convert model scores to probabilities using sigmoid function
predicted_probs = sigmoid(model_scores)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for WaveRNN_wave_O')
plt.legend(loc="lower right")
save_path = r"E:\AI Generated Audio Detection using Machine Learning\ROC AUC\WaveRNN_wave_O.png" 
plt.savefig(save_path)
plt.show()