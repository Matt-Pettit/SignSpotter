import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import math
from alive_progress import alive_bar
import numpy as np
from PIL import Image
#for confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
imagesize = 28
import string


#For jaccard_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score, recall_score, f1_score



labels = [letter for letter in string.ascii_uppercase if letter not in ['I', 'J']]
# label_dictionary = {0: '+', 1: '-', 2: '=', 3: 'division', 4: 'infinity', 5: 'integral', 6: 'pi', 7: 'sum', 8: 'theta', 9: 'times'}
# label_dictionary = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
label_dictionary = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

def calculate_jaccard_scores_per_class(y_test, y_pred):
    jaccard_scores = jaccard_score(y_test, y_pred, average=None)
    return jaccard_scores

def load_single_image(image_path):

    image = np.array(Image.open(image_path).convert('L'))
    return image.flatten()

def most_common(lst):
    # Return the mode of the set (ie the most common)
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

def Minkowsk(point, data,dimension):
    # M inkowsk distance between points a & data
    return np.sum((np.absolute(point - data))**dimension, axis=1)**(1/dimension)

def citblock(point, data):
    # Cityblock distance between points a & data
    return np.sum(np.absolute(point - data), axis=1)

def plot_confusion_matrix(cm, k, labels):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels, ax=ax)

    # Set the title and axis labels
    ax.set_title(f'Confusion Matrix (K={k})')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    # Adjust the layout to prevent labels from overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

def calculate_metrics(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    return f1

def knn_predict(X_train, y_train, X_test, k,p):
    neighbors = []
    # If length of test data is greater than 1, show a progres bar for the user
    if len(X_test) > 1:
        with alive_bar(len(X_test)) as bar:
            for x in X_test:
                # Calculate Distances
                distances = Minkowsk(x, X_train,p)
                # Append K Nearest Neighbours Based On Lowest Distances
                y_sorted = [y for _, y in sorted(zip(distances, y_train))]
                neighbors.append(y_sorted[:k])
                # Update Progress Bar
                bar()
    else:       
        for x in X_test:
            # Calculate Distances
            distances = Minkowsk(x, X_train,p)
            # Append K Nearest Neighbours Based On Lowest Distances
            y_sorted = [y for _, y in sorted(zip(distances, y_train))]
            neighbors.append(y_sorted[:k])

    return list(map(most_common, neighbors))





    return list(map(most_common, neighbors))


def knn_random_examples(X_train, y_train, X_test, y_test, k, numExamples,p):
    #print(len(X_test))
    for i in range(numExamples):
        idx = random.randint(0, len(X_test))  # Index of the random example
        print(f"Actual: {label_dictionary[y_test[idx]]}")
        x_test_example = X_test[idx].reshape(1, -1)  # Reshape to have a single row
        y_pred = knn_predict(X_train, y_train, x_test_example, k, dist_metric,p)
        print(f"Predicted: {label_dictionary[y_pred[0]]}")
        print("=======================")

def knn_random_examplesPic(X_train, y_train, X_test, y_test, k, numExamples,p):
    num_rows = math.ceil(numExamples / 3)
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 3 * num_rows))

    for i, idx in enumerate(random.sample(range(5, len(X_test)), numExamples)):
        row, col = i // 3, i % 3
        print(f"Actual: {label_dictionary[y_test[idx]]}")
        x_test_example = X_test[idx].reshape(1, -1)  # Reshape to have a single row
        y_pred = knn_predict(X_train, y_train, x_test_example, k,p)
        print(f"Predicted: {label_dictionary[y_pred[0]]}")

        # Visualize the image
        image = X_test[idx].reshape(28,28)  # Reshape to the correct shape for visualization
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'Actual: {label_dictionary[y_test[idx]]}\nPredicted: {label_dictionary[y_pred[0]]}')
        axes[row, col].axis('off')
        print("=======================")

    print(f"Actual: Your File")
    y_pred = knn_predict(X_train, y_train, [load_single_image("L.jpg")], k,p)
    print(f"Predicted: {label_dictionary[y_pred[0]]}")

    # Visualize the image
    image = X_test[idx].reshape(imagesize,imagesize)  # Reshape to the correct shape for visualization
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(f'Actual: {label_dictionary[y_test[idx]]}\nPredicted: {label_dictionary[y_pred[0]]}')
    axes[row, col].axis('off')
    print("=======================")

    plt.tight_layout()
    plt.show()

def knn_evaluate(X_train, y_train, X_test, y_test, k,p):
    y_pred = knn_predict(X_train, y_train, X_test, k,p)
    accuracy = sum(y_pred == y_test) / len(y_test)
    return accuracy, y_pred

# Load the new datasets
X_train = np.load('X_train.npy')
print("=====DATASET SIZES=====")
print("X_train Size:", len(X_train))
X_test = np.load('X_test.npy')
print("X_test Size:", len(X_test))
y_train = np.load('y_train.npy')
print("y_train Size:", len(y_train))
y_test = np.load('y_test.npy')
print("y_test Size:", len(y_test))

# Preprocess data
ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

# Test one thing
k = 20
p = 2
print("=======================")
print("Trying Random Examples:")
print("=======================")
knn_random_examplesPic(X_train, y_train, X_test, y_test, k, 6,p)

accuracies = []
y_preds = []
ks = []  # Initialize an empty list for k values
jaccard_scores_list = [] 
f1_scores_list = [] 

print("Benchmarking System:")
print("Training Data Size:", len(X_train))
print("Test Data Size:", len(X_test))
print("=======================")


# p = 1, p = 2 , maybe p = inf
p = 2
for k in [20,30]:
    print("K Value:", k)
    
    accuracy, y_pred = knn_evaluate(X_train, y_train, X_test, y_test, k,p)
    print("Accuracy:", accuracy)
    accuracies.append(accuracy)
    y_preds.append(y_pred)
    ks.append(k)  # Append the current k value to the ks list



    jaccard_scores = calculate_jaccard_scores_per_class(y_test, y_pred)
    jaccard_scores_list.append(jaccard_scores)  # Append the scores to the list

    # Calculate the average Jaccard similarity coefficient score
    avg_jaccard_score = np.mean(jaccard_scores) 
    print(f"Average Jaccard similarity coefficient score: {avg_jaccard_score}%")


    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix:")
    #print(cm)

    calculate_metrics(y_test, y_pred)

    #   f1_scores_list.append(calculate_metrics(y_test, y_pred))
    # Plot the confusion matrix
    plot_confusion_matrix(cm, k, labels)

# Visualize accuracy vs. k
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(ks, accuracies, 'r-', label='Accuracy')
ax1.set_xlabel("k")
ax1.set_ylabel("Accuracy")
ax1.set_title("Performance of SignSpotter")
ax1.legend(loc='upper left')

# # Calculate average Jaccard scores
# avg_jaccard_scores = [np.mean(scores)  for scores in jaccard_scores_list]

# ax2 = ax1.twinx()  # Create a second y-axis
# ax2.plot(ks, avg_jaccard_scores, 'b--', label='Average Jaccard Score')
# ax2.set_ylabel("Average Jaccard Similarity Coefficient Score (%)")
# ax2.legend(loc='upper right')

# # Visualize accuracy vs. F1 Scores
# fig, ax1 = plt.subplots(figsize=(12, 6))
# ax1.plot(ks, accuracies, 'r-', label='Accuracy')
# ax1.set_xlabel("k")
# ax1.set_ylabel("Accuracy")
# ax1.set_title("Performance of SymbolClassifier")
# ax1.legend(loc='upper left')

# Calculate average F1 scores

# ax2 = ax1.twinx()  # Create a second y-axis
# ax2.plot(ks, f1_scores_list, 'b--', label='F1 Score')
# ax2.set_ylabel("F1 Score (%)")
# ax2.legend(loc='upper right')





# Plot Jaccard scores per class
fig, ax = plt.subplots(figsize=(15, 8))
x = np.arange(len(labels))
bar_width = 0.15
colors = ['black','navy', 'royalblue', 'deepskyblue', 'lightblue']

for i, scores in enumerate(jaccard_scores_list):
    offset = i * bar_width
    ax.bar(x + offset, scores, width=bar_width, label=f"K={ks[i]}", color=colors[i])
    ax.set_xticks(x + bar_width * (len(ks) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel("Classes")
    ax.set_ylabel("Jaccard Similarity Coefficient Score")
    ax.set_title("Jaccard Similarity Coefficient Scores per Class")
    ax.legend()

plt.tight_layout()
plt.show()