import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def load_images_from_folder(folder, data_type):
    images = []
    labels = []

    for letter in os.listdir(os.path.join(folder, data_type)):
        letter_path = os.path.join(folder, data_type, letter)
        print(letter)
        count = 0
        if os.path.isdir(letter_path):
            for image_file in os.listdir(letter_path):
                count += 1
                image_path = os.path.join(letter_path, image_file)
                image = np.array(Image.open(image_path).convert('L'))
                images.append(image.flatten())
                labels.append(letter)
                if count == 500:
                    break

    return np.array(images), labels

# Specify the path to the folder containing Train and Test subfolders
data_folder = "C:\\Users\\Matt\\Documents\\UCT\\Year 4\\EEE4114F\\ML Project\\datasets\\handsigns"

# Load training data
X_train, y_train = load_images_from_folder(data_folder, "Train")

# Load test data
X_test, y_test = load_images_from_folder(data_folder, "Test")

# Encode labels as integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Print the label dictionary as a dictionary
label_dict = dict(zip(le.transform(le.classes_), le.classes_))
print("Label Dictionary:")
print(label_dict)
print()

# Save the dataset as NumPy arrays
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)