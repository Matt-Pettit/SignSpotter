import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Main one:
# ["-", "+", "=", "times", "div", "infty", "sum", "theta", "int", "pi"]

# ["0","1","2","3","4","5","6","7","8","9"]

def load_images_from_folder(folder):
    images = []
    labels = []
    for symbol_type in os.listdir(folder):
        
        if (symbol_type not in ["-", "+", "=", "times", "div", "infty", "sum", "theta", "int", "pi"]):
            continue
        print(symbol_type)
        symbol_path = os.path.join(folder, symbol_type)
        count = 0
        if os.path.isdir(symbol_path):
            for image_file in os.listdir(symbol_path):
                count = count + 1
                image_path = os.path.join(symbol_path, image_file)
                image = np.array(Image.open(image_path).convert('L'))
                images.append(image.flatten())
                labels.append(symbol_type)
                if count == 1000:
                    break
    return np.array(images), labels

# Specify the path to the extracted_images folder
data_folder = "C:\\Users\\Matt\\Desktop\\extracted_images"

# Load images and labels
X, y = load_images_from_folder(data_folder)

# Encode labels as integers
le = LabelEncoder()
y = le.fit_transform(y)

# Print the label dictionary as a dictionary
label_dict = dict(zip(le.transform(le.classes_), le.classes_))
print("Label Dictionary:")
print(label_dict)
print()

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the dataset as NumPy arrays
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)