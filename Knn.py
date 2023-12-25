from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from buildDataset import buildCharDB, CharDataBase
import numpy as np
import cv2

buildCharDB() 

# Extract features and labels from your CharDataBase
features = []  # Add the features you want to use for similarity
labels = []    # Add the corresponding labels


def Knn_init(k = 1):
    for char_instance in CharDataBase:
        # Assuming col_sum is a 2D array, flatten it to 1D
        flattened_col_sum = char_instance.col_sum.flatten()
        
        
        # Concatenate or combine features as needed
        combined_features = np.concatenate([flattened_col_sum])
        
        # Append combined features and label to lists
        features.append(combined_features)
        labels.append(char_instance.char)

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k,p=2,metric='euclidean')
    return knn


def testKnn(knn):
    # Split the data into training and testing sets
    train_input, test_input, train_output, test_output = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the classifier
    knn.fit(train_input, train_output)

    # Predict using the trained classifier
    predictions = knn.predict(test_input)

    # Evaluate the accuracy
    accuracy = accuracy_score(test_output, predictions)
    print("Accuracy:", accuracy)

def trainKnn(knn):
    # Train the classifier
    knn.fit(features, labels)


def predictKnn(knn, letters):
    predictions = knn.predict(letters)
    return predictions