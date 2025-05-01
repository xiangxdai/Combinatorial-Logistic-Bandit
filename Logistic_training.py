import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the movie data from the .npz file
data = np.load('data/movies.npz')

# Extract the three arrays: rating matrix, genre list, and genre matrix
W, genre_list, genre_mat = data['arr_0'], data['arr_1'], data['arr_2']

# Transpose W so that rows represent users and columns represent movies
W = W.T

# Get the number of users (m) and movies (L)
m, L = W.shape

# Randomly select half of the users for training (without replacement)
train_rows = np.random.choice(m, int(m / 2), replace=False)

# Get the remaining users for testing
test_rows = np.setdiff1d(range(m), train_rows)

# Split the rating matrix into training and test sets
W_train, W_test = W[train_rows, :], W[test_rows, :]

# Binarize the ratings: ratings >= 3 are 1 (like), others are 0 (dislike)
threshold = 3
W_train_binary = (W_train >= threshold).astype(int)
W_test_binary = (W_test >= threshold).astype(int)

# Initialize lists to store features and labels for training and testing
X_train = []
y_train = []
X_test = []
y_test = []

# Prepare training data: create feature-label pairs for rated movies
for user_idx, user_row in enumerate(W_train_binary):
    for movie_idx, rating in enumerate(user_row):
        if W_train[user_idx, movie_idx] != 0:  # Only include rated movies
            X_train.append(genre_mat[movie_idx])  # Use genre features for the movie
            y_train.append(rating)  # Use binarized rating as label

# Prepare test data: create feature-label pairs for rated movies
for user_idx, user_row in enumerate(W_test_binary):
    for movie_idx, rating in enumerate(user_row):
        if W_test[user_idx, movie_idx] != 0:  # Only include rated movies
            X_test.append(genre_mat[movie_idx])  # Use genre features for the movie
            y_test.append(rating)  # Use binarized rating as label

# Convert lists to NumPy arrays for model training
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

# Predict labels and probabilities on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# Evaluate the model using accuracy and AUC
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Print the evaluation results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")