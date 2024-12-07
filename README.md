# MLKit 🤖

MLKit is a Python library featuring machine learning algorithms implemented from scratch, inspired by the concepts and functionality of popular libraries like Scikit-Learn.

Please note that this repo is a work in progress, my first project of this magnitude, and its main purpose is educational.

## Installation

To install the MLKit library, run the following command:

```bash
pip install git+https://github.com/b-ionut-r/mlkit.git@main
```

## Usage

Each model follows the Scikit-Learn API (is endowed with .fit(), .predict(), .score() methods)

### Regression

To use the linear regression model for regression tasks, follow the example below:

```python
from mlkit.linear_model import LinearRegression
from mlkit.model_selection import train_test_split
from mlkit.preprocessing import MinMaxScaler

# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
reg_model = LinearRegression(n_iters = 25000, 
                             lr = 1e-3, 
                             reg = ("l2", 1e-3),
                             method = "gd",
                             batch_size = 128, # if method is "sgd"
                             scaler = MinMaxScaler(), 
                             verbose = 1000,
                             random_state = 42)
reg_model.fit(X_train, y_train)

# Evaluate the model
print(reg_model.score(X_test, y_test))
```


### Binary Classification

To use the logistic regression model for binary classification, follow the example below:

```python
from mlkit.linear_model import LogisticRegression
from mlkit.preprocessing import MinMaxScaler
from mlkit.model_selection import train_test_split

# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
cls_model = LogisticRegression(n_iters = 25000, 
                               lr = 1e-3, 
                               reg = ("l2", 1e-3), 
                               method = "gd",
                               batch_size = 128,
                               scaler = MinMaxScaler(), 
                               threshold = 0.5,
                               verbose = 1000,
                               random_state = 42)
cls_model.fit(X_train, y_train)

# Evaluate the model
print(cls_model.score(X_test, y_test))
```

### Multiclass Classification

To use the logistic regression model for multiclass classification, follow the example below:

```python
from mlkit.linear_model import MultiLogisticRegression
from mlkit.model_selection import train_test_split
from mlkit.preprocessing import MinMaxScaler, LabelEncoder


# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
cls_model = MultiLogisticRegression(n_iters = 25000, 
                                    lr = 1e-3, 
                                    reg = ("l2", 1e-3), 
                                    method = "gd",
                                    batch_size = 128,
                                    scaler = MinMaxScaler(), 
                                    label_encoder = LabelEncoder(), # one hot encodes label categories
                                    verbose = 1000,
                                    random_state = 42)
cls_model.fit(X_train, y_train)

# Evaluate the model
print(cls_model.score(X_test, y_test))
```

### Decision Tree

To use the decision tree model for classification / regression tasks, follow the example below:

```python
from mlkit.tree import DecisionTreeClassifier # or DecisionTreeRegressor
from mlkit.model_selection import train_test_split

# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
tree_model = DecisionTreeClassifier(criterion = "gini",
                                    splitter = "best",
                                    max_depth = 10, 
                                    min_samples_split = 5, 
                                    min_samples_leaf = 2,
                                    max_features = "sqrt",
                                    max_leaf_nodes = 100,
                                    min_info_gain = 0.1,
                                    random_state = 42)
tree_model.fit(X_train, y_train)

# Evaluate the model
print(tree_model.score(X_test, y_test))

# Plot the tree
tree_model.plot_tree(feature_names)
```


### K-Nearest Neighbors

To use the K-Nearest Neighbors model for classification tasks, follow the example below:

```python
from mlkit.neighbors import KNeighborsClassifier
from mlkit.model_selection import train_test_split
from mlkit.preprocessing import MinMaxScaler

# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
knn_model = KNeighborsClassifier(k = 5, 
                                 metric = 'euclidean',
                                 scaler = MinMaxScaler())
knn_model.fit(X_train, y_train)

# Evaluate the model
print(knn_model.score(X_test, y_test))
```

## Demos

The repository contains several Jupyter notebooks demonstrating the usage of the MLKit library for various machine learning tasks:

- `demo_binary.ipynb`: Binary classification using logistic regression.
- `demo_multiclass.ipynb`: Multiclass classification using logistic regression.
- `demo_simple_reg.ipynb`: Simple linear regression.
- `demo_multiple_reg.ipynb`: Multiple linear regression.
- `demo_multilabel_reg.ipynb`: Multilabel regression.
- `demo_kneighbors.ipynb`: K-Nearest Neighbors algorithm.
- `demo_binary_tuner.ipynb`: Hyperparameter tuning for binary classification.
- `demo_regression_tuner.ipynb`: Hyperparameter tuning for regression.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact the repository owner.
