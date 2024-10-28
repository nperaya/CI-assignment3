import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neural_network import MLPClassifier

# Load data
print("Loading data...")
data = pd.read_csv('C:\\Users\\PC\\Desktop\\CI_assignment3\\wdbc.data', header=None)
X = data.iloc[:, 2:].values  # Features
y = data.iloc[:, 1].map({'M': 1, 'B': 0}).values  # Labels

# Standardization
print("Standardizing data...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Cross-validation function
def cross_validation(X, y, hidden_layers, nodes, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf = MLPClassifier(hidden_layer_sizes=(nodes,) * hidden_layers, max_iter=500, learning_rate_init=0.001, tol=1e-3, warm_start=True, random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)

# Genetic Algorithm
def genetic_algorithm(population_size, generations):
    print("Starting Genetic Algorithm...")
    population = []
    fitness_history = []
    average_fitness_history = []
    min_fitness_history = []

    for _ in range(population_size):
        hidden_layers = np.random.randint(1, 7)  # 1 to 6 hidden layers
        nodes = np.random.randint(5, 101)  # 5 to 100 nodes
        population.append((hidden_layers, nodes))
    
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}...")
        fitness_scores = [cross_validation(X, y, hl, nd)[0] for hl, nd in population]
        std_devs = [cross_validation(X, y, hl, nd)[1] for hl, nd in population]

        fitness_history.append(max(fitness_scores))
        average_fitness_history.append(np.mean(fitness_scores))
        min_fitness_history.append(np.min(fitness_scores))
        
        sorted_population = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
        top_half = [x[1] for x in sorted_population[:population_size // 2]]

        new_population = []
        for _ in range(population_size // 2):
            parent_indices = np.random.choice(len(top_half), 2, replace=False)
            parent1_params = top_half[parent_indices[0]]
            parent2_params = top_half[parent_indices[1]]

            child = (parent1_params[0], parent2_params[1])  # Crossover
            if np.random.rand() < 0.1:  # Mutation
                child = (np.random.randint(1, 7), np.random.randint(5, 101))
            new_population.append(child)
        
        population = top_half + new_population
    
    # Plot fitness scores over generations
    plt.plot(fitness_history, label='Best Fitness Score (Accuracy)')
    plt.plot(average_fitness_history, label='Average Fitness Score (Accuracy)')
    plt.plot(min_fitness_history, label='Minimum Fitness Score (Accuracy)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score (Accuracy)')
    plt.title('Fitness Progression Over Generations')
    plt.legend()
    plt.grid()
    plt.show()

    return sorted_population[0][1]  # Return the best parameters

# Run Genetic Algorithm
print("Running Genetic Algorithm...")
best_parameters = genetic_algorithm(population_size=20, generations=20)
print(f"Best parameters found: Hidden Layers: {best_parameters[0]}, Nodes: {best_parameters[1]}")

# Analyze Results
print("Analyzing results...")
final_accuracy, std_dev = cross_validation(X, y, best_parameters[0], best_parameters[1])
print(f"Final Model Accuracy with Best Parameters: {final_accuracy:.2f} ± {std_dev:.2f}")

# Train final model for confusion matrix
print("Training final model for confusion matrix...")
final_model = MLPClassifier(
    hidden_layer_sizes=(best_parameters[1],) * best_parameters[0],
    max_iter=500,
    random_state=42
)
final_model.fit(X, y)

# Display confusion matrix
print("Displaying confusion matrix...")
y_pred = final_model.predict(X)
conf_matrix = confusion_matrix(y, y_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=['Benign', 'Malignant']).plot()
plt.title("Confusion Matrix of the Best Model")
plt.show()

# Additional metrics: Precision, Recall, F1 Score
print("Calculating additional metrics...")
report = classification_report(y, y_pred, target_names=['Benign', 'Malignant'])
print("Classification Report:")
print(report)
