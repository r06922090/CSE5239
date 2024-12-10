import pandas as pd
import joblib  # To save and load the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the CSV file and preprocess
def load_data(csv_file):
    df = pd.read_csv(csv_file)

    # Encode categorical columns using label encoding
    label_encoder = LabelEncoder()

    # Store original labels for later use
    original_labels = {
        'Model': label_encoder.fit(df['Model']).classes_,
        'Optimizer': label_encoder.fit(df['Optimizer']).classes_
    }

    # Encode categorical columns
    df['Model'] = label_encoder.fit_transform(df['Model'])
    df['Optimizer'] = label_encoder.fit_transform(df['Optimizer'])

    # Ensure the columns are numeric
    df['Batch Size'] = pd.to_numeric(df['Batch Size'], errors='coerce')
    df['Learning Rate'] = pd.to_numeric(df['Learning Rate'], errors='coerce')
    df['Epochs'] = pd.to_numeric(df['Epochs'], errors='coerce')

    # Ensure the 'Test Accuracy' column is numeric
    if 'Test Accuracy' in df.columns:
        df['Test Accuracy'] = pd.to_numeric(df['Test Accuracy'], errors='coerce')
    else:
        raise KeyError("'Test Accuracy' column not found in CSV file.")

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Features (X) and target (y)
    X = df[['Batch Size', 'Learning Rate', 'Model', 'Optimizer', 'Epochs']].values
    y = df['Test Accuracy'].values

    # Scale the features for better training
    scaler = StandardScaler()

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, original_labels

# Train the linear regression model
def train_model(X_train, y_train, X_test, y_test):
    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    test_loss = mean_squared_error(y_test, y_pred)

    return test_loss, model

# Save the model to a file
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Load the model from a file
def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

# Main function
def main():
    # Ask the user if they want to run in training mode or testing mode
    mode = input("Do you want to run in training mode or testing mode? (training/testing): ").strip().lower()

    if mode == "training":
        # Load and preprocess the data
        X, y, original_labels = load_data('record.csv')

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store the best combination of features and the corresponding accuracy
        best_accuracy = -float('inf')  # Start with a very low value
        best_input = None
        best_input_labels = None
        best_model = None

        # Train the model and find the best input (highest accuracy)
        for i in range(len(X_train)):
            # Train the model with current train subset
            X_train_subset = X_train[:i+1]  # Using the first i+1 samples for training
            y_train_subset = y_train[:i+1]

            # Train the model
            test_loss, model = train_model(X_train_subset, y_train_subset, X_test, y_test)

            # Get the accuracy (instead of test_loss)
            test_accuracy = 1 - test_loss  # Convert error to accuracy

            # Check if this is the best model so far (highest accuracy)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_input = X_train_subset[-1]  # Save the last input (which maximizes accuracy)
                best_input_labels = {
                    'Batch Size': best_input[0],
                    'Learning Rate': best_input[1],
                    'Model': original_labels['Model'][int(best_input[2])],  # Map encoded value back
                    'Optimizer': original_labels['Optimizer'][int(best_input[3])],  # Map encoded value back
                    'Epochs': best_input[4]
                }
                best_model = model  # Save the best model

        # Print the best labels and accuracy without the raw input
        print(f"Best Input Labels (original values): {best_input_labels}")
        print(f"Highest Accuracy: {best_accuracy}")

        # Save the best model to a file
        save_model(best_model, 'hyperparameter.joblib')

    elif mode == "testing":
        # Load the best model directly (no user input for file path)
        try:
            model = load_model('hyperparameter.joblib')
            
            # Use the best model to make predictions and calculate highest accuracy
            print("Testing mode activated.")

            # Load the data and preprocess it
            X, y, original_labels = load_data('record.csv')

            # Make predictions using the loaded model
            y_pred = model.predict(X)
            
            # Calculate the highest accuracy and corresponding hyperparameters
            highest_acc = max(y_pred)
            best_index = list(y_pred).index(highest_acc)
            best_input = X[best_index]
            
            # Map the best input back to original labels
            best_input_labels = {
                'Batch Size': best_input[0],
                'Learning Rate': best_input[1],
                'Model': original_labels['Model'][int(best_input[2])],  # Map encoded value back
                'Optimizer': original_labels['Optimizer'][int(best_input[3])],  # Map encoded value back
                'Epochs': best_input[4]
            }

            print(f"Predicted Highest Accuracy: {highest_acc}")
            print(f"Corresponding Hyperparameters: {best_input_labels}")

        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Invalid mode. Please enter 'training' or 'testing'.")

if __name__ == "__main__":
    main()