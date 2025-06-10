#  Step 1: Imported libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#  Step 2: Create sample CSV data
def create_sample_csv():
    data = {
        'Name': ['Ali', 'Fatima', 'John', 'Sara', 'Ahmed'],
        'Age': [20, 22, 21, None, 23],
        'Marks': [85, 90, 88, 92, None],
        'Passed': ['Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    df = pd.DataFrame(data)
    df.to_csv("students_data.csv", index=False)
    print(" Sample CSV file created: students_data.csv")

#  Step 3: ETL functions
def extract_data(file_path):
    print(" Extracting data...")
    return pd.read_csv(file_path)

def transform_data(df):
    print(" Transforming data...")

    # Separating features and target
    X = df.drop(columns=['Name', 'Passed'])
    y = df['Passed']

    # Define numerical columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Define pipeline for numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combining transformers
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor

def load_data(X, y):
    print(" Loading data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(" Data loaded and ready for training/testing!")
    print("X_train shape:", X_train.shape)
    print("y_train sample:", y_train.head())

#  Step 4: Main runner
if __name__ == "__main__":
    create_sample_csv()  # Create the file
    df = extract_data("students_data.csv")
    X_transformed, y, preprocessor = transform_data(df)
    load_data(X_transformed, y)
