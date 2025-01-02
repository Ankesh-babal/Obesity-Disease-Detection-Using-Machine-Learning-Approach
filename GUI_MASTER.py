from subprocess import call
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
import xgboost as xgb

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import dump
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

root = tk.Tk()
root.title("Obesity Detection System Using ML")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

##### For background Image
# Load the image and resize it using Image.Resampling.LANCZOS
image2 = Image.open('2.jpg')
image2 = image2.resize((w, h), Image.Resampling.LANCZOS)

# Convert the image to a format suitable for Tkinter
background_image = ImageTk.PhotoImage(image2)

# Store a reference to prevent garbage collection
root.background_image = background_image

# Create a label to display the background image
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0)

lbl = tk.Label(root, text="Obesity Disease Detection", font=("Times New Roman", 20, 'bold'),
               background="white", borderwidth=5, relief='solid', fg="red", width=50)
lbl.place(x=250, y=20)

# Define the function for training the SVM model
def Model_Training_SVM():
    df = pd.read_csv('train.csv')

    # Encode categorical features
    label_encoders = {}
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'MTRANS']
    for column in categorical_features:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Encode the target variable
    le_target = LabelEncoder()
    df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])

    # Separate features and labels
    X = df.drop('NObeyesdad', axis=1).values
    y = df['NObeyesdad'].values

    # Standardize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize SVC classifier
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    class_names = [str(cls) for cls in le_target.classes_]

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - SVM', fontsize=18)
    plt.savefig("svm_CM.png")
    plt.show()

    # Display accuracy and save model
    label4 = tk.Label(root, text=f"{report}\nAccuracy: {accuracy * 100:.2f}%\nModel saved as svm_obesity_model.joblib",
                      bg='#abebc6', fg='black', font=("Tempus Sans ITC", 14), borderwidth=2, relief='solid')
    label4.place(x=350, y=200)

    dump(model, "svm_obesity_model.joblib")
    print("Model saved as svm_obesity_model.joblib")

    ##### Update background image with confusion matrix
    cm_image = Image.open('svm_CM.png')
    cm_image = cm_image.resize((320, 320), Image.Resampling.LANCZOS)
    cm_photo = ImageTk.PhotoImage(cm_image)

    root.cm_image = cm_photo  # Store reference to prevent garbage collection

    cm_label = tk.Label(root, image=cm_photo, borderwidth=3, relief='solid')
    cm_label.place(x=800, y=200)


# Define the function for training the XGBoost model
def Model_Training_XG():
    df = pd.read_csv('train.csv')

    # Encode categorical features
    label_encoders = {}
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'MTRANS']
    for column in categorical_features:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Encode the target variable
    le_target = LabelEncoder()
    df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])

    # Separate features and labels
    X = df.drop('NObeyesdad', axis=1).values
    y = df['NObeyesdad'].values

    # Standardize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    class_names = [str(cls) for cls in le_target.classes_]

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - XGBoost', fontsize=18)
    plt.savefig("xgboost_CM.png")
    plt.show()

    # Display accuracy and save model
    label4 = tk.Label(root, text=f"{report}\nAccuracy: {accuracy * 100:.2f}%\nModel saved as xgboost_obesity_model.joblib",
                      bg='#abebc6', fg='black', font=("Tempus Sans ITC", 14), borderwidth=2, relief='solid')
    label4.place(x=350, y=200)

    dump(model, "xgboost_obesity_model.joblib")
    print("Model saved as xgboost_obesity_model.joblib")

    ##### Update background image with confusion matrix
    cm_image = Image.open('xgboost_CM.png')
    cm_image = cm_image.resize((320, 320), Image.Resampling.LANCZOS)
    cm_photo = ImageTk.PhotoImage(cm_image)

    root.cm_image = cm_photo  # Store reference to prevent garbage collection

    cm_label = tk.Label(root, image=cm_photo, borderwidth=3, relief='solid')
    cm_label.place(x=800, y=200)


# Define the function for training the RandomForest model
def Model_Training_RF():
    df = pd.read_csv('train.csv')

    # Encode categorical features
    label_encoders = {}
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'MTRANS']
    for column in categorical_features:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Encode the target variable
    le_target = LabelEncoder()
    df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])

    # Separate features and labels
    X = df.drop('NObeyesdad', axis=1).values
    y = df['NObeyesdad'].values

    # Standardize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize RandomForest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    class_names = [str(cls) for cls in le_target.classes_]

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Oranges)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - RandomForest', fontsize=18)
    plt.savefig("rf_CM.png")
    plt.show()

    # Display accuracy and save model
    label4 = tk.Label(root, text=f"{report}\nAccuracy: {accuracy * 100:.2f}%\nModel saved as rf_obesity_model.joblib",
                      bg='#abebc6', fg='black', font=("Tempus Sans ITC", 14), borderwidth=2, relief='solid')
    label4.place(x=350, y=200)

    dump(model, "rf_obesity_model.joblib")
    print("Model saved as rf_obesity_model.joblib")

    ##### Update background image with confusion matrix
    cm_image = Image.open('rf_CM.png')
    cm_image = cm_image.resize((320, 320), Image.Resampling.LANCZOS)
    cm_photo = ImageTk.PhotoImage(cm_image)

    root.cm_image = cm_photo  # Store reference to prevent garbage collection

    cm_label = tk.Label(root, image=cm_photo, borderwidth=3, relief='solid')
    cm_label.place(x=800, y=200)

# Define CNN Model Training
def cnn_train_file():
    df = pd.read_csv('train.csv')

    # Encode categorical features
    label_encoders = {}
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'MTRANS']
    for column in categorical_features:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Encode the target variable
    le_target = LabelEncoder()
    df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])

    # Separate features and labels
    X = df.drop('NObeyesdad', axis=1).values
    y = df['NObeyesdad'].values

    # Standardize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize a simple Sequential model (Fully connected layers for tabular data)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Number of classes in y_train

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Make predictions
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Classification report
    class_names = [str(cls) for cls in le_target.classes_]
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Purples)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - CNN', fontsize=18)
    plt.savefig("cnn_CM.png")
    plt.show()

    # Display accuracy and save model
    label4 = tk.Label(root, text=f"Accuracy: {accuracy * 100:.2f}%\nModel saved as cnn_obesity_model.h5",
                      bg='#abebc6', fg='black', font=("Tempus Sans ITC", 14), borderwidth=2, relief='solid')
    label4.place(x=350, y=200)

    model.save("cnn_obesity_model.h5")
    print("Model saved as cnn_obesity_model.h5")

    ##### Update background image with confusion matrix
    cm_image = Image.open('cnn_CM.jpg')
    cm_image = cm_image.resize((320, 320), Image.Resampling.LANCZOS)
    cm_photo = ImageTk.PhotoImage(cm_image)

    root.cm_image = cm_photo  # Store reference to prevent garbage collection

    cm_label = tk.Label(root, image=cm_photo, borderwidth=3, relief='solid')
    cm_label.place(x=800, y=200)

def call_file():
    root.destroy()
    import test_gui


def window():
    root.destroy()


# Define buttons and their layout
button3 = tk.Button(root, font=('times', 18, 'bold'), bg="#6495ED", fg="white",
                    text="SVM Model Training", command=Model_Training_SVM, width=20, height=1)
button3.place(x=20, y=150)

button3 = tk.Button(root, foreground="white", background="#6495ED", font=("times", 18, "bold"),
                    text="XG-Boost Model Training", command=Model_Training_XG, width=20, height=1)
button3.place(x=20, y=220)

button3 = tk.Button(root, foreground="white", background="#6495ED", font=("times", 18, "bold"),
                    text="RF Model Training", command=Model_Training_RF, width=20, height=1)
button3.place(x=20, y=290)



button4 = tk.Button(root, foreground="white", background="green", font=("times", 18, "bold"),
                    text="Test Obesity", command=call_file, width=20, height=1)
button4.place(x=20, y=460)

exit1 = tk.Button(root, text="Exit", command=window, width=20, height=1, font=('times', 18, 'bold'),
                  bg="red", fg="white")
exit1.place(x=20, y=530)

root.mainloop()
