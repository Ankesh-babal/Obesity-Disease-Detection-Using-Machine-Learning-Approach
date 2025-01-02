from subprocess import call
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve
from mlxtend.plotting import plot_confusion_matrix
import xgboost as xgb

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


root = tk.Tk()
root.title("Obesity Detection System Using ML")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

##### For background Image
# Load the image and resize it using Image.Resampling.LANCZOS
image2 = Image.open('C:/Users/ADMIN/Downloads/all data/all data/100% Obesity Disease Detection/100% Obesity Disease Detection/2.jpg')
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
    df = pd.read_csv('C:/Users/ADMIN/Downloads/all data/all data/100% Obesity Disease Detection/100% Obesity Disease Detection/train.csv')

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
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig("C:/Users/ADMIN/Downloads/all data/all data/100% Obesity Disease Detection/100% Obesity Disease Detection/svm_CM.png")
    plt.show()

    # Display accuracy and save model
    label4 = tk.Label(root, text=f"{report}\nAccuracy: {accuracy * 100:.2f}%\nModel saved as svm_obesity_model.joblib",
                      bg='#abebc6', fg='black', font=("Tempus Sans ITC", 14), borderwidth=2, relief='solid')
    label4.place(x=350, y=200)

    from joblib import dump
    dump(model, "C:/Users/ADMIN/Downloads/all data/all data/100% Obesity Disease Detection/100% Obesity Disease Detection/svm_obesity_model.joblib")
    print("Model saved as svm_obesity_model.joblib")

    ##### Update background image with confusion matrix
    cm_image = Image.open('C:/Users/ADMIN/Downloads/all data/all data/100% Obesity Disease Detection/100% Obesity Disease Detection/svm_CM.png')
    cm_image = cm_image.resize((320, 320), Image.Resampling.LANCZOS)
    cm_photo = ImageTk.PhotoImage(cm_image)

    root.cm_image = cm_photo  # Store reference to prevent garbage collection

    cm_label = tk.Label(root, image=cm_photo, borderwidth=3, relief='solid')
    cm_label.place(x=800, y=200)


# Define the function for training the RandomForest model
def Model_Training_RF():
    # (Similar structure as Model_Training_SVM)
    pass  # Complete the RandomForest model function similarly


# Define the function for training the XGBoost model
def Model_Training_XG():
    # (Similar structure as Model_Training_SVM)
    pass  # Complete the XGBoost model function similarly


# Define CNN Model Training
def cnn_train_file():
    # (Similar structure as Model_Training_SVM)
    pass  # Complete the CNN model function similarly


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

button3 = tk.Button(root, foreground="white", background="#6495ED", font=("times", 18, "bold"),
                    text="CNN Model Training", command=cnn_train_file, width=20, height=1)
button3.place(x=20, y=360)

button4 = tk.Button(root, foreground="white", background="green", font=("times", 18, "bold"),
                    text="Test Obesity", command=call_file, width=20, height=1)
button4.place(x=20, y=460)

exit1 = tk.Button(root, text="Exit", command=window, width=20, height=1, font=('times', 18, 'bold'),
                  bg="red", fg="white")
exit1.place(x=20, y=530)

root.mainloop()

