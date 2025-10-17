import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class FraudApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Credit Card Fraud Detection")
        self.root.geometry("800x600")

        # Upload button
        tk.Button(root, text="Upload CSV", command=self.load_csv).pack(pady=10)

        # Train button
        tk.Button(root, text="Train Model", command=self.train_model).pack(pady=10)

        # Evaluate button
        tk.Button(root, text="Evaluate Model", command=self.evaluate_model).pack(pady=10)

        # Output box
        self.output = scrolledtext.ScrolledText(root, width=90, height=25)
        self.output.pack(pady=10)

        # Variables
        self.df = None
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.history = None

    def log(self, text):
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        self.df = pd.read_csv(file_path)
        self.log("CSV Loaded: " + file_path)
        self.log(str(self.df.head()))

        # Preprocess
        x = self.df.drop('Class', axis=1)
        y = self.df['Class']

        scaler = StandardScaler()
        x[['Amount','Time']] = scaler.fit_transform(x[['Amount','Time']])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        self.log("Data Preprocessing Complete.")

    def build_model(self, input_dim):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),

            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model

    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please upload a CSV first.")
            return

        self.model = self.build_model(self.x_train.shape[1])

        # Compute class weights
        cw = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        cw = {i: cw[i] for i in range(len(cw))}
        self.log("Class Weights: " + str(cw))

        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=5,  # fewer for demo
            batch_size=2048,
            validation_data=(self.x_test, self.y_test),
            class_weight=cw,
            verbose=1
        )

        self.log("Training complete.")
        self.plot_training()

    def plot_training(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Acc')
        plt.plot(self.history.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title("Loss")

        plt.tight_layout()
        plt.show()

    def evaluate_model(self):
        if not self.model:
            messagebox.showerror("Error", "Train the model first.")
            return

        y_probs = self.model.predict(self.x_test)
        y_pred = (y_probs > 0.5).astype(int)

        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        self.log("Confusion Matrix:\n" + str(cm))
        self.log("\nClassification Report:\n" + report)

        # Plot ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.title("ROC Curve")
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = FraudApp(root)
    root.mainloop()
