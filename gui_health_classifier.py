import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import threading

class HealthClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Health Terms Classifier")
        self.root.geometry("800x600")
        
        self.trained_model = None
        
        self.setup_gui()
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    
    def setup_gui(self):
        #Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        #Title
        title_label = ttk.Label(main_frame, text="Health Terms Classifier", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        #Training section
        training_frame = ttk.LabelFrame(main_frame, text="Model Training", padding="10")
        training_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(training_frame, text="Select training data file:").grid(row=0, column=0, sticky=tk.W)
        self.training_file_var = tk.StringVar()
        ttk.Entry(training_frame, textvariable=self.training_file_var, width=50).grid(row=0, column=1, padx=(10, 5))
        ttk.Button(training_frame, text="Browse", command=self.browse_training_file).grid(row=0, column=2)
        
        ttk.Button(training_frame, text="Train Model", command=self.start_training).grid(row=1, column=0, pady=(10, 0))
        
        #Prediction section
        prediction_frame = ttk.LabelFrame(main_frame, text="Make Predictions", padding="10")
        prediction_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(prediction_frame, text="Select file to predict:").grid(row=0, column=0, sticky=tk.W)
        self.predict_file_var = tk.StringVar()
        ttk.Entry(prediction_frame, textvariable=self.predict_file_var, width=50).grid(row=0, column=1, padx=(10, 5))
        ttk.Button(prediction_frame, text="Browse", command=self.browse_predict_file).grid(row=0, column=2)
        
        ttk.Button(prediction_frame, text="Make Predictions", command=self.start_prediction).grid(row=1, column=0, pady=(10, 0))
        
        #Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        #Text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.results_text = tk.Text(text_frame, height=20, width=80)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        #Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
    
    def browse_training_file(self):
        filename = filedialog.askopenfilename(
            title="Select Training Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.training_file_var.set(filename)
    
    def browse_predict_file(self):
        filename = filedialog.askopenfilename(
            title="Select File to Predict",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.predict_file_var.set(filename)
    
    def log_message(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def start_training(self):
        if not self.training_file_var.get():
            messagebox.showerror("Error", "Please select a training data file.")
            return
        
        #Run training in separate thread to prevent GUI freezing
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def train_model(self):
        try:
            self.results_text.delete(1.0, tk.END)
            self.log_message("=" * 50)
            self.log_message("TRAINING MODEL")
            self.log_message("=" * 50)
            
            #Load training data
            self.log_message("Loading training data...")
            df = pd.read_csv(self.training_file_var.get(), encoding='latin1')
            
            relevant_cols = ['Term Names', 'Definitions', 'Suggested TOPIC_DESC']
            df = df[relevant_cols]
            df.dropna(subset=['Suggested TOPIC_DESC'], inplace=True)
            df['Term Names'].fillna('', inplace=True)
            df['Definitions'].fillna('', inplace=True)
            df['text_feature'] = df['Term Names'] + ' ' + df['Definitions']
            df['text_feature'] = df['text_feature'].apply(self.clean_text)
            
            self.log_message(f"Data loaded: {df.shape[0]} samples")
            self.log_message(f"Categories: {df['Suggested TOPIC_DESC'].nunique()}")
            
            X = df['text_feature']
            y = df['Suggested TOPIC_DESC']
            
            #Split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            #Train and evaluate
            self.log_message("Training and evaluating model...")
            eval_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                ('clf', LinearSVC(random_state=42, C=0.5, class_weight='balanced'))
            ])
            
            eval_pipeline.fit(X_train, y_train)
            y_pred = eval_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.log_message(f"Model Accuracy: {accuracy:.4f}")
            self.log_message("\nClassification Report:")
            self.log_message(classification_report(y_test, y_pred, zero_division=0))
            
            #Train final model on all data
            self.log_message("Training final model on complete dataset...")
            self.trained_model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                ('clf', LinearSVC(random_state=42, C=0.5, class_weight='balanced'))
            ])
            
            self.trained_model.fit(X, y)
            self.log_message("Training complete! Model ready for predictions.")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during training:\n{str(e)}")
            self.log_message(f"ERROR: {str(e)}")
    
    def start_prediction(self):
        if not self.trained_model:
            messagebox.showerror("Error", "Please train a model first.")
            return
        
        if not self.predict_file_var.get():
            messagebox.showerror("Error", "Please select a file to predict.")
            return
        
        #Ask for output file
        output_file = filedialog.asksaveasfilename(
            title="Save Predictions As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        #Run prediction in separate thread
        threading.Thread(target=self.make_predictions, args=(output_file,), daemon=True).start()
    
    def make_predictions(self, output_file):
        try:
            self.log_message("\n" + "=" * 50)
            self.log_message("MAKING PREDICTIONS")
            self.log_message("=" * 50)
            
            #Load data to predict
            self.log_message("Loading data to predict...")
            df = pd.read_csv(self.predict_file_var.get(), encoding='latin1')
            
            #Check required columns
            required_cols = ['Term Names', 'Definitions']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise Exception(f"Missing required columns: {missing_cols}")
            
            #Prepare data
            df['Term Names'].fillna('', inplace=True)
            df['Definitions'].fillna('', inplace=True)
            df['text_feature'] = df['Term Names'] + ' ' + df['Definitions']
            df['text_feature'] = df['text_feature'].apply(self.clean_text)
            
            #Make predictions
            self.log_message(f"Making predictions for {len(df)} terms...")
            predictions = self.trained_model.predict(df['text_feature'])
            df['Suggested TOPIC_DESC'] = predictions
            
            #Save results
            df.to_csv(output_file, index=False)
            self.log_message(f"Predictions saved to: {output_file}")
            
            #Show sample predictions
            self.log_message("\nSample predictions:")
            for i, row in df.head(5).iterrows():
                self.log_message(f"Term: '{row['Term Names']}'")
                self.log_message(f"Predicted: '{row['Suggested TOPIC_DESC']}'")
                self.log_message("-" * 30)
            
            messagebox.showinfo("Success", f"Predictions completed and saved to:\n{output_file}")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{str(e)}")
            self.log_message(f"ERROR: {str(e)}")

def main():
    root = tk.Tk()
    app = HealthClassifierGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
