import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AirQualityClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = 'overall_aqi_category'
        
    def prepare_features(self, df):
        """
        Prepare features from your dataset format
        Your data appears to be already normalized/standardized
        """
        # Core features from your dataset
        core_features = [
            'hour', 'day_of_week', 'month',
            'aqi_value_no2', 'aqi_value_o3', 'aqi_value_pm25',
            'concentration_no2', 'concentration_o3', 'concentration_pm25',
            'overall_aqi', 'season_encoded'
        ]
        
        # Create additional derived features for better classification
        df['aqi_max'] = df[['aqi_value_no2', 'aqi_value_o3', 'aqi_value_pm25']].max(axis=1)
        df['aqi_mean'] = df[['aqi_value_no2', 'aqi_value_o3', 'aqi_value_pm25']].mean(axis=1)
        df['aqi_std'] = df[['aqi_value_no2', 'aqi_value_o3', 'aqi_value_pm25']].std(axis=1)
        
        df['conc_max'] = df[['concentration_no2', 'concentration_o3', 'concentration_pm25']].max(axis=1)
        df['conc_mean'] = df[['concentration_no2', 'concentration_o3', 'concentration_pm25']].mean(axis=1)
        df['conc_std'] = df[['concentration_no2', 'concentration_o3', 'concentration_pm25']].std(axis=1)
        
        # Pollutant ratios (important for air quality assessment)
        df['no2_to_pm25_ratio'] = df['concentration_no2'] / (np.abs(df['concentration_pm25']) + 1e-6)
        df['o3_to_pm25_ratio'] = df['concentration_o3'] / (np.abs(df['concentration_pm25']) + 1e-6)
        df['no2_to_o3_ratio'] = df['concentration_no2'] / (np.abs(df['concentration_o3']) + 1e-6)
        
        # Temporal interaction features
        df['hour_season_interaction'] = df['hour'] * df['season_encoded']
        df['day_season_interaction'] = df['day_of_week'] * df['season_encoded']
        
        # AQI deviation from overall
        df['aqi_no2_deviation'] = df['aqi_value_no2'] - df['overall_aqi']
        df['aqi_o3_deviation'] = df['aqi_value_o3'] - df['overall_aqi']
        df['aqi_pm25_deviation'] = df['aqi_value_pm25'] - df['overall_aqi']
        
        # Additional derived features
        derived_features = [
            'aqi_max', 'aqi_mean', 'aqi_std',
            'conc_max', 'conc_mean', 'conc_std',
            'no2_to_pm25_ratio', 'o3_to_pm25_ratio', 'no2_to_o3_ratio',
            'hour_season_interaction', 'day_season_interaction',
            'aqi_no2_deviation', 'aqi_o3_deviation', 'aqi_pm25_deviation'
        ]
        
        # Select final feature set
        self.feature_columns = core_features + derived_features
        
        # Filter features that actually exist in the dataframe and handle NaN values
        available_features = []
        for col in self.feature_columns:
            if col in df.columns:
                # Fill NaN values for derived features
                if col in derived_features:
                    df[col] = df[col].fillna(0)
                available_features.append(col)
        
        self.feature_columns = available_features
        
        return df[self.feature_columns + [self.target_column]].copy()
    
    def build_model(self, input_dim, num_classes):
        """
        Build a neural network optimized for TinyML deployment on ESP32-S3
        Architecture designed for air quality classification
        """
        model = Sequential([
            # Input layer with batch normalization
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Hidden layers - balanced for accuracy and efficiency
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Use Adam optimizer with learning rate scheduling
        optimizer = Adam(learning_rate=0.001, decay=1e-6)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        return model
    
    def train(self, data_file, test_size=0.2, validation_size=0.1, epochs=100):
        # Store training data for TensorFlow Lite conversion
        self.X_train_scaled = None
        """
        Train the air quality classification model on your dataset
        """
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(data_file)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for the target column
        if self.target_column not in df.columns:
            print(f"Target column '{self.target_column}' not found!")
            print("Available columns:", list(df.columns))
            return None
        
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Remove rows with missing values
        df_processed = df_processed.dropna()
        print(f"Data shape after cleaning: {df_processed.shape}")
        
        # Check class distribution
        print("\nClass distribution:")
        class_counts = df_processed[self.target_column].value_counts()
        print(class_counts)
        
        # Separate features and target
        X = df_processed[self.feature_columns].values
        y = df_processed[self.target_column].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nFeature set ({len(self.feature_columns)} features):")
        for i, feature in enumerate(self.feature_columns):
            print(f"  {i+1:2d}. {feature}")
        
        print(f"\nClass labels:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {label}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), 
            random_state=42, stratify=y_temp
        )
        
        print(f"\nDataset splits:")
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features (your data might already be normalized, but we'll apply scaling anyway)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store for TensorFlow Lite conversion
        self.X_train_scaled = X_train_scaled
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_model(X_train_scaled.shape[1], num_classes)
        
        print(f"\nModel architecture:")
        self.model.summary()
        
        # Calculate class weights for imbalanced dataset
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"\nClass weights for balancing:")
        for i, weight in class_weight_dict.items():
            print(f"  {self.label_encoder.classes_[i]}: {weight:.3f}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_air_quality_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("\nStarting training...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=64,  # Increased batch size for better training
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy, test_top_k = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-K Accuracy: {test_top_k:.4f}")
        
        # Generate predictions for detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred_classes, target_names=target_names))
        
        # Feature importance analysis
        self.analyze_feature_importance(X_train_scaled, y_train)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred_classes, target_names)
        
        # Save model components
        self.save_model_components()
        
        return history
    
    def analyze_feature_importance(self, X_train, y_train):
        """
        Analyze feature importance using permutation importance
        """
        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import RandomForestClassifier
        
        print("\nAnalyzing feature importance...")
        
        # Use Random Forest to get feature importance as a baseline
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Top 10 most important features:")
        for i in range(min(10, len(self.feature_columns))):
            idx = indices[i]
            print(f"  {i+1:2d}. {self.feature_columns[idx]}: {importances[idx]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = min(15, len(self.feature_columns))
        plt.title('Top Feature Importances')
        plt.barh(range(top_features), importances[indices[:top_features]])
        plt.yticks(range(top_features), [self.feature_columns[i] for i in indices[:top_features]])
        plt.xlabel('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, target_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_components(self):
        """Save model and preprocessing components"""
        # Save the trained model
        self.model.save('air_quality_model.h5')
        
        # Save preprocessing components
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        joblib.dump(self.feature_columns, 'feature_columns.pkl')
        
        print("\nModel components saved:")
        print("- air_quality_model.h5")
        print("- scaler.pkl")
        print("- label_encoder.pkl")
        print("- feature_columns.pkl")
    
        # Convert model to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        # Save the model
        with open('air_quality_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # Create C array header file for ESP32
        self.create_c_header(tflite_model)
        
        print(f"TinyML model saved:")
        print(f"- air_quality_model.tflite ({len(tflite_model)} bytes)")
        print("- air_quality_model.h (C header file)")
        
        # Model size analysis
        try:
            original_size = os.path.getsize('air_quality_model.h5')
            tflite_size = len(tflite_model)
            compression_ratio = original_size / tflite_size
            
            print(f"\nModel Analysis:")
            print(f"- Quantization Type: {quantization_type}")
            print(f"- Original Keras model: {original_size/1024:.1f} KB")
            print(f"- TensorFlow Lite model: {tflite_size/1024:.1f} KB")
            print(f"- Compression ratio: {compression_ratio:.1f}x")
            print(f"- ESP32-S3 Compatible: {'✅ Yes' if tflite_size < 1000000 else '⚠️  Large but possible'}")
            
        except Exception as e:
            print(f"Could not analyze model size: {e}")
            print(f"TensorFlow Lite model size: {len(tflite_model)} bytes")
    
    def create_c_header(self, tflite_model):
        """Create C header file for ESP32 integration"""
        model_size = len(tflite_model)
        
        header_content = f"""#ifndef AIR_QUALITY_MODEL_H
#define AIR_QUALITY_MODEL_H

#include <stdint.h>

// Model generated for ESP32-S3 TinyML deployment
// Dataset: Air Quality with NO2, O3, PM2.5 measurements
// Model size: {model_size} bytes

const unsigned int air_quality_model_len = {model_size};

// Air quality class labels
const char* air_quality_labels[] = {{
"""
        
        # Add class labels
        for i, label in enumerate(self.label_encoder.classes_):
            header_content += f'    "{label}"'
            if i < len(self.label_encoder.classes_) - 1:
                header_content += ","
            header_content += "\n"
        
        header_content += "};\n\n"
        
        # Add feature information
        header_content += f"// Number of input features: {len(self.feature_columns)}\n"
        header_content += f"const unsigned int num_features = {len(self.feature_columns)};\n"
        header_content += f"const unsigned int num_classes = {len(self.label_encoder.classes_)};\n\n"
        
        header_content += "// Feature names for reference\n"
        header_content += "const char* feature_names[] = {\n"
        for i, feature in enumerate(self.feature_columns):
            header_content += f'    "{feature}"'
            if i < len(self.feature_columns) - 1:
                header_content += ","
            header_content += "\n"
        header_content += "};\n\n"
        
        # Add preprocessing information
        header_content += "// Preprocessing notes:\n"
        header_content += "// 1. Your dataset appears to be pre-normalized\n"
        header_content += "// 2. Apply the saved scaler.pkl for consistency\n"
        header_content += "// 3. Handle derived features as per training pipeline\n\n"
        
        # Add model data
        header_content += "// TensorFlow Lite model data\n"
        header_content += "const unsigned char air_quality_model[] = {\n"
        
        # Convert bytes to hex array
        for i in range(0, len(tflite_model), 16):
            header_content += "    "
            for j in range(16):
                if i + j < len(tflite_model):
                    header_content += f"0x{tflite_model[i + j]:02x}"
                    if i + j < len(tflite_model) - 1:
                        header_content += ", "
            header_content += "\n"
        
        header_content += "};\n\n#endif // AIR_QUALITY_MODEL_H\n"
        
        # Save header file
        with open('air_quality_model.h', 'w') as f:
            f.write(header_content)

if __name__ == "__main__":
    # Initialize classifier
    classifier = AirQualityClassifier()
    
    # Train the model with your dataset
    data_file = 'openaq_aqi_dataset.csv'  # Replace with your actual file
    
    if not os.path.exists(data_file):
        print(f"Dataset file '{data_file}' not found")
    try:
        print("Starting training with your air quality dataset...")
        history = classifier.train(data_file, epochs=75)
        
        if history is not None:
            print("\nTraining completed successfully!")
            print("\nGenerated files:")
            print("1. air_quality_model.h5 - Keras model")
            print("2. air_quality_model.tflite - TensorFlow Lite model")
            print("3. air_quality_model.h - C header for ESP32")
            print("4. scaler.pkl - Feature scaler")
            print("5. label_encoder.pkl - Label encoder")
            print("6. feature_columns.pkl - Feature list")
            print("7. Visualization plots (PNG files)")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please check your dataset format and try again.")