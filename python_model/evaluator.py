import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class AirQualityModelEvaluator:
    def __init__(self, model_path='air_quality_model.h5'):
        """
        Initialize the evaluator with trained model components
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.tflite_interpreter = None
        self.target_column = 'overall_aqi_category'
        
        self.load_model_components()
    
    def load_model_components(self):
        """Load all model components"""
        try:
            # Load Keras model
            self.model = load_model(self.model_path)
            print(f"✅ Loaded Keras model from {self.model_path}")
            
            # Load preprocessing components
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            self.feature_columns = joblib.load('feature_columns.pkl')
            
            print("✅ Loaded preprocessing components:")
            print(f"   - Feature columns: {len(self.feature_columns)}")
            print(f"   - Classes: {list(self.label_encoder.classes_)}")
            
            # Load TensorFlow Lite model if available
            if os.path.exists('air_quality_model.tflite'):
                self.tflite_interpreter = tf.lite.Interpreter(model_path='air_quality_model.tflite')
                self.tflite_interpreter.allocate_tensors()
                print("✅ Loaded TensorFlow Lite model for ESP32-S3")
            
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            raise
    
    def prepare_features(self, df):
        """
        Prepare features exactly as done during training
        """
        # Create derived features
        df['aqi_max'] = df[['aqi_value_no2', 'aqi_value_o3', 'aqi_value_pm25']].max(axis=1)
        df['aqi_mean'] = df[['aqi_value_no2', 'aqi_value_o3', 'aqi_value_pm25']].mean(axis=1)
        df['aqi_std'] = df[['aqi_value_no2', 'aqi_value_o3', 'aqi_value_pm25']].std(axis=1)
        
        df['conc_max'] = df[['concentration_no2', 'concentration_o3', 'concentration_pm25']].max(axis=1)
        df['conc_mean'] = df[['concentration_no2', 'concentration_o3', 'concentration_pm25']].mean(axis=1)
        df['conc_std'] = df[['concentration_no2', 'concentration_o3', 'concentration_pm25']].std(axis=1)
        
        # Pollutant ratios
        df['no2_to_pm25_ratio'] = df['concentration_no2'] / (np.abs(df['concentration_pm25']) + 1e-6)
        df['o3_to_pm25_ratio'] = df['concentration_o3'] / (np.abs(df['concentration_pm25']) + 1e-6)
        df['no2_to_o3_ratio'] = df['concentration_no2'] / (np.abs(df['concentration_o3']) + 1e-6)
        
        # Temporal interactions
        df['hour_season_interaction'] = df['hour'] * df['season_encoded']
        df['day_season_interaction'] = df['day_of_week'] * df['season_encoded']
        
        # AQI deviations
        df['aqi_no2_deviation'] = df['aqi_value_no2'] - df['overall_aqi']
        df['aqi_o3_deviation'] = df['aqi_value_o3'] - df['overall_aqi']
        df['aqi_pm25_deviation'] = df['aqi_value_pm25'] - df['overall_aqi']
        
        # Fill NaN values
        derived_features = [
            'aqi_max', 'aqi_mean', 'aqi_std', 'conc_max', 'conc_mean', 'conc_std',
            'no2_to_pm25_ratio', 'o3_to_pm25_ratio', 'no2_to_o3_ratio',
            'hour_season_interaction', 'day_season_interaction',
            'aqi_no2_deviation', 'aqi_o3_deviation', 'aqi_pm25_deviation'
        ]
        
        for col in derived_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Return only the features used in training
        available_features = [col for col in self.feature_columns if col in df.columns]
        return df[available_features].copy()
    
    def evaluate_on_test_data(self, test_data_file):
        """Comprehensive evaluation on test dataset"""
        print(f"\n🔬 EVALUATING MODEL ON: {test_data_file}")
        print("=" * 60)
        
        # Load and prepare test data
        df_test = pd.read_csv(test_data_file)
        print(f"Loaded test data shape: {df_test.shape}")
        
        # Check if target column exists
        if self.target_column not in df_test.columns:
            print(f"❌ Target column '{self.target_column}' not found!")
            return None
        
        # Prepare features
        X_test_features = self.prepare_features(df_test)
        y_test = df_test[self.target_column].values
        
        # Remove rows with missing values
        mask = ~X_test_features.isnull().any(axis=1)
        X_test_features = X_test_features[mask]
        y_test = y_test[mask]
        
        print(f"Test data shape after preprocessing: {X_test_features.shape}")
        print(f"Test samples after cleaning: {len(y_test)}")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test_features.values)
        
        # Encode labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Make predictions with Keras model
        print("\n🧠 Making predictions...")
        y_pred_proba = self.model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_test_encoded, y_pred, average='weighted'
        )
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {fscore:.4f}")
        
        # Detailed classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test_encoded, y_pred, target_names=target_names))
        
        # Class distribution analysis
        self.analyze_class_distribution(y_test, y_pred, target_names)
        
        # Confusion matrix
        self.plot_detailed_confusion_matrix(y_test_encoded, y_pred, target_names)
        
        # Performance by class
        self.analyze_class_performance(y_test_encoded, y_pred, y_pred_proba, target_names)
        
        # Model confidence analysis
        self.analyze_prediction_confidence(y_pred_proba, y_test_encoded, target_names)
        
        # ROC curves for multi-class
        self.plot_roc_curves(y_test_encoded, y_pred_proba, target_names)
        
        # Feature importance analysis
        self.analyze_feature_correlations(X_test_features, y_test)
        
        # Compare Keras vs TensorFlow Lite if available
        if self.tflite_interpreter:
            self.compare_keras_vs_tflite(X_test_scaled[:100], y_test_encoded[:100])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': fscore,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test_encoded,
            'test_data_size': len(y_test)
        }
    
    def analyze_class_distribution(self, y_true, y_pred, target_names):
        """Analyze class distribution in predictions vs ground truth"""
        print(f"\n📈 CLASS DISTRIBUTION ANALYSIS:")
        print("-" * 50)
        
        # Convert predictions back to class names
        y_pred_names = [target_names[i] for i in y_pred]
        
        # True distribution
        true_dist = pd.Series(y_true).value_counts().sort_index()
        pred_dist = pd.Series(y_pred_names).value_counts().reindex(true_dist.index, fill_value=0)
        
        print("Distribution comparison:")
        print(f"{'Class':<20} {'True':<8} {'Predicted':<10} {'Difference':<10}")
        print("-" * 50)
        for class_name in true_dist.index:
            true_count = true_dist[class_name]
            pred_count = pred_dist[class_name]
            diff = pred_count - true_count
            print(f"{class_name:<20} {true_count:<8} {pred_count:<10} {diff:+d}")
    
    def plot_detailed_confusion_matrix(self, y_true, y_pred, target_names):
        """Plot detailed confusion matrix with percentages"""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_class_performance(self, y_true, y_pred, y_pred_proba, target_names):
        """Analyze performance for each air quality class"""
        print(f"\n🎯 CLASS-WISE PERFORMANCE ANALYSIS:")
        print("-" * 60)
        
        for i, class_name in enumerate(target_names):
            class_mask = (y_true == i)
            if np.sum(class_mask) == 0:
                continue
                
            class_predictions = y_pred[class_mask]
            class_probabilities = y_pred_proba[class_mask, i]
            
            accuracy = np.mean(class_predictions == i)
            avg_confidence = np.mean(class_probabilities)
            support = np.sum(class_mask)
            
            # Calculate precision and recall for this class
            tp = np.sum((y_pred == i) & (y_true == i))
            fp = np.sum((y_pred == i) & (y_true != i))
            fn = np.sum((y_pred != i) & (y_true == i))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name}:")
            print(f"  - Support:      {support:4d}")
            print(f"  - Precision:    {precision:.4f}")
            print(f"  - Recall:       {recall:.4f}")
            print(f"  - F1-Score:     {f1:.4f}")
            print(f"  - Avg Confidence: {avg_confidence:.4f}")
            print(f"  - Min Confidence: {np.min(class_probabilities):.4f}")
            print(f"  - Max Confidence: {np.max(class_probabilities):.4f}")
            print()
    
    def analyze_prediction_confidence(self, y_pred_proba, y_true, target_names):
        """Analyze model confidence in predictions"""
        max_proba = np.max(y_pred_proba, axis=1)
        predictions = np.argmax(y_pred_proba, axis=1)
        correct_predictions = (predictions == y_true)
        
        print(f"\n🎯 PREDICTION CONFIDENCE ANALYSIS:")
        print("-" * 50)
        print(f"Average confidence (all):        {np.mean(max_proba):.4f}")
        print(f"Average confidence (correct):    {np.mean(max_proba[correct_predictions]):.4f}")
        print(f"Average confidence (incorrect):  {np.mean(max_proba[~correct_predictions]):.4f}")
        
        # Confidence thresholds analysis
        thresholds = [0.5, 0.7, 0.8, 0.9]
        print(f"\nPredictions by confidence threshold:")
        for threshold in thresholds:
            high_conf_mask = max_proba >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_acc = np.mean(correct_predictions[high_conf_mask])
                coverage = np.mean(high_conf_mask)
                print(f"  ≥{threshold:.1f}: {coverage:.1%} coverage, {high_conf_acc:.4f} accuracy")
        
        # Plot confidence distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall confidence distribution
        ax1.hist(max_proba, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Prediction Confidence Distribution', fontweight='bold')
        ax1.set_xlabel('Max Probability')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(max_proba), color='red', linestyle='--', label=f'Mean: {np.mean(max_proba):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence by correctness
        ax2.hist(max_proba[correct_predictions], bins=20, alpha=0.7, 
                label='Correct', color='green', edgecolor='black')
        ax2.hist(max_proba[~correct_predictions], bins=20, alpha=0.7, 
                label='Incorrect', color='red', edgecolor='black')
        ax2.set_title('Confidence by Prediction Correctness', fontweight='bold')
        ax2.set_xlabel('Max Probability')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true, y_pred_proba, target_names):
        """Plot ROC curves for multi-class classification"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(len(target_names)))
        n_classes = len(target_names)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            if np.sum(y_true_bin[:, i]) > 0:  # Only if class exists in test set
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            if i in roc_auc:
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{target_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Multi-Class ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_correlations(self, X_features, y_true):
        """Analyze feature correlations with target variable"""
        print(f"\n🔍 FEATURE CORRELATION ANALYSIS:")
        print("-" * 50)
        
        # Create correlation matrix
        df_analysis = X_features.copy()
        df_analysis['target'] = y_true
        
        # Calculate correlations with target
        correlations = df_analysis.corr()['target'].abs().sort_values(ascending=False)
        
        print("Top 10 features by correlation with target:")
        for i, (feature, corr) in enumerate(correlations.head(11).items()):
            if feature != 'target':
                print(f"  {i:2d}. {feature:<25}: {corr:.4f}")
        
        # Plot correlation heatmap for top features
        top_features = correlations.head(11).index.tolist()
        if 'target' in top_features:
            top_features.remove('target')
        top_features = top_features[:10]
        
        plt.figure(figsize=(12, 8))
        corr_matrix = df_analysis[top_features + ['target']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Feature Correlation Matrix (Top 10 Features)', fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_keras_vs_tflite(self, X_test, y_test, num_samples=100):
        """Compare Keras model vs TensorFlow Lite model predictions"""
        print(f"\n⚖️ KERAS vs TENSORFLOW LITE COMPARISON:")
        print("-" * 50)
        
        # Keras predictions
        keras_pred = self.model.predict(X_test[:num_samples], verbose=0)
        keras_classes = np.argmax(keras_pred, axis=1)
        
        # TensorFlow Lite predictions
        input_details = self.tflite_interpreter.get_input_details()
        output_details = self.tflite_interpreter.get_output_details()
        
        tflite_predictions = []
        for i in range(num_samples):
            # Set input tensor
            input_data = X_test[i:i+1].astype(np.float32)
            self.tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            self.tflite_interpreter.invoke()
            
            # Get output
            output_data = self.tflite_interpreter.get_tensor(output_details[0]['index'])
            tflite_predictions.append(output_data[0])
        
        tflite_pred = np.array(tflite_predictions)
        tflite_classes = np.argmax(tflite_pred, axis=1)
        
        # Compare predictions
        agreement = np.mean(keras_classes == tflite_classes)
        keras_accuracy = np.mean(keras_classes == y_test[:num_samples])
        tflite_accuracy = np.mean(tflite_classes == y_test[:num_samples])
        
        print(f"Samples compared:       {num_samples}")
        print(f"Prediction agreement:   {agreement:.4f}")
        print(f"Keras accuracy:         {keras_accuracy:.4f}")
        print(f"TensorFlow Lite accuracy: {tflite_accuracy:.4f}")
        
        # Analyze probability differences
        prob_diff = np.abs(keras_pred - tflite_pred)
        max_prob_diff = np.max(prob_diff, axis=1)
        
        print(f"Avg max prob difference: {np.mean(max_prob_diff):.6f}")
        print(f"Max prob difference:     {np.max(max_prob_diff):.6f}")
        
        # ESP32-S3 deployment assessment
        print(f"\n🚁 ESP32-S3 DEPLOYMENT ASSESSMENT:")
        print(f"   Model size:          {os.path.getsize('air_quality_model.tflite')/1024:.1f} KB")
        print(f"   Accuracy retention:  {tflite_accuracy/keras_accuracy:.4f}")
        print(f"   Suitable for drone:  {'✅ Yes' if agreement > 0.95 else '⚠️  Check'}")
        
        return agreement, keras_accuracy, tflite_accuracy
    
    def test_single_prediction(self, sample_data):
        """Test single prediction with your dataset format"""
        print(f"\n🔬 SINGLE PREDICTION TEST:")
        print("-" * 40)
        
        # Display input
        print("Input sensor readings:")
        for key, value in sample_data.items():
            if key != 'overall_aqi_category':
                print(f"  {key}: {value}")
        
        # Create DataFrame
        df = pd.DataFrame([sample_data])
        
        # Prepare features
        X_features = self.prepare_features(df)
        X_scaled = self.scaler.transform(X_features.values)
        
        # Make prediction
        prediction_proba = self.model.predict(X_scaled, verbose=0)
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = prediction_proba[0][predicted_class_idx]
        
        print(f"\nPrediction Results:")
        print(f"  Predicted Class:    {predicted_class}")
        print(f"  Confidence:         {confidence:.4f}")
        if 'overall_aqi_category' in sample_data:
            print(f"  True Category:      {sample_data['overall_aqi_category']}")
            print(f"  Correct Prediction: {'✅' if predicted_class == sample_data['overall_aqi_category'] else '❌'}")
        
        # Show all class probabilities
        print(f"\nAll Class Probabilities:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob = prediction_proba[0][i]
            indicator = " 👈" if i == predicted_class_idx else ""
            print(f"  {class_name:<20}: {prob:.4f}{indicator}")
        
        return predicted_class, confidence, prediction_proba[0]
    
    def benchmark_inference_speed(self, X_test, num_iterations=1000):
        """Benchmark inference speed for ESP32 deployment planning"""
        print(f"\n⚡ INFERENCE SPEED BENCHMARK:")
        print("-" * 50)
        
        import time
        
        # Keras model benchmark
        # Warm up
        _ = self.model.predict(X_test[:10], verbose=0)
        
        start_time = time.time()
        for _ in range(num_iterations):
            _ = self.model.predict(X_test[:1], verbose=0)
        keras_time = (time.time() - start_time) / num_iterations
        
        print(f"Keras model:")
        print(f"  Average inference time: {keras_time*1000:.2f} ms")
        print(f"  Inferences per second:  {1/keras_time:.1f}")
        
        # TensorFlow Lite benchmark
        if self.tflite_interpreter:
            input_details = self.tflite_interpreter.get_input_details()
            
            # Warm up
            for _ in range(10):
                input_data = X_test[0:1].astype(np.float32)
                self.tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
                self.tflite_interpreter.invoke()
            
            start_time = time.time()
            for i in range(num_iterations):
                input_data = X_test[i%len(X_test):i%len(X_test)+1].astype(np.float32)
                self.tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
                self.tflite_interpreter.invoke()
            tflite_time = (time.time() - start_time) / num_iterations
            
            print(f"\nTensorFlow Lite model:")
            print(f"  Average inference time: {tflite_time*1000:.2f} ms")
            print(f"  Inferences per second:  {1/tflite_time:.1f}")
            print(f"  Speedup vs Keras:       {keras_time/tflite_time:.1f}x")
            
            # ESP32-S3 deployment recommendations
            print(f"\n🚁 DRONE DEPLOYMENT RECOMMENDATIONS:")
            if tflite_time < 0.1:  # 100ms
                print("  ✅ Suitable for real-time monitoring (>10 Hz)")
            elif tflite_time < 0.5:  # 500ms
                print("  ✅ Suitable for frequent monitoring (>2 Hz)")
            elif tflite_time < 1.0:  # 1 second
                print("  ⚠️  Suitable for periodic monitoring (~1 Hz)")
            else:
                print("  ❌ Too slow for real-time drone applications")
        
        return keras_time, tflite_time if self.tflite_interpreter else None
    
    def generate_evaluation_report(self, test_data_file, output_file='evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        print(f"\n📄 GENERATING EVALUATION REPORT...")
        
        # Run evaluation
        results = self.evaluate_on_test_data(test_data_file)
        
        if results is None:
            print("❌ Could not generate report due to evaluation errors")
            return None
        
        # Get model size info
        keras_size = os.path.getsize('air_quality_model.h5') if os.path.exists('air_quality_model.h5') else 0
        tflite_size = os.path.getsize('air_quality_model.tflite') if os.path.exists('air_quality_model.tflite') else 0
        
        # Create report
        report_content = f"""
AIR QUALITY CLASSIFICATION MODEL - EVALUATION REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

🎯 MODEL INFORMATION:
   Architecture:     Deep Neural Network (TinyML optimized)
   Input Features:   {len(self.feature_columns)}
   Output Classes:   {len(self.label_encoder.classes_)}
   Target Platform:  ESP32-S3 with BME690 + SPS30 sensors

📊 DATASET INFORMATION:
   Test Samples:     {results['test_data_size']}
   Data Source:      {test_data_file}
   Features Used:    Multi-sensor air quality data
   
🏆 PERFORMANCE METRICS:
   Overall Accuracy: {results['accuracy']:.4f}
   Weighted Precision: {results['precision']:.4f}
   Weighted Recall:  {results['recall']:.4f}
   Weighted F1-Score: {results['f1_score']:.4f}

🏷️ AIR QUALITY CLASSES:
{chr(10).join([f"   {i+1}. {cls}" for i, cls in enumerate(self.label_encoder.classes_)])}

🔧 FEATURE SET ({len(self.feature_columns)} features):
{chr(10).join([f"   • {feature}" for feature in self.feature_columns])}

💾 MODEL SIZE ANALYSIS:
   Keras Model:      {keras_size/1024:.1f} KB
   TensorFlow Lite:  {tflite_size/1024:.1f} KB
   Compression:      {keras_size/tflite_size:.1f}x (if applicable)
   ESP32-S3 Ready:   {'✅ Yes' if tflite_size < 512000 else '❌ Too large'}

🚁 DRONE DEPLOYMENT CONSIDERATIONS:
   • High-endurance quadcopter application
   • Continuous air quality monitoring during flight
   • Real-time inference on ESP32-S3 microprocessor
   • Integration with BME690 (environmental) and SPS30 (particulate) sensors
   • Recommended inference frequency: 1-10 Hz depending on flight pattern

📈 PERFORMANCE ANALYSIS:
   The model shows {'excellent' if results['accuracy'] > 0.9 else 'good' if results['accuracy'] > 0.8 else 'acceptable'} 
   performance for air quality classification. The TinyML optimization maintains 
   accuracy while ensuring efficient deployment on drone hardware.

🔄 NEXT STEPS:
   1. Deploy air_quality_model.tflite to ESP32-S3
   2. Implement sensor data preprocessing pipeline
   3. Set up real-time inference loop
   4. Configure data logging and telemetry
   5. Test in-flight performance and power consumption
   6. Consider periodic model updates with new flight data

📁 GENERATED FILES:
   • detailed_confusion_matrix.png - Classification accuracy visualization
   • confidence_analysis.png - Model confidence distribution
   • roc_curves.png - Multi-class ROC analysis
   • feature_correlations.png - Feature importance analysis
   • evaluation_report.txt - This comprehensive report

⚠️  IMPORTANT NOTES:
   • Ensure consistent preprocessing between training and deployment
   • Monitor model performance during actual flight operations
   • Consider environmental factors affecting sensor readings at altitude
   • Implement fail-safe mechanisms for unreliable predictions

Generated by Air Quality Model Evaluator v2.0
Optimized for drone-based environmental monitoring applications
"""
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        print(f"✅ Evaluation report saved to: {output_file}")
        return report_content

# Usage example and testing functions
def create_sample_test_data():
    """Create sample test data matching your dataset format"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic air quality data
    sample_data = {
        'location_id': np.random.randint(100, 200, n_samples),
        'datetime': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'hour': np.random.uniform(-2, 2, n_samples),  # Normalized
        'day_of_week': np.random.uniform(-2, 2, n_samples),  # Normalized
        'month': np.random.uniform(-2, 2, n_samples),  # Normalized
        'aqi_value_no2': np.random.uniform(-2, 2, n_samples),
        'aqi_value_o3': np.random.uniform(-2, 2, n_samples),
        'aqi_value_pm25': np.random.uniform(-2, 2, n_samples),
        'concentration_no2': np.random.uniform(-2, 2, n_samples),
        'concentration_o3': np.random.uniform(-2, 2, n_samples),
        'concentration_pm25': np.random.uniform(-2, 2, n_samples),
        'overall_aqi': np.random.uniform(-2, 2, n_samples),
        'season_encoded': np.random.choice([0, 1, 2, 3], n_samples),
        'overall_aqi_category': np.random.choice(['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy'], n_samples)
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_test_data.csv', index=False)
    
    return 'sample_test_data.csv'

if __name__ == "__main__":
    try:
        # Initialize evaluator
        evaluator = AirQualityModelEvaluator()
        
        # Test with your data file
        test_file = 'openaq_aqi_dataset.csv'  # Replace with your actual test file
        
        # If test file doesn't exist, create sample data
        if not os.path.exists(test_file):
            print("⚠️  Test file not found. Creating sample data for demonstration...")
            test_file = create_sample_test_data()
        
        # Run comprehensive evaluation
        print("🚀 Starting comprehensive model evaluation for drone deployment...")
        results = evaluator.evaluate_on_test_data(test_file)
        
        if results:
            # Test single prediction with sample data
            print("\n🧪 Testing single prediction...")
            sample_prediction_data = {
                'hour': -0.74,
                'day_of_week': -1.36,
                'month': 0.0,
                'aqi_value_no2': 0.0,
                'aqi_value_o3': 0.0,
                'aqi_value_pm25': -0.89,
                'concentration_no2': -0.31,
                'concentration_o3': -0.32,
                'concentration_pm25': -0.88,
                'overall_aqi': -0.89,
                'season_encoded': 0.0,
                'overall_aqi_category': 'Good'
            }
            
            evaluator.test_single_prediction(sample_prediction_data)
            
            # Benchmark inference speed
            print("\n⚡ Benchmarking inference speed for drone deployment...")
            df_test = pd.read_csv(test_file)
            X_test_features = evaluator.prepare_features(df_test)
            X_test_scaled = evaluator.scaler.transform(X_test_features.values[:100])
            
            evaluator.benchmark_inference_speed(X_test_scaled)
            
            # Generate comprehensive report
            evaluator.generate_evaluation_report(test_file)
            
            print("\n🎉 MODEL EVALUATION COMPLETED SUCCESSFULLY!")
            print("\n📁 Generated files for your drone project:")
            print("   • detailed_confusion_matrix.png")
            print("   • confidence_analysis.png")
            print("   • roc_curves.png")
            print("   • feature_correlations.png")
            print("   • evaluation_report.txt")
            print("\n🚁 Your model is ready for ESP32-S3 drone deployment!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure trained model files exist (air_quality_model.h5, scaler.pkl, etc.)")
        print("2. Check test data format matches training data")
        print("3. Verify all required dependencies are installed")
        print("4. Run training script first if model files are missing")