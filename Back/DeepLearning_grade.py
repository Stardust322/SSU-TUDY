import sys
import json
import traceback
import os
import time
import sqlite3
import pandas as pd
import numpy as np

import os
db_absolute_path = r'C:\Programming\C_C++\src\grades.db'
print(f"Current working directory: {os.getcwd()}")
print(f"Database absolute path: {db_absolute_path}")

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

def safe_print(message):
    try:
        print(message, flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"Output error: {e}", flush=True)

def debug_environment():
    safe_print("=== Python Environment Info ===")
    safe_print(f"Python version: {sys.version}")
    safe_print(f"Working directory: {os.getcwd()}")
    safe_print(f"Python executable: {sys.executable}")
    safe_print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"Database absolute path: {db_absolute_path}")
    safe_print(f"grades.db exists: {os.path.exists(db_absolute_path)}")
    
def test_basic_functions():
    safe_print("=== Basic Function Test ===")
    
    test_json = {"test": "success", "number": 42}
    safe_print(f"JSON test: {json.dumps(test_json)}")

def check_dependencies():
    safe_print("=== Library Dependency Check ===")
    
    dependencies = {
        'sqlite3': True,
        'pandas': False,
        'numpy': False,
        'sklearn': False,
        'tensorflow': False
    }
    

    try:
        import sqlite3
        safe_print("✅ sqlite3: Available")
        dependencies['sqlite3'] = True
    except ImportError as e:
        safe_print(f"❌ sqlite3: {e}")
        dependencies['sqlite3'] = False
    
    try:
        import pandas as pd
        safe_print(f"✅ pandas: {pd.__version__}")
        dependencies['pandas'] = True
    except ImportError as e:
        safe_print(f"❌ pandas: {e}")
        dependencies['pandas'] = False
    
    try:
        import numpy as np
        safe_print(f"✅ numpy: {np.__version__}")
        dependencies['numpy'] = True
    except ImportError as e:
        safe_print(f"❌ numpy: {e}")
        dependencies['numpy'] = False
    
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import sklearn
        safe_print(f"✅ scikit-learn: {sklearn.__version__}")
        dependencies['sklearn'] = True
    except ImportError as e:
        safe_print(f"⚠️ scikit-learn not available: {e}")
        dependencies['sklearn'] = False
    
    try:
        import tensorflow as tf
        safe_print(f"✅ TensorFlow: {tf.__version__}")
        dependencies['tensorflow'] = True
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        
    except ImportError as e:
        safe_print(f"⚠️ TensorFlow not available: {e}")
        dependencies['tensorflow'] = False

    if not dependencies['sqlite3']:
        raise RuntimeError("SQLite3 is required")
    
    safe_print(f"Dependency check complete: {dependencies}")
    return dependencies

class DeepLearningGradePredictor:
    def __init__(self, db_path=None):
        if db_path is None:
            self.db_path = db_absolute_path
        else:
            self.db_path = db_path
        self.model = None
        self.scaler = None
        self.completed_df = None
        self.current_df = None
        
        if not os.path.exists(self.db_path):
            safe_print(f"❌ Database file not found: {self.db_path}")
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        else:
            safe_print(f"✅ Database file confirmed: {self.db_path}")
        
    def load_data(self, student_id):
        safe_print(f"=== Loading data for student {student_id} ===")
        
        try:
            conn = sqlite3.connect(self.db_path)
            safe_print(f"✅ Database connection successful: {self.db_path}")
            
            completed_query = """
            SELECT student_id, subject, subject_type, category, credit, grade_point, status
            FROM grades 
            WHERE student_id = ? AND status = 'completed' AND grade_point > 0
            """
            
            current_query = """
            SELECT student_id, subject, subject_type, category, credit, grade_point, status
            FROM grades 
            WHERE student_id = ? AND status = 'current'
            """
            
            self.completed_df = pd.read_sql_query(completed_query, conn, params=(student_id,))
            self.current_df = pd.read_sql_query(current_query, conn, params=(student_id,))
            
            conn.close()
            
            safe_print(f"✅ Completed subjects: {len(self.completed_df)}")
            safe_print(f"✅ Current subjects: {len(self.current_df)}")
            
            # Data preview
            if len(self.completed_df) > 0:
                safe_print("Completed subjects preview:")
                safe_print(str(self.completed_df.head()))
            
            if len(self.current_df) > 0:
                safe_print("Current subjects preview:")
                safe_print(str(self.current_df.head()))
            
            return len(self.completed_df) > 0 and len(self.current_df) > 0
            
        except Exception as e:
            safe_print(f"❌ Data loading failed: {e}")
            safe_print(f"Error details: {traceback.format_exc()}")
            return False
    
    def create_features(self, df):
        safe_print("=== Feature Engineering ===")
        
        features = []
        
        for idx, row in df.iterrows():
            feature_vector = []
            
            subject_type_map = {'전공': 2, '교양필수': 1, '교양선택': 0}
            feature_vector.append(subject_type_map.get(row['subject_type'], 0))
            
            category_map = {'시험': 3, '플젝': 2, '팀플': 1, '출석': 0}
            feature_vector.append(category_map.get(row['category'], 0))
            

            feature_vector.append(row['credit'])
            
            if self.completed_df is not None and len(self.completed_df) > 0:
                same_type_avg = self.completed_df[
                    self.completed_df['subject_type'] == row['subject_type']
                ]['grade_point'].mean()
                feature_vector.append(same_type_avg if not pd.isna(same_type_avg) else 3.5)
                
                same_category_avg = self.completed_df[
                    self.completed_df['category'] == row['category']
                ]['grade_point'].mean()
                feature_vector.append(same_category_avg if not pd.isna(same_category_avg) else 3.5)
                
                overall_gpa = (self.completed_df['grade_point'] * self.completed_df['credit']).sum() / self.completed_df['credit'].sum()
                feature_vector.append(overall_gpa)
                
                grade_std = self.completed_df['grade_point'].std()
                feature_vector.append(grade_std if not pd.isna(grade_std) else 0.5)
                
            else:
                feature_vector.extend([3.5, 3.5, 3.5, 0.5])
            
            features.append(feature_vector)
        
        safe_print(f"✅ Feature vector generation complete: {len(features)} samples, dimension: {len(features[0]) if features else 0}")
        return np.array(features)
    
    def prepare_training_data(self):
        safe_print("=== Preparing Training Data ===")
        
        if len(self.completed_df) < 3:
            safe_print("❌ Insufficient training data (minimum 3 required) ")
            return None, None
        
        X = self.create_features(self.completed_df)
        y = self.completed_df['grade_point'].values
        
        X_augmented = []
        y_augmented = []
        
        for i in range(len(X)):
            X_augmented.append(X[i])
            y_augmented.append(y[i])
            
            for _ in range(3):
                noise = np.random.normal(0, 0.1, X[i].shape)
                X_augmented.append(X[i] + noise)
                y_augmented.append(y[i])
        
        X_final = np.array(X_augmented)
        y_final = np.array(y_augmented)
        
        safe_print(f"✅ Data augmentation complete: {len(X_final)} samples")
        
        try:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_final)
            safe_print("✅ Data normalization complete")
            return X_scaled, y_final
        except ImportError:
            safe_print("⚠️ StandardScaler not available, skipping normalization")
            return X_final, y_final
    
    def build_model(self, input_dim):
        safe_print("=== Building Deep Learning Model ===")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_dim,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='linear')  
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            safe_print("✅ Deep learning model construction complete")
            return model
            
        except ImportError as e:
            safe_print(f"❌ TensorFlow not available: {e}")
            return None
    
    def train_model(self):
        safe_print("=== Model Training Started ===")
        
        X, y = self.prepare_training_data()
        if X is None:
            return False
        
        self.model = self.build_model(X.shape[1])
        if self.model is None:
            return False
        
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            from sklearn.model_selection import train_test_split
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            safe_print(f"Training data: {len(X_train)}, Validation data: {len(X_val)}")
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=400,
                batch_size=8,
                callbacks=[early_stopping],
                verbose=0
            )
            
            safe_print(f"✅ Model training complete (epochs: {len(history.history['loss'])}) ")
            
            final_loss = history.history['val_loss'][-1]
            final_mae = history.history['val_mae'][-1]
            safe_print(f"Final validation loss: {final_loss:.4f}")
            safe_print(f"Final validation MAE: {final_mae:.4f}")
            
            return True
            
        except ImportError as e:
            safe_print(f"❌ Training libraries not available: {e}")
            return False
        except Exception as e:
            safe_print(f"❌ Model training failed: {e}")
            safe_print(f"Error details: {traceback.format_exc()}")
            return False
    
    def predict_current_grades(self):
        safe_print("=== Performing Grade Prediction ===")
        
        if self.model is None:
            safe_print("❌ No trained model available")
            return None
        
        if len(self.current_df) == 0:
            safe_print("❌ No current subjects to predict")
            return None
        
        try:
            X_current = self.create_features(self.current_df)
            
            if self.scaler is not None:
                X_current_scaled = self.scaler.transform(X_current)
            else:
                X_current_scaled = X_current
            
            predictions = self.model.predict(X_current_scaled, verbose=0)
            
            predictions = np.clip(predictions.flatten(), 0.0, 4.5)
            
            safe_print(f"✅ Prediction complete: {len(predictions)} subjects")
            
            return predictions
            
        except Exception as e:
            safe_print(f"❌ Prediction failed: {e}")
            safe_print(f"Error details: {traceback.format_exc()}")
            return None
    
    def calculate_confidence(self, predictions):
        if self.completed_df is None or len(self.completed_df) == 0:
            return [75] * len(predictions)
        
        current_gpa = (self.completed_df['grade_point'] * self.completed_df['credit']).sum() / self.completed_df['credit'].sum()
        confidences = []
        
        for pred in predictions:
            diff = abs(pred - current_gpa)
            confidence = max(60, min(95, 100 - diff * 15))
            confidences.append(int(confidence))
        
        return confidences

def gpa_to_grade(gpa):
    if gpa >= 4.3: return "A+"
    elif gpa >= 4.0: return "A0"
    elif gpa >= 3.7: return "A-"
    elif gpa >= 3.3: return "B+"
    elif gpa >= 3.0: return "B0"
    elif gpa >= 2.7: return "B-"
    elif gpa >= 2.3: return "C+"
    elif gpa >= 2.0: return "C0"
    elif gpa >= 1.7: return "C-"
    elif gpa >= 1.3: return "D+"
    elif gpa >= 1.0: return "D0"
    else: return "F"

def safe_convert_to_python(value):
    if hasattr(value, 'item'):  
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif pd.isna(value):
        return None
    else:
        return value

def create_deep_learning_result(student_id, dependencies):
    safe_print("=== Generating Deep Learning Prediction Result ===")
    
    try:
        predictor = DeepLearningGradePredictor(db_absolute_path)
        
        if not predictor.load_data(student_id):
            return {
                "success": False,
                "error": "Cannot load student data",
                "model_type": "data_load_failed"
            }
        
        if not dependencies['tensorflow'] or not dependencies['sklearn']:
            safe_print("⚠️ Deep learning libraries insufficient, using basic prediction")
            return create_basic_fallback_result(student_id, predictor.completed_df, predictor.current_df)
        
        if not predictor.train_model():
            safe_print("⚠️ Model training failed, using basic prediction")
            return create_basic_fallback_result(student_id, predictor.completed_df, predictor.current_df)
        

        predictions = predictor.predict_current_grades()
        if predictions is None:
            return {
                "success": False,
                "error": "Grade prediction failed",
                "model_type": "prediction_failed"
            }
        
        confidences = predictor.calculate_confidence(predictions)
        
        prediction_results = []
        semester_points = 0.0
        semester_credits = 0.0
        
        for i, (_, row) in enumerate(predictor.current_df.iterrows()):
            pred_gpa = safe_convert_to_python(predictions[i])
            credit = safe_convert_to_python(row['credit'])
            confidence = safe_convert_to_python(confidences[i])
            
            pred_result = {
                "subject": str(row['subject']),
                "subject_type": str(row['subject_type']),
                "category": str(row['category']) if pd.notna(row['category']) else "Unclassified",
                "credit": int(credit),
                "predicted_gpa": round(float(pred_gpa), 2),
                "predicted_grade": gpa_to_grade(pred_gpa),
                "confidence": int(confidence)
            }
            prediction_results.append(pred_result)
            
            semester_points += float(pred_gpa) * float(credit)
            semester_credits += float(credit)
        
        past_total_points = safe_convert_to_python((predictor.completed_df['grade_point'] * predictor.completed_df['credit']).sum())
        past_total_credits = safe_convert_to_python(predictor.completed_df['credit'].sum())
        past_gpa = float(past_total_points) / float(past_total_credits) if past_total_credits > 0 else 0.0
        
        semester_gpa = semester_points / semester_credits if semester_credits > 0 else 0.0
        total_gpa = (float(past_total_points) + semester_points) / (float(past_total_credits) + semester_credits)
        
        min_pred_idx = int(np.argmin(predictions))
        weak_subject = str(predictor.current_df.iloc[min_pred_idx]['subject'])
        weak_category_val = predictor.current_df.iloc[min_pred_idx]['category']
        weak_category = str(weak_category_val) if pd.notna(weak_category_val) else "Unclassified"
        
        improvement_suggestions = [
            f"🎯 {weak_subject} 과목에 더 집중하세요",
            f"📚 {weak_category} 분야 학습 방법을 개선하세요",
            "💪 꾸준한 학습으로 예측 성적을 향상시킬 수 있습니다"
        ]
        
        gpa_change = total_gpa - past_gpa
        if gpa_change > 0:
            improvement_suggestions.append("📈 현재 추세가 좋습니다. 계속 유지하세요!")
        else:
            improvement_suggestions.append("⚠️ 성적 하락이 예상됩니다. 학습 전략을 재검토하세요")
        
        tf_version = "Unknown"
        try:
            import tensorflow as tf
            tf_version = str(tf.__version__)
        except:
            pass
        
        summary = {
            "current_semester_gpa": round(semester_gpa, 2),
            "current_semester_grade": gpa_to_grade(semester_gpa),
            "past_gpa": round(past_gpa, 2),
            "predicted_total_gpa": round(total_gpa, 2),
            "predicted_total_grade": gpa_to_grade(total_gpa),
            "model_used": "Deep Learning Neural Network",
            "past_credits": int(past_total_credits),
            "current_credits": int(semester_credits),
            "total_credits": int(float(past_total_credits) + semester_credits),
            "gpa_change": round(gpa_change, 3),
            "weak_subject": weak_subject,
            "weak_category": weak_category,
            "improvement_suggestions": improvement_suggestions
        }
        
        result = {
            "success": True,
            "predictions": prediction_results,
            "summary": summary,
            "model_type": "deep_learning",
            "data_points": int(len(predictor.completed_df)),
            "ai_confidence": "High",
            "feature_count": 7,
            "tensorflow_version": tf_version,
            "training_samples": int(len(predictor.completed_df) * 4)  
        }
        
        safe_print("✅ Deep learning prediction result generation complete")
        return result
        
    except Exception as e:
        safe_print(f"❌ Deep learning prediction failed: {e}")
        safe_print(f"Error trace: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Deep learning prediction failed: {e}",
            "model_type": "deep_learning_error"
        }

def create_basic_fallback_result(student_id, completed_df, current_df):
    safe_print("=== Basic Prediction Mode ===")
    
    try:
        if len(completed_df) == 0:
            return {
                "success": False,
                "error": "No completed subjects",
                "model_type": "no_data"
            }
        
        if len(current_df) == 0:
            return {
                "success": False,
                "error": "No current subjects",
                "model_type": "no_current_subjects"
            }
        
        total_points = 0.0
        total_credits = 0.0
        
        for _, grade in completed_df.iterrows():
            grade_point = safe_convert_to_python(grade['grade_point'])
            credit = safe_convert_to_python(grade['credit'])
            total_points += float(grade_point) * float(credit)
            total_credits += float(credit)
        
        past_gpa = total_points / total_credits if total_credits > 0 else 0.0
        
        safe_print(f"Past GPA calculation: {total_points}/{total_credits} = {past_gpa}")
        
        predictions = []
        semester_points = 0.0
        semester_credits = 0.0
        
        for _, row in current_df.iterrows():
            predicted_gpa = past_gpa * 0.95  
            predicted_gpa = max(0.0, min(4.5, predicted_gpa))
            
            credit = safe_convert_to_python(row['credit'])
            subject = str(row['subject'])
            subject_type = str(row['subject_type'])
            category = str(row['category']) if pd.notna(row['category']) else "Unclassified"
            
            predictions.append({
                "subject": subject,
                "subject_type": subject_type,
                "category": category,
                "credit": int(float(credit)),
                "predicted_gpa": round(predicted_gpa, 2),
                "predicted_grade": gpa_to_grade(predicted_gpa),
                "confidence": 75
            })
            
            semester_points += predicted_gpa * float(credit)
            semester_credits += float(credit)
        
        semester_gpa = semester_points / semester_credits if semester_credits > 0 else 0.0
        total_gpa = (total_points + semester_points) / (total_credits + semester_credits)
        
        summary = {
            "current_semester_gpa": round(semester_gpa, 2),
            "current_semester_grade": gpa_to_grade(semester_gpa),
            "past_gpa": round(past_gpa, 2),
            "predicted_total_gpa": round(total_gpa, 2),
            "predicted_total_grade": gpa_to_grade(total_gpa),
            "model_used": "Basic Statistical Prediction",
            "past_credits": int(total_credits),
            "current_credits": int(semester_credits),
            "total_credits": int(total_credits + semester_credits),
            "gpa_change": round(total_gpa - past_gpa, 3),
            "weak_subject": predictions[0]["subject"] if predictions else "None",
            "weak_category": "Basic Analysis",
            "improvement_suggestions": [
                "Basic prediction was used",
                "Installing deep learning libraries can improve accuracy",
                "Continuous learning is recommended"
            ]
        }
        
        result = {
            "success": True,
            "predictions": predictions,
            "summary": summary,
            "model_type": "basic_fallback",
            "data_points": int(len(completed_df)),
            "ai_confidence": "Medium",
            "feature_count": 4
        }
        
        safe_print("✅ Basic prediction result generation complete")
        return result
        
    except Exception as e:
        safe_print(f"❌ Basic prediction failed: {e}")
        return {
            "success": False,
            "error": f"Basic prediction failed: {e}",
            "model_type": "critical_error"
        }

def main():
    try:
        safe_print("=" * 60)
        safe_print("🚀 Deep Learning Grade Prediction System Start")
        safe_print("=" * 60)
        
        if len(sys.argv) > 1:
            student_id = sys.argv[1]
        else:
            student_id = ""
            
        safe_print(f"🎯 Student ID: {student_id}")
        
        debug_environment()
        test_basic_functions()
        dependencies = check_dependencies()
        safe_print("=== Deep Learning Prediction Start ===")
        result = create_deep_learning_result(student_id, dependencies)
    
        try:
            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            safe_print(f"JSON length: {len(json_str)} characters")
            safe_print("=== JSON Output Preparation ===")
            safe_print("JSON_START")
            print(json_str)
            
        except Exception as json_error:
            safe_print(f"JSON serialization error: {json_error}")
            minimal_result = {
                "success": False,
                "error": f"JSON serialization failed: {json_error}",
                "model_type": "json_error"
            }
            print(json.dumps(minimal_result))
        
        safe_print("JSON_END")
        
        safe_print("=" * 60)
        safe_print("🎉 Deep Learning Grade Prediction System Complete")
        safe_print("=" * 60)
        
    except Exception as e:
        safe_print(f"❌ Critical error occurred: {e}")
        safe_print(f"Error trace: {traceback.format_exc()}")
        
        try:
            safe_print("JSON_START")
            error_result = {
                "success": False,
                "error": str(e),
                "model_type": "critical_error",
                "traceback": traceback.format_exc()
            }
            print(json.dumps(error_result, ensure_ascii=False))
            safe_print("JSON_END")
        except Exception as final_error:
            safe_print(f"Final safety net also failed: {final_error}")
            print('{"success": false, "error": "complete_failure"}')

if __name__ == "__main__":
    main()
