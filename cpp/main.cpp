#include "httplib.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <Python.h>
#include <string>
#include <windows.h>
#include <vector>
#include <sstream>
#include <map>
#include <sqlite3.h>
#include <regex>
#include <algorithm>

using json = nlohmann::json;

std::string id, pw;
std::vector<std::string> x;

sqlite3* db;

std::string pointToGrade(double point) {
    if (point >= 4.40) return "A+";
    else if (point >= 4.15) return "A0";
    else if (point >= 3.75) return "A-";
    else if (point >= 3.40) return "B+";
    else if (point >= 3.15) return "B0";
    else if (point >= 2.75) return "B-";
    else if (point >= 2.40) return "C+";
    else if (point >= 2.15) return "C0";
    else if (point >= 1.75) return "C-";
    else if (point >= 1.40) return "D+";
    else if (point >= 1.15) return "D0";
    else if (point >= 0.5) return "D-";
    else return "F";
}



double gradeToPoint(const std::string& grade) {
    if (grade == "A+") return 4.5;
    else if (grade == "A0") return 4.3;  
    else if (grade == "A-") return 4.0;  
    else if (grade == "B+") return 3.5;
    else if (grade == "B0") return 3.3;  
    else if (grade == "B-") return 3.0;  
    else if (grade == "C+") return 2.5;
    else if (grade == "C0") return 2.3;  
    else if (grade == "C-") return 2.0;  
    else if (grade == "D+") return 1.5;
    else if (grade == "D0") return 1.3;  
    else if (grade == "D-") return 1.0; 
    else if (grade == "F") return 0.0;
    else return 0.0;
}


std::string unvisible(const std::string& password) {
    return std::string(password.length(), '*');
}

std::map<std::string, std::string> parse_query_string(const std::string& query) {
    std::map<std::string, std::string> result;
    std::istringstream stream(query);
    std::string pair;

    while (std::getline(stream, pair, '&')) {
        auto equal_pos = pair.find('=');
        if (equal_pos != std::string::npos) {
            std::string key = pair.substr(0, equal_pos);
            std::string value = pair.substr(equal_pos + 1);
            result[key] = value;
        }
    }
    return result;
}

std::string get_query_from_target(const std::string& target) {
    auto pos = target.find('?');
    if (pos != std::string::npos) {
        return target.substr(pos + 1);
    }
    return "";
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    size_t last = str.find_last_not_of(" \t\n\r");
    if (first == std::string::npos || last == std::string::npos)
        return "";
    return str.substr(first, last - first + 1);
}

std::string removeExtraSpaces(const std::string& str) {
    if (str.empty()) return str;
    
    std::string result;
    bool prevWasSpace = false;
    
    for (char c : str) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!prevWasSpace && !result.empty()) {
                result += ' ';
                prevWasSpace = true;
            }
        } else {
            result += c;
            prevWasSpace = false;
        }
    }
    
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}

bool hasGrades(const std::string& studentId) {
    sqlite3_stmt* stmt;
    const char* sql = "SELECT COUNT(*) FROM grades WHERE student_id = ?";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return false;
    
    sqlite3_bind_text(stmt, 1, studentId.c_str(), -1, SQLITE_STATIC);
    
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    
    sqlite3_finalize(stmt);
    return count > 0;
}

double getAverage(const json& averages, const std::string& category) {
    if (averages.contains(category) && averages[category].is_number()) {
        return averages[category].get<double>();
    }
    return 0.0;
}

void recreateDatabase() {
    std::cout << "[INFO] 기존 데이터베이스 테이블 삭제 및 재생성 중..." << std::endl;
    
    const char* drop_sql = "DROP TABLE IF EXISTS grades; DROP TABLE IF EXISTS grade_summary;";
    char* errMsg = 0;
    int rc = sqlite3_exec(db, drop_sql, 0, 0, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "테이블 삭제 오류: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
    
    const char* create_sql = R"(
        CREATE TABLE grades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            category TEXT,
            subject_type TEXT NOT NULL,
            grade TEXT,
            credit REAL NOT NULL,
            grade_point REAL DEFAULT 0,
            status TEXT DEFAULT 'completed',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE grade_summary (
            student_id TEXT PRIMARY KEY,
            memory_avg REAL DEFAULT 0,
            teamwork_avg REAL DEFAULT 0,
            project_avg REAL DEFAULT 0,
            attendance_avg REAL DEFAULT 0,
            exam_avg REAL DEFAULT 0,
            total_avg REAL DEFAULT 0
        );
    )";
    
    rc = sqlite3_exec(db, create_sql, 0, 0, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "테이블 생성 오류: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    } else {
        std::cout << "[SUCCESS] 데이터베이스 테이블 재생성 완료" << std::endl;
    }
}

void initDatabase() {
    int rc = sqlite3_open("grades.db", &db);
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        exit(1);
    }

    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS grades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            category TEXT,
            subject_type TEXT NOT NULL,
            grade TEXT,
            credit REAL NOT NULL,
            grade_point REAL DEFAULT 0,
            status TEXT DEFAULT 'completed',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS grade_summary (
            student_id TEXT PRIMARY KEY,
            memory_avg REAL DEFAULT 0,
            teamwork_avg REAL DEFAULT 0,
            project_avg REAL DEFAULT 0,
            attendance_avg REAL DEFAULT 0,
            exam_avg REAL DEFAULT 0,
            total_avg REAL DEFAULT 0
        );
    )";

    char* errMsg = 0;
    rc = sqlite3_exec(db, sql, 0, 0, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    } 
}



json getCurrentSubjects(const std::string& studentId) {
    sqlite3_stmt* stmt;
    const char* sql = R"(
        SELECT subject, subject_type, credit, category, created_at
        FROM grades 
        WHERE student_id = ? AND status = 'current'
        ORDER BY created_at DESC
    )";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    json subjects = json::array();
    
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, studentId.c_str(), -1, SQLITE_STATIC);
        
        int count = 0;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            json subject;
            const char* name = (const char*)sqlite3_column_text(stmt, 0);
            const char* type = (const char*)sqlite3_column_text(stmt, 1);
            double credit = sqlite3_column_double(stmt, 2);
            const char* category = (const char*)sqlite3_column_text(stmt, 3);
            const char* date = (const char*)sqlite3_column_text(stmt, 4);
            
            subject["name"] = name ? name : "";
            subject["type"] = type ? type : "";
            subject["credit"] = credit;
            subject["category"] = category ? category : "";
            subject["added_date"] = date ? date : "";
            
            subjects.push_back(subject);
            count++;
            
        }
        
    } else {
        // std::cout << sqlite3_errmsg(db) << std::endl;
    }
    
    sqlite3_finalize(stmt);
    return subjects;
}

json getAllGrades(const std::string& studentId) {
    sqlite3_stmt* stmt;
    const char* sql = R"(
        SELECT id, subject, subject_type, credit, category, grade, grade_point, status, created_at
        FROM grades 
        WHERE student_id = ?
        ORDER BY status DESC, created_at DESC
    )";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    json grades = json::array();
    
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, studentId.c_str(), -1, SQLITE_STATIC);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            json grade;
            grade["id"] = sqlite3_column_int(stmt, 0);
            grade["subject"] = sqlite3_column_text(stmt, 1) ? (char*)sqlite3_column_text(stmt, 1) : "";
            grade["subject_type"] = sqlite3_column_text(stmt, 2) ? (char*)sqlite3_column_text(stmt, 2) : "";
            grade["credit"] = sqlite3_column_double(stmt, 3);
            grade["category"] = sqlite3_column_text(stmt, 4) ? (char*)sqlite3_column_text(stmt, 4) : "";
            grade["grade"] = sqlite3_column_text(stmt, 5) ? (char*)sqlite3_column_text(stmt, 5) : "";
            grade["grade_point"] = sqlite3_column_double(stmt, 6);
            grade["status"] = sqlite3_column_text(stmt, 7) ? (char*)sqlite3_column_text(stmt, 7) : "";
            grade["created_at"] = sqlite3_column_text(stmt, 8) ? (char*)sqlite3_column_text(stmt, 8) : "";
            
            grades.push_back(grade);
        }
    }
    
    sqlite3_finalize(stmt);
    return grades;
}

bool deleteGrade(const std::string& studentId, int gradeId) {
    sqlite3_stmt* stmt;
    const char* sql = "DELETE FROM grades WHERE student_id = ? AND id = ?";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return false;
    
    sqlite3_bind_text(stmt, 1, studentId.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, gradeId);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

double calculateGPA(const std::string& studentId) {
    sqlite3_stmt* stmt;
    const char* sql = R"(
        SELECT 
            SUM(grade_point * credit) as total_points,
            SUM(credit) as total_credits
        FROM grades 
        WHERE student_id = ? AND status = 'completed' AND grade_point > 0
    )";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    double gpa = 0.0;
    
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, studentId.c_str(), -1, SQLITE_STATIC);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            double total_points = sqlite3_column_double(stmt, 0);
            double total_credits = sqlite3_column_double(stmt, 1);
            if (total_credits > 0) {
                gpa = total_points / total_credits;
            }
        }
    }
    
    sqlite3_finalize(stmt);
    return gpa;
}

json getGradeAverages(const std::string& studentId) {
    sqlite3_stmt* stmt;
    const char* sql = R"(
        SELECT 
            category,
            AVG(grade_point) as avg_score
        FROM grades 
        WHERE student_id = ? AND status = 'completed' AND grade_point > 0
        GROUP BY category
    )";
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    json result;
    
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, studentId.c_str(), -1, SQLITE_STATIC);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string category = (char*)sqlite3_column_text(stmt, 0);
            double avg = sqlite3_column_double(stmt, 1);
            result[category] = avg;
        }
    }
    
    sqlite3_finalize(stmt);
    return result;
}

json performBasicPrediction(const std::string& studentId) {
    
    sqlite3_stmt* stmt;
    
    const char* completed_sql = R"(
        SELECT subject, subject_type, category, credit, grade_point
        FROM grades 
        WHERE student_id = ? AND status = 'completed' AND grade_point > 0
        ORDER BY created_at DESC
    )";
    
    std::vector<std::tuple<std::string, std::string, std::string, double, double>> completed;
    
    if (sqlite3_prepare_v2(db, completed_sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, studentId.c_str(), -1, SQLITE_STATIC);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string subject = (char*)sqlite3_column_text(stmt, 0);
            std::string subject_type = (char*)sqlite3_column_text(stmt, 1);
            std::string category = sqlite3_column_text(stmt, 2) ? (char*)sqlite3_column_text(stmt, 2) : "";
            double credit = sqlite3_column_double(stmt, 3);
            double grade_point = sqlite3_column_double(stmt, 4);
            
            completed.push_back({subject, subject_type, category, credit, grade_point});
        }
        sqlite3_finalize(stmt);
    }
    
    const char* current_sql = R"(
        SELECT subject, subject_type, category, credit
        FROM grades 
        WHERE student_id = ? AND status = 'current'
        ORDER BY created_at DESC
    )";
    
    std::vector<std::tuple<std::string, std::string, std::string, double>> current;
    
    if (sqlite3_prepare_v2(db, current_sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, studentId.c_str(), -1, SQLITE_STATIC);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string subject = (char*)sqlite3_column_text(stmt, 0);
            std::string subject_type = (char*)sqlite3_column_text(stmt, 1);
            std::string category = sqlite3_column_text(stmt, 2) ? (char*)sqlite3_column_text(stmt, 2) : "";
            double credit = sqlite3_column_double(stmt, 3);
            
            current.push_back({subject, subject_type, category, credit});
        }
        sqlite3_finalize(stmt);
    }
    
    if (completed.size() < 3 || current.empty()) {
        return json{
            {"success", false},
            {"error", "기본 예측을 위한 데이터 부족"},
            {"completed_count", completed.size()},
            {"current_count", current.size()},
            {"model_type", "validation_error"}
        };
    }
    
    double total_points = 0, total_credits = 0;
    std::map<std::string, std::vector<double>> type_grades;
    std::map<std::string, std::vector<double>> category_grades;
    
    for (const auto& [subject, type, category, credit, grade_point] : completed) {
        total_points += grade_point * credit;
        total_credits += credit;
        
        type_grades[type].push_back(grade_point);
        if (!category.empty()) {
            category_grades[category].push_back(grade_point);
        }
    }
    
    double past_gpa = total_points / total_credits;
    
    std::map<std::string, double> type_avg;
    std::map<std::string, double> category_avg;
    
    for (const auto& [type, grades] : type_grades) {
        double sum = 0;
        for (double g : grades) sum += g;
        type_avg[type] = sum / grades.size();
    }
    
    for (const auto& [cat, grades] : category_grades) {
        double sum = 0;
        for (double g : grades) sum += g;
        category_avg[cat] = sum / grades.size();
    }
    
    json predictions = json::array();
    double semester_points = 0, semester_credits = 0;
    
    for (const auto& [subject, type, category, credit] : current) {
        double predicted_gpa = past_gpa;
        
        if (type_avg.find(type) != type_avg.end()) {
            predicted_gpa = (predicted_gpa * 0.6) + (type_avg[type] * 0.4);
        }
        
        if (!category.empty() && category_avg.find(category) != category_avg.end()) {
            predicted_gpa = (predicted_gpa * 0.7) + (category_avg[category] * 0.3);
        }
        
        if (type == "전공") {
            predicted_gpa *= 0.97;
        } else if (type == "교양선택") {
            predicted_gpa *= 1.03;
        }
        
        predicted_gpa = std::max(0.0, std::min(4.5, predicted_gpa));
        
        predictions.push_back({
            {"subject", subject},
            {"subject_type", type},
            {"category", category.empty() ? "미분류" : category},
            {"credit", (int)credit},
            {"predicted_gpa", std::round(predicted_gpa * 100) / 100},
            {"predicted_grade", pointToGrade(predicted_gpa)},
            {"confidence", 82}
        });
        
        semester_points += predicted_gpa * credit;
        semester_credits += credit;
    }
    
    double semester_gpa = semester_points / semester_credits;
    double total_gpa = (total_points + semester_points) / (total_credits + semester_credits);
    
    json summary = {
        {"current_semester_gpa", std::round(semester_gpa * 100) / 100},
        {"current_semester_grade", pointToGrade(semester_gpa)},
        {"past_gpa", std::round(past_gpa * 100) / 100},
        {"predicted_total_gpa", std::round(total_gpa * 100) / 100},
        {"predicted_total_grade", pointToGrade(total_gpa)},
        {"model_used", "Advanced Statistical Analysis"},
        {"past_credits", (int)total_credits},
        {"current_credits", (int)semester_credits},
        {"total_credits", (int)(total_credits + semester_credits)},
        {"gpa_change", std::round((total_gpa - past_gpa) * 1000) / 1000},
        {"weak_subject", std::get<0>(current[0])},
        {"weak_category", "통계 분석 기반"},
        {"improvement_suggestions", json::array({
            "📊 통계적 분석 결과입니다",
            "🔍 더 정확한 예측을 위해 데이터를 추가하세요",
            "📈 현재 학습 패턴을 유지하시면 됩니다"
        })}
    };
    
    return json{
        {"success", true},
        {"predictions", predictions},
        {"summary", summary},
        {"model_type", "advanced_statistical"},
        {"augmentation_used", false},
        {"data_points", completed.size()},
        {"ai_confidence", "Medium"},
        {"feature_count", 8}
    };
}

std::string createPythonScript(const std::string& studentId) {
    std::cout << "[DEBUG] === Deep Learning Python Script Creation Started ===" << std::endl;
    std::cout << "[DEBUG] Student ID: " << studentId << std::endl;
    
    char cwd[1024];
#ifdef _WIN32
    if (_getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "[DEBUG] Current working directory: " << cwd << std::endl;
    }
#else
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "[DEBUG] Current working directory: " << cwd << std::endl;
    }
#endif
    
    std::string scriptName = "predict_grades_" + studentId + ".py";
    std::cout << "[DEBUG] Target filename: " << scriptName << std::endl;
    
    if (std::ifstream(scriptName).good()) {
        std::cout << "[DEBUG] Existing file found, attempting to delete..." << std::endl;
        if (std::remove(scriptName.c_str()) == 0) {
            std::cout << "[DEBUG] ✅ Existing file deleted successfully" << std::endl;
        } else {
            std::cout << "[DEBUG] ⚠️ Failed to delete existing file" << std::endl;
        }
    }
    
    std::cout << "[DEBUG] Attempting to open file stream..." << std::endl;
    std::ofstream pythonFile(scriptName, std::ios::out);
    
    if (!pythonFile.is_open()) {
        std::cout << "[ERROR] ❌ Failed to open file stream!" << std::endl;
        return "";
    }
    
    std::cout << "[DEBUG] ✅ File stream opened successfully" << std::endl;
    std::cout << "[DEBUG] Writing deep learning script content..." << std::endl;

    pythonFile << R"(#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    
    # Check SQLite3
    try:
        import sqlite3
        safe_print("✅ sqlite3: Available")
        dependencies['sqlite3'] = True
    except ImportError as e:
        safe_print(f"❌ sqlite3: {e}")
        dependencies['sqlite3'] = False
    
    # Check Pandas
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
        
        # Set TensorFlow log level
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
                
                # Average grade by category
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
                Dense(1, activation='linear')  # Linear activation for regression
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
            
            # Normalize (if scaler is available)
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
    if hasattr(value, 'item'):  # NumPy scalar
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
            "training_samples": int(len(predictor.completed_df) * 4)  # Including data augmentation
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
            # Very simple prediction (based on past average)
            predicted_gpa = past_gpa * 0.95  # Slightly conservative prediction
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
        
        # Get student ID from command line arguments
        if len(sys.argv) > 1:
            student_id = sys.argv[1]
        else:
            student_id = ")" + studentId + R"("
            
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
)";
    
    pythonFile.close();
    
    if (std::ifstream(scriptName).good()) {
        std::ifstream file(scriptName, std::ios::ate);
        auto size = file.tellg();
        file.close();
        
        return scriptName;
    } else {
        return "";
    }
}

std::string runPythonScript(const std::string& studentId) {
    std::cout << "[DEBUG] === Python Script Execution Started ===" << std::endl;
    std::string scriptName = createPythonScript(studentId);
    if (scriptName.empty()) {
        return R"({"success": false, "error": "Script creation failed"})";
    }
    
    std::ifstream testFile(scriptName);
    if (!testFile.good()) {
        return R"({"success": false, "error": "File access failed"})";
    }
    testFile.close();
    std::vector<std::string> pythonCommands = {
        "C:\\Users\\stardust\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
        "python",
        "python3",
        "py"
    };
    
    for (size_t i = 0; i < pythonCommands.size(); i++) {
        const auto& pythonCmd = pythonCommands[i];
        std::ostringstream cmdStream;
        cmdStream << "chcp 65001 > nul 2>&1 && "            
                  << "\"" << pythonCmd << "\" \"" << scriptName << "\" \"" << studentId << "\" 2>&1";
        std::string command = cmdStream.str();
            
        FILE* pipe = _popen(command.c_str(), "r");
        if (!pipe) {
            continue;
        }
        
        std::string result;
        std::string fullOutput;
        char buffer[2048];
        bool json_started = false;
        bool json_ended = false;
        int lineCount = 0;
        
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            fullOutput += line;
            lineCount++;
        
            try {
                std::cout << line;
            } catch (...) {
                std::cout << "[Korean output error]" << std::endl;
            }
            
            if (line.find("JSON_START") != std::string::npos) {
                json_started = true;
                continue;
            }
            
            if (line.find("JSON_END") != std::string::npos) {
                json_ended = true;
                break;
            }
            
            if (json_started && !json_ended) {
                result += line;
            }
        }
        
        int exitCode = _pclose(pipe);
        
        if (json_started && json_ended && !result.empty()) {

            try {
                auto test_json = json::parse(result);
                std::remove(scriptName.c_str());
                return result;
            } catch (const json::parse_error& e) {
                std::cout << "[DEBUG] ❌ JSON parsing failed: " << e.what() << std::endl;
            }
        } else {
            std::cout << "[DEBUG] ❌ JSON extraction failed" << std::endl;
        }
    }
    
    std::remove(scriptName.c_str());
    return R"({"success": false, "error": "Python execution failed"})";
}

int main() {
    try {
        std::cout.imbue(std::locale(""));
    } catch (...) {
        std::cout << "[WARNING] Locale setting failed" << std::endl;
    }
    
    system("chcp 65001 > nul");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    #ifdef _WIN32
    _putenv("PYTHONIOENCODING=utf-8");
    #else
    setenv("PYTHONIOENCODING", "utf-8", 1);
    #endif
    
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    
    httplib::Server svr;
    Py_SetPythonHome(L"C:/Users/stardust/AppData/Local/Programs/Python/Python313");

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::string html = R"(
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/crypto-js/4.1.1/crypto-js.min.js"></script>
<title>SSU:TUDY 로그인</title>
<link rel="website icon" type="png" href="http://localhost:8080/ssutudy_logo.png">
<style>
    body {
        font-family: Arial, sans-serif;
        text-align: center;
        margin: 0;
        padding: 0;
        background-color: #ffffff;
    }

    .logo {
        margin-top: 50px;
        display: flex;
        justify-content: center;
    }

    .logo img {
        height: 100px;
    }

    .slogan {
        font-size: 20px;
        margin-top: 20px;
        font-weight: 500;
    }

    .login-box {
        background-color: #ddd;
        width: 300px;
        margin: 30px auto;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }

    .login-box label {
        display: block;
        margin-top: 10px;
        text-align: left;
    }

    .login-box input[type="text"],
    .login-box input[type="password"] {
        width: 93%;
        padding: 8px;
        margin-top: 5px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }

    .login-box button {
        background-color: #006694;
        color: white;
        font-size: 18px;
        border: none;
        width: 100%;
        padding: 10px;
        border-radius: 4px;
        margin-top: 20px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }

    .login-box button:hover {
        background-color: #009bcb;
    }

    .footer {
        margin-top: 240px;
        font-size: 14px;
        color: #555;
    }
</style>
</head>
<body>
<div class="logo">
    <img src="ssutudy_logo.png" alt="SSU:TUDY 로고" />
    <div class="slogan">당신과 함께,<br />SSU:TUDY.</div>
</div>

<div class="login-box">
    <label for="student-id">학번 :</label>
    <input type="text" id="student-id" onkeydown="enterKey(event) " />

    <label for="password">비밀번호 :</label>
    <input type="password" id="password" onkeydown="enterKey(event) "/>

    <button onclick="login() ">
        🔒 로그인
    </button>
</div>

<div class="footer">
    Copyrights 2025. Lee SangHwa All rights reserved.
</div>

<script>
async function getKeyFromPassword(password, salt = 'salt') {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
        'raw',
        enc.encode(password),
        { name: 'PBKDF2' },
        false,
        ['deriveKey']
    );

    return crypto.subtle.deriveKey(
        {
            name: 'PBKDF2',
            salt: enc.encode(salt),
            iterations: 100000,
            hash: 'SHA-256',
        },
        keyMaterial,
        { name: 'AES-GCM', length: 256 },
        false,
        ['encrypt', 'decrypt']
    );
}

async function encryptAES(text, password) {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const key = await getKeyFromPassword(password);

    const enc = new TextEncoder();
    const encrypted = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv: iv },
        key,
        enc.encode(text)
    );

    const ivBase64 = btoa(String.fromCharCode(...iv));
    const encryptedBase64 = btoa(String.fromCharCode(...new Uint8Array(encrypted)));

    return `${ivBase64}:${encryptedBase64}`;
}

async function login() {
    const id = document.getElementById("student-id").value;
    const pw = document.getElementById("password").value;

    if (id === "" || pw === "") {
        alert("학번과 비밀번호를 모두 입력해주세요.");
        return;
    }

    try {
        const password = 'dUmgen8FDXnFn3J9LGRmcQ==:m7BI1/aK+fI+XgE4KRLuBbTM8LyVhzM9wumwHRMHe5f0sRSUwg82XQr+pkAdKzm8';
        const encryptedPw = await encryptAES(pw, password);

        const url = `/loading?id=${encodeURIComponent(id)}&pw=${encodeURIComponent(encryptedPw)}`;
        window.location.href = url;
    } catch (e) {
        console.error("암호화 오류:", e);
        alert("암호화 중 문제가 발생했습니다.");
    }
}

function enterKey(event) {
    if (event.key === "Enter") {
        login();
    }
}
</script>
</body>
</html>
        )";
        res.set_content(html, "text/html");
    });

    svr.Get("/ssutudy_logo.png", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream ifs("../image/ssutudy_logo.png", std::ios::binary);
        if (ifs) {
            std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            res.set_content(content, "image/png");
        } else {
            res.status = 404;
            res.set_content("Image not found", "text/plain");
        }
    });

    svr.Get("/loading", [](const httplib::Request& req, httplib::Response& res) {
        std::string full_target = req.target;
        std::string query_string = get_query_from_target(full_target);
        auto query_map = parse_query_string(query_string);
        id = query_map["id"];
        pw = query_map["pw"];
        
        std::string html = R"(
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>SSU:TUDY Loading...</title>
<link rel="website icon" type="png" href="http://localhost:8080/ssutudy_logo.png">
<style>
body {
    margin: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

.loading span {
    display: inline-block;
    width: 10px;
    height: 10px;
    background-color: gray;
    border-radius: 50%;
    animation: loading 1s 0s linear infinite;
}

.loading span:nth-child(1) {
    animation-delay: 0s;
    background-color: red;
}

.loading span:nth-child(2) {
    animation-delay: 0.2s;
    background-color: orange;
}

.loading span:nth-child(3) {
    animation-delay: 0.4s;
    background-color: yellowgreen;
}

@keyframes loading {
    0%, 100% {
        opacity: 0;
        transform: scale(0.5);
    }
    50% {
        opacity: 1;
        transform: scale(1.2);
    }
}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/crypto-js/4.1.1/crypto-js.min.js"></script>
<script>
async function getKeyFromPassword(password, salt = 'salt') {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
        'raw',
        enc.encode(password),
        { name: 'PBKDF2' },
        false,
        ['deriveKey']
    );

    return crypto.subtle.deriveKey(
        {
            name: 'PBKDF2',
            salt: enc.encode(salt),
            iterations: 100000,
            hash: 'SHA-256',
        },
        keyMaterial,
        { name: 'AES-GCM', length: 256 },
        false,
        ['encrypt', 'decrypt']
    );
}

async function decryptAES(encryptedStr, password) {
    try {
        const [ivBase64, encryptedBase64] = encryptedStr.split(':');
        const iv = Uint8Array.from(atob(ivBase64), c => c.charCodeAt(0));
        const encryptedBytes = Uint8Array.from(atob(encryptedBase64), c => c.charCodeAt(0));

        const key = await getKeyFromPassword(password);
        const decryptedBuffer = await crypto.subtle.decrypt(
            { name: 'AES-GCM', iv: iv },
            key,
            encryptedBytes
        );

        const dec = new TextDecoder();
        return dec.decode(decryptedBuffer);
    } catch (e) {
        console.error("복호화 실패:", e);
        return null;
    }
}

const password = 'dUmgen8FDXnFn3J9LGRmcQ==:m7BI1/aK+fI+XgE4KRLuBbTM8LyVhzM9wumwHRMHe5f0sRSUwg82XQr+pkAdKzm8';

async function login() {
    const id = ")" + id + R"(";
    const pw = await decryptAES(decodeURIComponent(")" + pw + R"("), password);
    
    if (id === "" || pw === "") {
        alert("학번과 비밀번호를 모두 입력해주세요.");
        return;
    }
    
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({id, pw})
    }).then(response => {
        if (response.ok) {
            window.location.href = "/main";
        } else {
            alert("로그인 실패: 학번 또는 비밀번호를 확인하세요.");
        }
    }).catch(error => {
        console.error("로그인 중 오류 발생:", error);
        alert("서버 오류가 발생했습니다.");
    });
}
login();
</script>
</head>
<body>
<div class="loading">
    <span></span>
    <span></span>
    <span></span>
</div>
<br>
<div class="footer">
    Loading...
</div>
</body>
</html>
        )";
        res.set_content(html, "text/html");
    });

    svr.Post("/login", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            id = j["id"];
            pw = j["pw"];
            std::cout << "[로그인 요청] ID: " << id << ", PW: " << unvisible(pw) << std::endl;
            std::cout << "[Response::200]:OK" << std::endl;
            
            std::string Py_code = R"(
import os
import time
import sys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import json
import requests
from bs4 import BeautifulSoup
import io
sys.stdout.reconfigure(encoding='utf-8')
sys.stdout = io.StringIO()
URL = "https://smartid.ssu.ac.kr/Symtra_sso/smln.asp?apiReturnUrl=https%3A%2F%2Fsaint.ssu.ac.kr%2FwebSSO%2Fsso.jsp"
options = uc.ChromeOptions()
options.add_argument("--disable-gpu") 
options.add_argument("--headless")
options.add_argument('--window-size=600,592')
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
driver = uc.Chrome(options=options)

driver.get(URL)
id = ")" + id + R"("
pw = ")" + pw + R"("
user_id = driver.find_element(By.ID,'userid')
user_id.send_keys(id)
user_pw = driver.find_element(By.ID, 'pwd')
user_pw.send_keys(pw)

btn = driver.find_element(By.XPATH, "//*[@id=\"sLogin\"]/div/div[1]/form/div/div[2]/a")
btn.click()
driver.implicitly_wait(5)
time.sleep(1)
s = requests.Session()
login_url = "https://smartid.ssu.ac.kr/Symtra_sso/smln_pcs.asp"
login_data = {
    "userid": id,
    "pwd": pw,
    "in_tp_bit": "0",
    "rqst_caus_cd": "03"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Referer": "https://saint.ssu.ac.kr/symtra_Sso/smln_login.asp",
    "Content-Type": "application/x-www-form-urlencoded",
}
s.headers.update(headers)

response_post = s.post(login_url, data=login_data, headers=headers)
response_post.raise_for_status()

for cookie in driver.get_cookies():
    c = {cookie['name'] : cookie['value']}
    s.cookies.update(c)

response = s.get("https://saint.ssu.ac.kr/webSSUMain/main_student.jsp")
soup = BeautifulSoup(response.text, 'html.parser')
name = str(soup.find_all("p", class_="main_title")[0].text).split("님")[0]
main_info = soup.find("div").find_all_next("strong")
print(name, end="/")
for i in main_info:
    print(str((i.text)).replace("과정", "과정 "), end="/")
            )";
            
            system("chcp 65001 > nul");
            
            Py_Initialize();
            if (PyRun_SimpleString(Py_code.c_str()) != 0) {
                std::cerr << "Python Error!\n";
            }
            
            PyObject* sysModule = PyImport_ImportModule("sys");
            PyObject* stdoutObj = PyObject_GetAttrString(sysModule, "stdout");

            if (sysModule && stdoutObj) {
                PyObject* output = PyObject_CallMethod(stdoutObj, "getvalue", nullptr);
                if (output && PyUnicode_Check(output)) {
                    std::string result = PyUnicode_AsUTF8(output);
                    std::istringstream ss(result);
                    std::string strBuffer;
                    x.clear();
                    while (getline(ss, strBuffer, '/')) {
                        strBuffer.erase(std::remove_if(strBuffer.begin(), strBuffer.end(),
                            [](unsigned char c) { return std::isspace(c); }),
                            strBuffer.end());
                        x.push_back(trim(strBuffer));
                    }

                    Py_DECREF(output);
                }
                Py_DECREF(stdoutObj);
                Py_DECREF(sysModule);
            }
            Py_Finalize();

            res.set_content("로그인 시도됨", "text/plain");
        } catch (const std::exception& e) {
            std::cerr << "JSON 파싱 에러: " << e.what() << std::endl;
            res.status = 400;
            res.set_content("Bad Request", "text/plain");
        }
    });

svr.Get("/main", [](const httplib::Request& req, httplib::Response& res) {
    bool gradesExist = hasGrades(id);
    if (!gradesExist) {
        std::string html = R"(<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SSU:TUDY</title>
    <link rel="website icon" type="png" href="http://localhost:8080/ssutudy_logo.png">
    <style>
        body {
            font-family: 'Malgun Gothic', Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .header {
            display: flex;
            align-items: center;
            padding: 10px 20px;
            border-bottom: 1px solid #e0e0e0;
        }
        .logo {
            display: flex;
            align-items: center;
        }
        .logo img {
            width: 70px;
            height: auto;
        }
        .logo-text {
            font-size: 24px;
            font-weight: bold;
            margin-left: 10px;
            color: #000;
        }
        .nav {
            background-color: #009bcb;
            display: flex;
            padding: 15px 20px;
        }
        .nav-item {
            margin-right: 30px;
            color: #c0f0ff;
            font-size: 18px;
            cursor: pointer;
        }
        .nav-item-main {
            margin-right: 30px;
            color: white;
            font-size: 18px;
            cursor: pointer;
        }
        .container {
            display: flex;
            padding: 20px;
            justify-content: center;
            align-items: center;
            min-height: 60vh;
        }
        .welcome-section {
            text-align: center;
            max-width: 600px;
        }
        .welcome-title {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }
        .welcome-message {
            font-size: 18px;
            margin-bottom: 40px;
            color: #666;
            line-height: 1.6;
        }
        .grade-input-btn {
            background-color: #009bcb;
            color: white;
            font-size: 20px;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 155, 203, 0.3);
        }
        .grade-input-btn:hover {
            background-color: #007ba3;
        }
        .profile-info {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: left;
        }
        .profile-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .profile-label {
            font-weight: bold;
            color: #333;
        }
        .profile-value {
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <img src="ssutudy_logo.png" alt="SSU:TUDY 로고" />
            <div class="logo-text">/ SSU:TUDY</div>
        </div>
    </div>

    <div class="nav">
            <div class="nav-item-main">메인</div>  
    <div class="nav-item" onclick="location.href='/grade-analysis'">성적 분석</div>
    <div class="nav-item" onclick="location.href='/grade-management'">성적 관리</div>
    <div class="nav-item">게시판</div>
    <div class="nav-item">설정</div>
    </div>

    <div class="container">
        <div class="welcome-section">
            <h1 class="welcome-title">환영해요! )" + x[0] + R"(님</h1>
            
            <div class="profile-info">
                <div class="profile-row">
                    <span class="profile-label">학번</span>
                    <span class="profile-value">)" + x[1] + R"(</span>
                </div>
                <div class="profile-row">
                    <span class="profile-label">소속</span>
                    <span class="profile-value">)" + x[2] + R"(</span>
                </div>
                <div class="profile-row">
                    <span class="profile-label">재학 여부</span>
                    <span class="profile-value">)" + x[3] + R"(</span>
                </div>
                <div class="profile-row">
                    <span class="profile-label">학년/학기</span>
                    <span class="profile-value">)" + x[4] + R"(</span>
                </div>
            </div>
            
            <div class="welcome-message">
                아직 등록된 성적이 없습니다.<br>
                성적을 입력하시면 개인 맞춤형 학습 분석을 제공해드립니다.
            </div>
            
            <button class="grade-input-btn" onclick="location.href='/grade-management'">
                📝 과목 추가하기
            </button>
        </div>
    </div>
</body>
</html>)";
        res.set_content(html, "text/html");
    } else {
        json averages = getGradeAverages(id);
        double totalGPA = calculateGPA(id);
        json currentSubjects = getCurrentSubjects(id);
        
        double teamworkAvg = getAverage(averages, "팀플");
        double projectAvg = getAverage(averages, "플젝");
        double attendanceAvg = getAverage(averages, "출석");
        double examAvg = getAverage(averages, "시험");
        
        double totalAvg = totalGPA;
        
        std::string currentSubjectsHtml = "";
        if (!currentSubjects.empty()) {
            std::ostringstream htmlStream;
            htmlStream << R"(
            <div class="current-subjects">
                <h3 style="color: #009bcb; margin-bottom: 15px;">📚 현재 수강 과목</h3>
                <div class="subjects-list">)";
            
            for (size_t i = 0; i < currentSubjects.size(); i++) {
                const auto& subject = currentSubjects[i];
                
                std::string name = "알 수 없음";
                std::string type = "미분류";
                std::string category = "";
                std::string credit = "0";
                
                try {
                    if (subject.contains("name") && subject["name"].is_string()) {
                        name = subject["name"].get<std::string>();
                    }
                    if (subject.contains("type") && subject["type"].is_string()) {
                        type = subject["type"].get<std::string>();
                    }
                    if (subject.contains("category") && subject["category"].is_string()) {
                        category = subject["category"].get<std::string>();
                    }
                    if (subject.contains("credit") && subject["credit"].is_number()) {
                        credit = std::to_string((int)subject["credit"].get<double>());
                    }
                } catch (const std::exception& e) {
                    std::cout << "[ERROR] JSON 처리 오류: " << e.what() << std::endl;
                }
                
                htmlStream << R"(
                    <div class="subject-item">
                        <div class="subject-main">
                            <span class="subject-name">)" << name << R"(</span>
                            <span class="subject-type">)" << type << R"(</span>
                        </div>
                        <div class="subject-details">
                            <span class="subject-credit">)" << credit << R"(학점</span>)";
                
                if (!category.empty()) {
                    htmlStream << R"(<span class="subject-category">)" << category << R"(</span>)";
                }
                
                htmlStream << R"(
                        </div>
                    </div>)";
            }
            
            htmlStream << R"(
                </div>
            </div>)";
            
            currentSubjectsHtml = htmlStream.str();
        }
        
        std::string weakestCategory = "팀 프로젝트";
        double minAvg = teamworkAvg;
        
        if (projectAvg < minAvg && projectAvg > 0) {
            minAvg = projectAvg;
            weakestCategory = "프로젝트";
        }
        if (attendanceAvg < minAvg && attendanceAvg > 0) {
            minAvg = attendanceAvg;
            weakestCategory = "출석";
        }
        if (examAvg < minAvg && examAvg > 0) {
            minAvg = examAvg;
            weakestCategory = "시험";
        }
        
        std::string html = R"(<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SSU:TUDY</title>
    <link rel="website icon" type="png" href="http://localhost:8080/ssutudy_logo.png">
    <style>
        body {
            font-family: 'Malgun Gothic', Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .header {
            display: flex;
            align-items: center;
            padding: 10px 20px;
            border-bottom: 1px solid #e0e0e0;
        }
        .logo {
            display: flex;
            align-items: center;
        }
        .logo img {
            width: 70px;
            height: auto;
        }
        .logo-text {
            font-size: 24px;
            font-weight: bold;
            margin-left: 10px;
            color: #000;
        }
        .nav {
            background-color: #009bcb;
            display: flex;
            padding: 15px 20px;
        }
        .nav-item {
            margin-right: 30px;
            color: #c0f0ff;
            font-size: 18px;
            cursor: pointer;
        }
        .nav-item-main {
            margin-right: 30px;
            color: white;
            font-size: 18px;
            cursor: pointer;
        }
        .container {
            display: flex;
            padding: 20px;
        }
        .profile-card {
            background-color: #f0f0f0;
            padding: 20px;
            width: 30%;
        }
        .profile-title {
            font-size: 20px;
            margin-bottom: 20px;
        }
        .profile-row {
            display: flex;
            border-bottom: 1px solid white;
            padding: 10px 0;
            background-color: white;
            margin-bottom: 10px;
        }
        .profile-label {
            width: 100px;
            font-weight: bold;
            padding-left: 10px;
        }
        .profile-value {
            flex-grow: 1;
            color: purple;
            text-align: right;
            padding-right: 10px;
        }
        .current-subjects {
            margin-top: 20px;
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #009bcb;
        }
        .subjects-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .subject-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 8px;
            border-bottom: 1px solid #f0f0f0;
            background-color: #f8f9fa;
            margin-bottom: 8px;
            border-radius: 8px;
            border-left: 4px solid #009bcb;
        }
        .subject-main {
            flex: 2;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .subject-name {
            font-weight: bold;
            color: #333;
            font-size: 15px;
        }
        .subject-type {
            color: #666;
            font-size: 12px;
            background-color: #e9ecef;
            padding: 2px 8px;
            border-radius: 12px;
            width: fit-content;
        }
        .subject-details {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 4px;
        }
        .subject-credit {
            color: #009bcb;
            font-size: 13px;
            font-weight: bold;
        }
        .subject-category {
            color: #28a745;
            font-size: 11px;
            background-color: #d4edda;
            padding: 2px 6px;
            border-radius: 10px;
            border: 1px solid #c3e6cb;
            
        }
        .score-section {
            margin-left: 20px;
            flex-grow: 1;
        }
        .score-display {
            font-size: 60px;
            font-weight: bold;
            color: #2E86DE;
            text-align: left;
            margin-bottom: 50px;
        }
        .score-max {
            font-size: 40px;
            color: #333;
        }
        .chart-container {
            display: flex;
            justify-content: space-around;
            align-items: flex-end;
            height: 250px;
            background-color: #f0f0f0;
            padding: 30px 20px;
            border-radius: 10px;
            position: relative;
        }
        .chart-bar {
            width: 18%;
            background-color: #2E86DE;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            border-radius: 5px 5px 0 0;
            position: relative;
            transition: all 0.3s ease;
        }
        .chart-bar:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(46, 134, 222, 0.3);
        }
        .chart-bar.yellow {
            background-color: #FFC107;
        }
        .chart-bar.red {
            background-color: #DC3545;
        }
        .chart-label {
            position: absolute;
            top: -30px;
            font-weight: bold;
            font-size: 14px;
            color: #333;
            background-color: white;
            padding: 4px 8px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chart-title {
            position: absolute;
            bottom: -35px;
            font-weight: bold;
            font-size: 16px;
            color: #333;
            text-align: center;
            width: 100%;
        }
        .message {
            text-align: center;
            margin-top: 30px;
            font-size: 18px;
            padding: 20px;
            background-color: #e7f3ff;
            border-radius: 10px;
            border-left: 5px solid #2E86DE;
        }
        .add-grade-btn {
            background-color: #009bcb;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
            font-size: 16px;
        }
        .add-grade-btn:hover {
            background-color: #007ba3;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <img src="ssutudy_logo.png" alt="SSU:TUDY 로고" />
            <div class="logo-text">/ SSU:TUDY</div>
        </div>
    </div>

    <div class="nav">
        <div class="nav-item-main">메인</div>
    <div class="nav-item" onclick="location.href='/grade-analysis'">성적 분석</div>
    <div class="nav-item" onclick="location.href='/grade-management'">성적 관리</div>
        <div class="nav-item">게시판</div>
        <div class="nav-item">설정</div>
    </div>

    <div class="container">
        <div class="profile-card">
            <div class="profile-title">환영해요! )" + x[0] + R"(님.</div>
            
            <div class="profile-row">
                <div class="profile-label">학번</div>
                <div class="profile-value">)" + x[1] + R"(</div>
            </div>
            
            <div class="profile-row">
                <div class="profile-label">소속</div>
                <div class="profile-value">)" + x[2] + R"(</div>
            </div>
            
            <div class="profile-row">
                <div class="profile-label">재학 여부</div>
                <div class="profile-value">)" + x[3] + R"(</div>
            </div>
            
            <div class="profile-row">
                <div class="profile-label">학년/학기</div>
                <div class="profile-value">)" + x[4] + R"(</div>
            </div>
            
            <button class="add-grade-btn" onclick="location.href='/grade-management'">
                📝 과목 추가하기
            </button>
            
            )" + currentSubjectsHtml + R"(
        </div>
        
        <div class="score-section">
            <div class="score-display">
                )" + std::to_string(totalAvg).substr(0, 4) + R"(<span class="score-max">/4.5</span>
            </div>
            
            <div class="chart-container">
                <div class="chart-bar)" + (teamworkAvg < 3.5 && teamworkAvg > 0 ? " yellow" : teamworkAvg < 2.5 && teamworkAvg > 0 ? " red" : "") + R"(" style="height: )" + std::to_string(teamworkAvg > 0 ? (teamworkAvg/4.5)*100 : 0) + R"(%;">
                    <span class="chart-label">)" + (teamworkAvg > 0 ? std::to_string(teamworkAvg).substr(0, 4) : "N/A") + R"(</span>
                    <span class="chart-title">팀플</span>
                </div>
                <div class="chart-bar)" + (projectAvg < 3.5 && projectAvg > 0 ? " yellow" : projectAvg < 2.5 && projectAvg > 0 ? " red" : "") + R"(" style="height: )" + std::to_string(projectAvg > 0 ? (projectAvg/4.5)*100 : 0) + R"(%;">
                    <span class="chart-label">)" + (projectAvg > 0 ? std::to_string(projectAvg).substr(0, 4) : "N/A") + R"(</span>
                    <span class="chart-title">플젝</span>
                </div>
                <div class="chart-bar)" + (attendanceAvg < 3.5 && attendanceAvg > 0 ? " yellow" : attendanceAvg < 2.5 && attendanceAvg > 0 ? " red" : "") + R"(" style="height: )" + std::to_string(attendanceAvg > 0 ? (attendanceAvg/4.5)*100 : 0) + R"(%;">
                    <span class="chart-label">)" + (attendanceAvg > 0 ? std::to_string(attendanceAvg).substr(0, 4) : "N/A") + R"(</span>
                    <span class="chart-title">출석</span>
                </div>
                <div class="chart-bar)" + (examAvg < 3.5 && examAvg > 0 ? " yellow" : examAvg < 2.5 && examAvg > 0 ? " red" : "") + R"(" style="height: )" + std::to_string(examAvg > 0 ? (examAvg/4.5)*100 : 0) + R"(%;">
                    <span class="chart-label">)" + (examAvg > 0 ? std::to_string(examAvg).substr(0, 4) : "N/A") + R"(</span>
                    <span class="chart-title">시험</span>
                </div>
            </div>
            
            <div class="message">
                💡 ")" + weakestCategory + R"(" 위주로 공부해볼까요?
            </div>
        </div>
    </div>
</body>
</html>)";
        res.set_content(html, "text/html");
    }
});

    initDatabase();
    // recreateDatabase(); 

svr.Get("/grade-management", [](const httplib::Request& req, httplib::Response& res) {

    json allGrades = getAllGrades(id);
    
    std::string html = R"(<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SSU:TUDY - 성적 관리</title>
    <link rel="website icon" type="png" href="http://localhost:8080/ssutudy_logo.png">
    <style>
        body {
            font-family: 'Malgun Gothic', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            display: flex;
            align-items: center;
            padding: 10px 20px;
            border-bottom: 1px solid #e0e0e0;
            background-color: white;
        }
        .logo {
            display: flex;
            align-items: center;
        }
        .logo img {
            width: 70px;
            height: auto;
        }
        .logo-text {
            font-size: 24px;
            font-weight: bold;
            margin-left: 10px;
            color: #000;
        }
        .nav {
            background-color: #009bcb;
            display: flex;
            padding: 15px 20px;
        }
        .nav-item {
            margin-right: 30px;
            color: #c0f0ff;
            font-size: 18px;
            cursor: pointer;
        }
        .nav-item-main {
            margin-right: 30px;
            color: white;
            font-size: 18px;
            cursor: pointer;
        }
        .container {
            max-width: 1200px;
            margin: 20px auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .page-title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 30px;
            text-align: center;
            color: #333;
        }
        .tab-container {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab {
            padding: 15px 30px;
            background-color: #f8f9fa;
            border: none;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px 10px 0 0;
            margin-right: 5px;
            transition: all 0.3s ease;
        }
        .tab.active {
            background-color: #009bcb;
            color: white;
        }
        .tab:hover:not(.active) {
            background-color: #e9ecef;
        }
        .form-section {
            display: none;
        }
        .form-section.active {
            display: block;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        .form-group input:focus, .form-group select:focus {
            border-color: #009bcb;
            outline: none;
        }
        .form-row {
            display: flex;
            gap: 20px;
        }
        .form-row .form-group {
            flex: 1;
        }
        .submit-btn {
            background-color: #009bcb;
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            transition: background-color 0.3s ease;
        }
        .submit-btn:hover {
            background-color: #007ba3;
        }
        .submit-btn.current {
            background-color: #28a745;
        }
        .submit-btn.current:hover {
            background-color: #218838;
        }
        .back-btn {
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            margin-bottom: 20px;
            text-decoration: none;
            display: inline-block;
        }
        .back-btn:hover {
            background-color: #5a6268;
        }
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #c3e6cb;
            display: none;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #f5c6cb;
            display: none;
        }
        .info-box {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #bee5eb;
        }
        .optional-label {
            color: #6c757d;
            font-size: 14px;
            font-weight: normal;
        }
        
        /* 성적 목록 스타일 */
        .grades-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .grades-table th,
        .grades-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .grades-table th {
            background-color: #009bcb;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }
        .grades-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .grades-table tr:hover {
            background-color: #e3f2fd;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-completed {
            background-color: #d4edda;
            color: #155724;
        }
        .status-current {
            background-color: #cce5ff;
            color: #004085;
        }
        .grade-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        .grade-A { background-color: #28a745; }
        .grade-B { background-color: #007bff; }
        .grade-C { background-color: #ffc107; color: #000; }
        .grade-D { background-color: #fd7e14; }
        .grade-F { background-color: #dc3545; }
        .delete-btn {
            background-color: #dc3545;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .delete-btn:hover {
            background-color: #c82333;
        }
        .no-data {
            text-align: center;
            padding: 40px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <img src="ssutudy_logo.png" alt="SSU:TUDY 로고" />
            <div class="logo-text">/ SSU:TUDY</div>
        </div>
    </div>

    <div class="nav">
        <div class="nav-item" onclick="location.href='/main'">메인</div>
        <div class="nav-item" onclick="location.href='/grade-analysis'">성적 분석</div>
        <div class="nav-item-main">성적 관리</div>
        <div class="nav-item">게시판</div>
        <div class="nav-item">설정</div>
    </div>

    <div class="container">
        <a href="/main" class="back-btn">← 메인으로 돌아가기</a>
        
        <h1 class="page-title">성적 관리</h1>
        
        <div id="success-message" class="success-message">
            작업이 성공적으로 완료되었습니다!
        </div>
        
        <div id="error-message" class="error-message">
            작업 중 오류가 발생했습니다. 다시 시도해주세요.
        </div>
        
        <!-- 탭 메뉴 -->
        <div class="tab-container">
            <button class="tab active" onclick="switchTab('list', event) ">성적 목록 관리</button>
            <button class="tab" onclick="switchTab('completed', event) ">완료된 과목 성적 입력</button>
            <button class="tab" onclick="switchTab('current', event) ">현재 수강 과목 추가</button>
        </div>
        
        <!-- 성적 목록 관리 섹션 -->
        <div id="list-section" class="form-section active">
            <div class="info-box">
                📋 등록된 모든 성적을 관리할 수 있습니다. 잘못 입력된 성적을 삭제하거나 수정할 수 있습니다.
            </div>
            
            <div id="grades-list-container">)";

    if (allGrades.empty()) {
        html += R"(
                <div class="no-data">
                    📝 등록된 성적이 없습니다.<br>
                    성적을 입력하려면 위의 탭을 이용하세요.
                </div>)";
    } else {
        html += R"(
                <table class="grades-table">
                    <thead>
                        <tr>
                            <th>과목명</th>
                            <th>구분</th>
                            <th>학점</th>
                            <th>카테고리</th>
                            <th>성적</th>
                            <th>상태</th>
                            <th>등록일</th>
                            <th>관리</th>
                        </tr>
                    </thead>
                    <tbody>)";
        
        for (const auto& grade : allGrades) {
            std::string subject = grade.value("subject", "");
            std::string subject_type = grade.value("subject_type", "");
            double credit = grade.value("credit", 0.0);
            std::string category = grade.value("category", "");
            std::string gradeStr = grade.value("grade", "");
            std::string status = grade.value("status", "");
            std::string created_at = grade.value("created_at", "");
            int gradeId = grade.value("id", 0);
            
            std::string displayDate = created_at.substr(0, 10);
            
            std::string statusClass = (status == "completed") ? "status-completed" : "status-current";
            std::string statusText = (status == "completed") ? "완료" : "수강중";
            
            std::string gradeBadgeClass = "";
            if (!gradeStr.empty()) {
                if (gradeStr[0] == 'A') gradeBadgeClass = "grade-A";
                else if (gradeStr[0] == 'B') gradeBadgeClass = "grade-B";
                else if (gradeStr[0] == 'C') gradeBadgeClass = "grade-C";
                else if (gradeStr[0] == 'D') gradeBadgeClass = "grade-D";
                else gradeBadgeClass = "grade-F";
            }
            
            html += R"(
                        <tr>
                            <td style="font-weight: bold;">)" + subject + R"(</td>
                            <td>)" + subject_type + R"(</td>
                            <td>)" + std::to_string((int)credit) + R"(</td>
                            <td>)" + (category.empty() ? "-" : category) + R"(</td>
                            <td>)";
            
            if (!gradeStr.empty()) {
                html += R"(<span class="grade-badge )" + gradeBadgeClass + R"(">)" + gradeStr + R"(</span>)";
            } else {
                html += "-";
            }
            
            html += R"(</td>
                            <td><span class="status-badge )" + statusClass + R"(">)" + statusText + R"(</span></td>
                            <td>)" + displayDate + R"(</td>
                            <td>
                                <button class="delete-btn" onclick="deleteGrade()" + std::to_string(gradeId) + R"() ">삭제</button>
                            </td>
                        </tr>)";
        }
        
        html += R"(
                    </tbody>
                </table>)";
    }
    
    html += R"(
            </div>
        </div>)";

    html += R"(
        
        <!-- 완료된 과목 성적 입력 폼 -->
        <div id="completed-section" class="form-section">
            <div class="info-box">
                💡 이미 수강 완료한 과목의 성적을 입력하세요. 성적은 GPA 계산에 반영됩니다.
            </div>
            
            <form id="completed-form">
                <div class="form-group">
                    <label for="completed-subject">과목명</label>
                    <input type="text" id="completed-subject" name="subject" required 
                           placeholder="예: 데이터구조, 프로그래밍기초, 미적분학">
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="completed-subject-type">과목 구분</label>
                        <select id="completed-subject-type" name="subject_type" required>
                            <option value="">구분을 선택하세요</option>
                            <option value="전공">전공</option>
                            <option value="교양필수">교양필수</option>
                            <option value="교양선택">교양선택</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="completed-credit">학점</label>
                        <select id="completed-credit" name="credit" required>
                            <option value="">학점을 선택하세요</option>
                            <option value="1.0">1.0</option>
                            <option value="2.0">2.0</option>
                            <option value="3.0">3.0</option>
                            <option value="4.0">4.0</option>
                            <option value="0.5">0.5</option>
                            <option value="1.5">1.5</option>
                            <option value="2.5">2.5</option>
                            <option value="3.5">3.5</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="completed-grade">성적</label>
                        <select id="completed-grade" name="grade" required>
                            <option value="">성적을 선택하세요</option>
                            <option value="A+">A+ (4.5)</option>
                            <option value="A0">A0 (4.3)</option>
                            <option value="A-">A- (4.0)</option>
                            <option value="B+">B+ (3.5)</option>
                            <option value="B0">B0 (3.3)</option>
                            <option value="B-">B- (3.0)</option>
                            <option value="C+">C+ (2.5)</option>
                            <option value="C0">C0 (2.3)</option>
                            <option value="C-">C- (2.0)</option>
                            <option value="D+">D+ (1.5)</option>
                            <option value="D0">D0 (1.3)</option>
                            <option value="D-">D- (1.0)</option>
                            <option value="F">F (0.0)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="completed-category">카테고리</label>
                        <select id="completed-category" name="category" required>
                            <option value="">카테고리를 선택하세요</option>
                            <option value="팀플">팀 프로젝트</option>
                            <option value="플젝">개인 프로젝트</option>
                            <option value="출석">출석</option>
                            <option value="시험">시험</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" class="submit-btn">성적 저장</button>
            </form>
        </div>
        
        <!-- 현재 수강 과목 추가 폼 -->
        <div id="current-section" class="form-section">
            <div class="info-box">
                📚 현재 수강중인 과목을 추가하세요. 카테고리를 설정하면 해당 분야의 학습 계획을 세우는 데 도움이 됩니다.
            </div>
            
            <form id="current-form">
                <div class="form-group">
                    <label for="current-subject">과목명</label>
                    <input type="text" id="current-subject" name="subject" required 
                           placeholder="예: 운영체제, 네트워크프로그래밍, 선형대수">
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="current-subject-type">과목 구분</label>
                        <select id="current-subject-type" name="subject_type" required>
                            <option value="">구분을 선택하세요</option>
                            <option value="전공">전공</option>
                            <option value="교양필수">교양필수</option>
                            <option value="교양선택">교양선택</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="current-credit">학점</label>
                        <select id="current-credit" name="credit" required>
                            <option value="">학점을 선택하세요</option>
                            <option value="1.0">1.0</option>
                            <option value="2.0">2.0</option>
                            <option value="3.0">3.0</option>
                            <option value="4.0">4.0</option>
                            <option value="0.5">0.5</option>
                            <option value="1.5">1.5</option>
                            <option value="2.5">2.5</option>
                            <option value="3.5">3.5</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="current-category">
                        카테고리 
                        <span class="optional-label">(선택사항 - 어떤 분야에 집중할지 미리 설정)</span>
                    </label>
                    <select id="current-category" name="category">
                        <option value="">카테고리를 선택하세요 (선택사항)</option>
                        <option value="팀플">팀 프로젝트</option>
                        <option value="플젝">개인 프로젝트</option>
                        <option value="출석">출석</option>
                        <option value="시험">시험</option>
                    </select>
                </div>
                
                <button type="submit" class="submit-btn current">현재 수강 과목 추가</button>
            </form>
        </div>
    </div>

    <script>
function switchTab(tabName, event) {
    // 모든 탭과 섹션 비활성화
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.form-section').forEach(section => section.classList.remove('active'));
    
    // 선택된 탭과 섹션 활성화
    if (event && event.target) {
        event.target.classList.add('active');
    } else {
        // event가 없는 경우 tabName으로 찾아서 활성화
        const tabs = document.querySelectorAll('.tab');
        tabs.forEach((tab, index) => {
            if ((tabName === 'list' && index === 0) ||
                (tabName === 'completed' && index === 1) ||
                (tabName === 'current' && index === 2)) {
                tab.classList.add('active');
            }
        });
    }
    document.getElementById(tabName + '-section').classList.add('active');
}
        
        // 성적 삭제 함수
        function deleteGrade(gradeId) {
            if (!confirm('정말로 이 성적을 삭제하시겠습니까?')) {
                return;
            }
            
            console.log('성적 삭제 요청:', gradeId);
            
            fetch('/delete-grade', {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ gradeId: gradeId })
            })
            .then(response => {
                if (response.ok) {
                    document.getElementById('success-message').textContent = '성적이 삭제되었습니다!';
                    document.getElementById('success-message').style.display = 'block';
                    document.getElementById('error-message').style.display = 'none';
                    
                    // 페이지 새로고침
                    setTimeout(() => {
                        location.reload();
                    }, 1000);
                } else {
                    document.getElementById('error-message').textContent = '삭제 중 오류가 발생했습니다.';
                    document.getElementById('error-message').style.display = 'block';
                    document.getElementById('success-message').style.display = 'none';
                }
            })
            .catch(error => {
                console.error('삭제 오류:', error);
                document.getElementById('error-message').textContent = '네트워크 오류가 발생했습니다.';
                document.getElementById('error-message').style.display = 'block';
                document.getElementById('success-message').style.display = 'none';
            });
        }
        
        // 완료된 과목 성적 입력 폼 처리 (기존과 동일)
        document.getElementById('completed-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = {
                subject: document.getElementById('completed-subject').value,
                subject_type: document.getElementById('completed-subject-type').value,
                credit: parseFloat(document.getElementById('completed-credit').value),
                grade: document.getElementById('completed-grade').value,
                category: document.getElementById('completed-category').value,
                status: 'completed'
            };
            
            console.log('[DEBUG] 완료된 과목 폼 데이터:', formData);
            submitForm(formData, this);
        });
        
        // 현재 수강 과목 추가 폼 처리 (기존과 동일)
        document.getElementById('current-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const categoryValue = document.getElementById('current-category').value;
            
            const formData = {
                subject: document.getElementById('current-subject').value,
                subject_type: document.getElementById('current-subject-type').value,
                credit: parseFloat(document.getElementById('current-credit').value),
                grade: null,
                category: categoryValue || null,
                status: 'current'
            };
            
            console.log('[DEBUG] 현재 수강 과목 폼 데이터:', formData);
            submitForm(formData, this);
        });
        
        function submitForm(formData, form) {
            console.log('[DEBUG] submitForm 호출됨:', formData);
            
            // 유효성 검사
            if (!formData.subject || !formData.subject_type || !formData.credit) {
                console.error('[ERROR] 필수 필드 누락');
                document.getElementById('error-message').textContent = '필수 필드를 모두 입력해주세요.';
                document.getElementById('error-message').style.display = 'block';
                document.getElementById('success-message').style.display = 'none';
                return;
            }
            
            if (formData.status === 'completed' && (!formData.grade || !formData.category)) {
                console.error('[ERROR] 완료된 과목 필수 필드 누락');
                document.getElementById('error-message').textContent = '완료된 과목은 성적과 카테고리를 모두 입력해야 합니다.';
                document.getElementById('error-message').style.display = 'block';
                document.getElementById('success-message').style.display = 'none';
                return;
            }
            
            console.log('[DEBUG] 유효성 검사 통과, 서버로 전송 중...');
            console.log('[DEBUG] 전송할 JSON:', JSON.stringify(formData));
            
            fetch('/save-grade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })
            .then(response => {
                console.log('[DEBUG] 서버 응답 상태:', response.status);
                if (response.ok) {
                    const message = formData.status === 'completed' ? 
                        '성적이 성공적으로 저장되었습니다!' : 
                        '현재 수강 과목이 추가되었습니다!';
                    
                    document.getElementById('success-message').textContent = message;
                    document.getElementById('success-message').style.display = 'block';
                    document.getElementById('error-message').style.display = 'none';
                    form.reset();
                    
                    console.log('[SUCCESS] 저장 완료:', message);
                    
                    // 2초 후 성적 목록 탭으로 이동
                    setTimeout(() => {
                        switchTab('list');
                        location.reload();
                    }, 2000);
                } else {
                    console.error('[ERROR] 서버 응답 오류:', response.status);
                    response.text().then(text => {
                        console.error('[ERROR] 서버 에러 메시지:', text);
                    });
                    document.getElementById('error-message').textContent = '저장 중 서버 오류가 발생했습니다.';
                    document.getElementById('error-message').style.display = 'block';
                    document.getElementById('success-message').style.display = 'none';
                }
            })
            .catch(error => {
                console.error('[ERROR] 네트워크 오류:', error);
                document.getElementById('error-message').textContent = '네트워크 오류가 발생했습니다.';
                document.getElementById('error-message').style.display = 'block';
                document.getElementById('success-message').style.display = 'none';
            });
        }
    </script>
</body>
</html>)";
    res.set_content(html, "text/html");
});

svr.Post("/save-grade", [](const httplib::Request& req, httplib::Response& res) {
    try {
        std::cout.flush();
        if (req.body.empty()) {
            res.status = 400;
            res.set_content("요청 데이터 없음", "text/plain");
            return;
        }
        
        auto j = json::parse(req.body);
        std::cout.flush();
        
        std::string subject = j.value("subject", "");
        std::string subject_type = j.value("subject_type", "");
        std::string status = j.value("status", "completed");
        double credit = j.value("credit", 0.0);
        
        std::string grade = "";
        std::string category = "";
        
        if (j.contains("grade") && !j["grade"].is_null()) {
            grade = j["grade"].get<std::string>();
        }
        
        if (j.contains("category") && !j["category"].is_null()) {
            category = j["category"].get<std::string>();
        }
        
        if (subject.empty() || subject_type.empty() || credit <= 0) {
            res.status = 400;
            res.set_content("필수 데이터 누락", "text/plain");
            return;
        }
        
        if (status == "completed" && (grade.empty() || category.empty())) {
            std::cout.flush();
            res.status = 400;
            res.set_content("완료된 과목은 성적과 카테고리가 필요합니다", "text/plain");
            return;
        }
        
        double grade_point = 0.0;
        if (!grade.empty()) {
            grade_point = gradeToPoint(grade);

        } else {
            //std::cout << "[DEBUG] 성적 없음, grade_point는 0.0" << std::endl;
        }

        if (id.empty()) {
            res.status = 400;
            res.set_content("로그인 필요", "text/plain");
            return;
        }
        
        if (!db) {
            res.status = 500;
            res.set_content("데이터베이스 오류", "text/plain");
            return;
        }
        
       
        sqlite3_stmt* stmt;
        const char* sql = "INSERT INTO grades (student_id, subject, category, subject_type, grade, credit, grade_point, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
        
        int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
        if (rc != SQLITE_OK) {
            res.status = 500;
            res.set_content("SQL 준비 오류", "text/plain");
            return;
        }


        rc = sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("데이터 바인딩 오류", "text/plain");
            return;
        }
        rc = sqlite3_bind_text(stmt, 2, subject.c_str(), -1, SQLITE_TRANSIENT);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("데이터 바인딩 오류", "text/plain");
            return;
        }

        if (category.empty()) {
            rc = sqlite3_bind_null(stmt, 3);
        } else {
            rc = sqlite3_bind_text(stmt, 3, category.c_str(), -1, SQLITE_TRANSIENT);
        }
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("데이터 바인딩 오류", "text/plain");
            return;
        }
        
        rc = sqlite3_bind_text(stmt, 4, subject_type.c_str(), -1, SQLITE_TRANSIENT);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("데이터 바인딩 오류", "text/plain");
            return;
        }
        
        if (grade.empty()) {
            rc = sqlite3_bind_null(stmt, 5);
        } else {
            rc = sqlite3_bind_text(stmt, 5, grade.c_str(), -1, SQLITE_TRANSIENT);
        }
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("데이터 바인딩 오류", "text/plain");
            return;
        }
        
        rc = sqlite3_bind_double(stmt, 6, credit);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("데이터 바인딩 오류", "text/plain");
            return;
        }
        
        rc = sqlite3_bind_double(stmt, 7, grade_point);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("데이터 바인딩 오류", "text/plain");
            return;
        }
        
        rc = sqlite3_bind_text(stmt, 8, status.c_str(), -1, SQLITE_TRANSIENT);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("데이터 바인딩 오류", "text/plain");
            return;
        }

        rc = sqlite3_step(stmt);
        
        if (rc == SQLITE_DONE) {
            sqlite3_finalize(stmt);
 
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            
            res.status = 200;
            res.set_content("저장 완료", "text/plain");
        } else {
            std::cout.flush();
            sqlite3_finalize(stmt);
            res.status = 500;
            res.set_content("저장 실패", "text/plain");
        }
        
    } catch (const json::parse_error& e) {
        res.status = 400;
        res.set_content("JSON 파싱 오류", "text/plain");
    } catch (const std::exception& e) {
        std::cout.flush();
        res.status = 500;
        res.set_content("서버 오류", "text/plain");
    }
});

svr.Delete("/delete-grade", [](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    
    try {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content("요청 데이터 없음", "text/plain");
            return;
        }
        
        auto j = json::parse(req.body);
        int gradeId = j.value("gradeId", 0);
        
        if (gradeId <= 0) {
            res.status = 400;
            res.set_content("유효하지 않은 성적 ID", "text/plain");
            return;
        }
        
        if (id.empty()) {
            res.status = 401;
            res.set_content("로그인이 필요합니다", "text/plain");
            return;
        }
        
        std::cout << "[DEBUG] 성적 삭제 요청: 학번=" << id << ", ID=" << gradeId << std::endl;
        
        bool success = deleteGrade(id, gradeId);
        
        if (success) {
            std::cout << "[SUCCESS] 성적 삭제 완료: ID=" << gradeId << std::endl;
            res.status = 200;
            res.set_content("삭제 완료", "text/plain");
        } else {
            std::cout << "[ERROR] 성적 삭제 실패: ID=" << gradeId << std::endl;
            res.status = 500;
            res.set_content("삭제 실패", "text/plain");
        }
        
    } catch (const json::parse_error& e) {
        std::cout << "[ERROR] JSON 파싱 오류: " << e.what() << std::endl;
        res.status = 400;
        res.set_content("JSON 파싱 오류", "text/plain");
    } catch (const std::exception& e) {
        std::cout << "[ERROR] 성적 삭제 중 오류: " << e.what() << std::endl;
        res.status = 500;
        res.set_content("서버 오류", "text/plain");
    }
});

svr.Options("/delete-grade", [](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.status = 200;
});

svr.Get("/grade-analysis", [](const httplib::Request& req, httplib::Response& res) {
    if (id.empty()) {
        res.status = 401;
        res.set_content("로그인이 필요합니다", "text/plain");
        return;
    }
    
    sqlite3_stmt* stmt;
    int completed_count = 0, current_count = 0;
    
    const char* completed_sql = "SELECT COUNT(*) FROM grades WHERE student_id = ? AND status = 'completed' AND grade_point > 0";
    if (sqlite3_prepare_v2(db, completed_sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            completed_count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    const char* current_sql = "SELECT COUNT(*) FROM grades WHERE student_id = ? AND status = 'current'";
    if (sqlite3_prepare_v2(db, current_sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            current_count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    std::string html = R"(<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SSU:TUDY - 🧠 AI 딥러닝 성적 분석</title>
    <link rel="website icon" type="png" href="http://localhost:8080/ssutudy_logo.png">
    <style>
        body {
            font-family: 'Malgun Gothic', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .header {
            display: flex;
            align-items: center;
            padding: 10px 20px;
            background-color: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }
        .logo {
            display: flex;
            align-items: center;
        }
        .logo img {
            width: 70px;
            height: auto;
        }
        .logo-text {
            font-size: 24px;
            font-weight: bold;
            margin-left: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .nav {
            background: linear-gradient(135deg, #009bcb 0%, #0066cc 100%);
            display: flex;
            padding: 15px 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        .nav-item {
            margin-right: 30px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 18px;
            cursor: pointer;
            transition: color 0.3s ease;
        }
        .nav-item:hover {
            color: white;
        }
        .nav-item-main {
            margin-right: 30px;
            color: white;
            font-size: 18px;
            cursor: pointer;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        }
        .container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 20px;
        }
        .page-title {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 30px;
            text-align: center;
            color: white;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
            background-size: 400% 400%;
            animation: gradientShift 3s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .back-btn {
            background: linear-gradient(45deg, #6c757d, #495057);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            margin-bottom: 20px;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .back-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        .status-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 25px;
            border-left: 6px solid #00d4aa;
        }
        .analysis-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 25px;
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        .loading-spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .ai-loading {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: aiGradient 2s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: bold;
            font-size: 20px;
        }
        @keyframes aiGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .predict-btn {
            background: linear-gradient(45deg, #00d4aa, #01a085);
            color: white;
            border: none;
            padding: 18px 40px;
            border-radius: 30px;
            font-size: 20px;
            cursor: pointer;
            margin: 25px 0;
            transition: all 0.3s ease;
            box-shadow: 0 6px 25px rgba(0, 212, 170, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            width: 100%;
        }
        .predict-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 35px rgba(0, 212, 170, 0.6);
        }
        .predict-btn:disabled {
            background: linear-gradient(45deg, #6c757d, #495057);
            cursor: not-allowed;
            transform: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .predict-btn.insufficient-data {
            background: linear-gradient(45deg, #ff9800, #f57c00);
            box-shadow: 0 6px 25px rgba(255, 152, 0, 0.4);
        }
        .predict-btn.insufficient-data:hover {
            box-shadow: 0 10px 35px rgba(255, 152, 0, 0.6);
        }
        .error-message {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        }
        .warning-message {
            background: linear-gradient(45deg, #ffa726, #ff9800);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(255, 167, 38, 0.3);
        }
        .prediction-results {
            display: none;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 35px;
        }
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            transition: transform 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .summary-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }
        .summary-card:hover::before {
            left: 100%;
        }
        .summary-card:hover {
            transform: translateY(-5px);
        }
        .summary-card.gpa-card {
            background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        }
        .summary-card.change-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .summary-card.model-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        .summary-card.total-card {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #333;
        }
        .summary-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .summary-label {
            font-size: 16px;
            opacity: 0.9;
            font-weight: 500;
        }
        .predictions-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .predictions-table th,
        .predictions-table td {
            padding: 18px 15px;
            text-align: left;
            border-bottom: 1px solid #f0f0f0;
        }
        .predictions-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            font-size: 16px;
        }
        .predictions-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .predictions-table tr:hover {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            transition: background 0.3s ease;
        }
        .grade-badge {
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            display: inline-block;
        }
        .grade-A { background: linear-gradient(45deg, #28a745, #20c997); }
        .grade-B { background: linear-gradient(45deg, #007bff, #0056b3); }
        .grade-C { background: linear-gradient(45deg, #ffc107, #e0a800); color: #000; text-shadow: none; }
        .grade-D { background: linear-gradient(45deg, #fd7e14, #dc6502); }
        .grade-F { background: linear-gradient(45deg, #dc3545, #c82333); }
        .confidence-badge {
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
            display: inline-block;
        }
        .confidence-high { background: linear-gradient(45deg, #28a745, #20c997); color: white; }
        .confidence-medium { background: linear-gradient(45deg, #ffc107, #e0a800); color: #000; }
        .confidence-low { background: linear-gradient(45deg, #dc3545, #c82333); color: white; }
        .info-section {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            padding: 30px;
            border-radius: 20px;
            border-left: 6px solid #2196f3;
            margin-bottom: 25px;
            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.2);
        }
        .tech-features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .tech-feature {
            background: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid rgba(33, 150, 243, 0.3);
            transition: all 0.3s ease;
        }
        .tech-feature:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.3);
        }
        .tech-feature h5 {
            margin: 0 0 10px 0;
            color: #1976d2;
            font-size: 16px;
        }
        .tech-feature p {
            margin: 0;
            font-size: 14px;
            color: #555;
        }
        .improvement-section {
            background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%);
            padding: 25px;
            border-radius: 20px;
            margin-top: 25px;
            border-left: 6px solid #ff9800;
            box-shadow: 0 6px 20px rgba(255, 152, 0, 0.2);
        }
        .improvement-list {
            list-style: none;
            padding: 0;
        }
        .improvement-list li {
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 152, 0, 0.2);
            font-size: 16px;
        }
        .improvement-list li:before {
            content: "🤖 ";
            margin-right: 10px;
        }
        .model-info {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            border-left: 6px solid #9c27b0;
            font-size: 16px;
            box-shadow: 0 4px 15px rgba(156, 39, 176, 0.2);
        }
        .data-requirements {
            background: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            text-align: left;
        }
        .requirement-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
            font-size: 16px;
        }
        .requirement-icon {
            margin-right: 10px;
            font-size: 20px;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e0e0e0;
            border-radius: 4px;
            margin: 15px 0;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, #4caf50, #8bc34a);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <img src="ssutudy_logo.png" alt="SSU:TUDY 로고" />
            <div class="logo-text">/ SSU:TUDY</div>
        </div>
    </div>

    <div class="nav">
        <div class="nav-item" onclick="location.href='/main'">메인</div>
        <div class="nav-item-main">성적 분석</div>
        <div class="nav-item" onclick="location.href='/grade-management'">성적 관리</div>
        <div class="nav-item">게시판</div>
        <div class="nav-item">설정</div>
    </div>

    <div class="container">
        <a href="/main" class="back-btn">← 메인으로 돌아가기</a>
        
        <h1 class="page-title">🧠 딥러닝 AI 성적 예측 시스템</h1>
        
        <div class="status-card">
            <h3>📊 학습 데이터 현황</h3>
            <div class="requirement-item">
                <span class="requirement-icon">📚</span>
                <span><strong>완료된 과목:</strong> )" + std::to_string(completed_count) + R"(개</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: )" + std::to_string(std::min(100, completed_count * 33)) + R"(%;"></div>
            </div>
            
            <div class="requirement-item">
                <span class="requirement-icon">🎯</span>
                <span><strong>현재 수강 과목:</strong> )" + std::to_string(current_count) + R"(개</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: )" + std::to_string(current_count > 0 ? 100 : 0) + R"(%;"></div>
            </div>
        </div>

        <div class="info-section">
            <h4>🤖 최첨단 딥러닝 AI 성적 예측</h4>
            <p>TensorFlow 딥러닝 모델과 고급 특성 공학을 통해 정확한 성적을 예측합니다.</p>
            <div class="tech-features">
                <div class="tech-feature">
                    <h5>🧬 데이터 증강</h5>
                    <p>가우시안 노이즈, 스마트 샘플링으로 4배 데이터 확장</p>
                </div>
                <div class="tech-feature">
                    <h5>🏗️ 딥러닝 아키텍처</h5>
                    <p>6층 신경망, BatchNorm, Dropout 정규화</p>
                </div>
                <div class="tech-feature">
                    <h5>📈 고급 특성 공학</h5>
                    <p>20+ 특성, 원핫인코딩, 상호작용 분석</p>
                </div>
                <div class="tech-feature">
                    <h5>⚡ 최적화</h5>
                    <p>Adam 옵티마이저, 조기종료, 앙상블</p>
                </div>
            </div>
        </div>
        
        <div class="analysis-section">)";
    
    if (completed_count < 3 || current_count == 0) {
        html += R"(
            <div class="warning-message">
                <h4>⚠️ 데이터 부족 감지</h4>
                <div class="data-requirements">
                    <h5>🎯 필요한 조건:</h5>)";
        
        if (completed_count < 3) {
            html += R"(
                    <div class="requirement-item">
                        <span class="requirement-icon">❌</span>
                        <span>완료된 과목: " + std::to_string(completed_count) + "/3개 (딥러닝 모델 훈련용)</span>
                    </div>)";
        } else {
            html += R"(
                    <div class="requirement-item">
                        <span class="requirement-icon">✅</span>
                        <span>완료된 과목: " + std::to_string(completed_count) + "/3개</span>
                    </div>)";
        }
        
        if (current_count == 0) {
            html += R"(
                    <div class="requirement-item">
                        <span class="requirement-icon">❌</span>
                        <span>현재 수강 과목: 0/1개 (예측 대상)</span>
                    </div>)";
        } else {
            html += R"(
                    <div class="requirement-item">
                        <span class="requirement-icon">✅</span>
                        <span>현재 수강 과목: " + std::to_string(current_count) + "개</span>
                    </div>)";
        }
        
        html += R"(
                </div>
                <p style="margin: 15px 0;">
                    데이터가 부족하지만 기본 예측을 시도할 수 있습니다.
                    <a href="/grade-management" style="color: white; text-decoration: underline; font-weight: bold;">
                        📝 과목 추가하기
                    </a>에서 더 많은 데이터를 입력하면 더 정확한 예측이 가능합니다.
                </p>
            </div>);
        
        // 데이터 부족해도 버튼 표시 (다른 스타일)
        html += R"(
            <button id="predictBtn" class="predict-btn insufficient-data" onclick="startDeepLearningPrediction() ">
                🔍 기본 AI 예측 시도
                <span style="font-size: 14px; opacity: 0.8;">데이터 부족시 통계 분석</span>
            </button>)";
    } else {
        html += R"(
            <button id="predictBtn" class="predict-btn" onclick="startDeepLearningPrediction() ">
                🚀 딥러닝 AI 성적 예측 시작
                <span style="font-size: 14px; opacity: 0.8;">TensorFlow/sklearn</span>
            </button>)";
    }
    
    html += R"(
            
            <div id="loading" class="loading" style="display: none;">
                <div class="loading-spinner"></div>
                <div class="ai-loading">🧠 딥러닝 AI가 고급 성적 분석을 수행하고 있습니다...</div>
                <div style="margin-top: 15px; font-size: 16px; color: #666;">
                    특성 공학 → 데이터 증강 → 모델 훈련 → 예측 수행 중...
                </div>
            </div>
            
            <div id="results" class="prediction-results">
                <div class="summary-grid">
                    <div class="summary-card gpa-card">
                        <div id="currentGPA" class="summary-value">-</div>
                        <div class="summary-label">이번 학기 예상 평점</div>
                    </div>
                    <div class="summary-card change-card">
                        <div id="gpaChange" class="summary-value">-</div>
                        <div class="summary-label">GPA 변동 예상</div>
                    </div>
                    <div class="summary-card model-card">
                        <div id="modelUsed" class="summary-value">AI</div>
                        <div class="summary-label">딥러닝 모델</div>
                    </div>
                    <div class="summary-card total-card">
                        <div id="totalGPA" class="summary-value">-</div>
                        <div class="summary-label">전체 누적 평점 (예상)</div>
                    </div>
                </div>
                
                <h3>🎯 과목별 딥러닝 AI 예측 결과</h3>
                <table id="predictionsTable" class="predictions-table">
                    <thead>
                        <tr>
                            <th>과목명</th>
                            <th>과목구분</th>
                            <th>카테고리</th>
                            <th>학점</th>
                            <th>AI 예상 성적</th>
                            <th>예상 등급</th>
                            <th>신뢰도</th>
                        </tr>
                    </thead>
                    <tbody>
                    </tbody>
                </table>
                
                <div class="model-info">
                    <strong>🔬 AI 모델 상세:</strong> 
                    <span id="modelDetails">딥러닝 신경망 (TensorFlow/sklearn), 고급 특성공학, 데이터증강</span>
                    <br><br>
                    <strong>📊 특성 개수:</strong> <span id="featureCount">-</span>개 | 
                    <strong>🎯 데이터포인트:</strong> <span id="dataPoints">-</span>개 |
                    <strong>🤖 AI 신뢰도:</strong> <span id="aiConfidence">-</span>
                </div>
                
                <div id="improvementSection" class="improvement-section" style="display: none;">
                    <h4>🤖 AI 맞춤 학습 전략 제안</h4>
                    <ul id="improvementList" class="improvement-list">
                    </ul>
                </div>
            </div>
            
            <div id="error" style="display: none;"></div>
        </div>
    </div>

    <script>
    
    // 딥러닝 예측 시작 함수 (별도 프로세스 방식 사용)
    async function startDeepLearningPrediction() {
    console.log('=== 딥러닝 예측 시작 (별도 프로세스 방식) ===');
    
    // UI 초기화
    const predictBtn = document.getElementById('predictBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    
    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.textContent = '🔄 예측 초기화 중...';
    }
    if (loading) loading.style.display = 'block';
    if (results) results.style.display = 'none';
    if (error) error.style.display = 'none';
    
    try {
        console.log('서버에 딥러닝 예측 요청 전송');
        
        const response = await fetch('/predict-grades-separate-process', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        
        console.log(`서버 응답: ${response.status}`);
        
        if (!response.ok) {
            throw new Error(`서버 오류: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('서버 응답 데이터:', data);
        
        // *** 중요: 서버 응답의 success와 result.success 모두 확인 ***
        if (data.success && data.result && data.result.success) {
            console.log('예측 성공, 결과 표시');
            displayResults(data.result);
        } else {
            // 실패 원인 확인
            let errorMsg = '예측 실패';
            if (data.result && data.result.error) {
                errorMsg = data.result.error;
            } else if (data.error) {
                errorMsg = data.error;
            }
            console.error('예측 실패:', errorMsg);
            throw new Error(errorMsg);
        }
        
    } catch (err) {
        console.error('예측 오류:', err.message);
        showError(`예측 실패: ${err.message}`);
    } finally {
        // UI 복원
        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtn.textContent = predictBtn.textContent.includes('기본') ? 
                '🔍 기본 AI 예측 시도' : '🚀 딥러닝 AI 성적 예측 시작';
        }
        if (loading) loading.style.display = 'none';
    }
}

    // 결과 표시 함수
    function displayResults(data) {
        console.log('=== 결과 표시 시작 ===');
        
        try {
            const results = document.getElementById('results');
            if (!results) {
                console.error('ERROR: results div 없음');
                return;
            }
            
            const summary = data.summary || {};
            const predictions = data.predictions || [];
            
            console.log(`요약 데이터: ${Object.keys(summary).length}개 항목`);
            console.log(`예측 데이터: ${predictions.length}개 과목`);
            
            // 기본 정보 설정
            const elements = {
                currentGPA: (summary.current_semester_gpa || 0) + ' (' + (summary.current_semester_grade || 'N/A') + ')',
                totalGPA: (summary.predicted_total_gpa || 0) + ' (' + (summary.predicted_total_grade || 'N/A') + ')',
                modelUsed: getModelDisplayName(data.model_type),
                featureCount: data.feature_count || 'N/A',
                dataPoints: data.data_points || 'N/A',
                aiConfidence: data.ai_confidence || 'N/A'
            };
            
            // 요소 업데이트
            Object.entries(elements).forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element) {
                    element.textContent = value;
                    console.log(`${id} 업데이트: ${value}`);
                } else {
                    console.log(`WARNING: ${id} 요소 없음`);
                }
            });
            
            // GPA 변경 표시
            const gpaChange = document.getElementById('gpaChange');
            if (gpaChange && summary.gpa_change !== undefined) {
                const change = summary.gpa_change;
                gpaChange.textContent = change > 0 ? `+${change}` : change.toString();
                gpaChange.style.color = change > 0 ? '#28a745' : change < 0 ? '#dc3545' : '#6c757d';
                console.log(`GPA 변화: ${change}`);
            }
            
            // 예측 테이블
            const tbody = document.querySelector('#predictionsTable tbody');
            if (tbody && predictions.length > 0) {
                tbody.innerHTML = '';
                console.log(`테이블에 ${predictions.length}개 행 추가`);
                
                predictions.forEach((pred, index) => {
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td style="font-weight: 600;">${pred.subject || '과목' + (index + 1)}</td>
                        <td><span style="font-size: 12px; background: #e9ecef; padding: 4px 8px; border-radius: 12px;">${pred.subject_type || '전공'}</span></td>
                        <td><span style="font-size: 12px; background: #d4edda; padding: 4px 8px; border-radius: 12px;">${pred.category || '기본'}</span></td>
                        <td style="font-weight: bold; color: #009bcb;">${pred.credit || 3}</td>
                        <td style="font-weight: bold; font-size: 18px;">${pred.predicted_gpa || 3.5}</td>
                        <td><span class="grade-badge ${getGradeBadgeClass(pred.predicted_grade || 'B+')}">${pred.predicted_grade || 'B+'}</span></td>
                        <td><span class="confidence-badge ${getConfidenceBadgeClass(pred.confidence || 80)}">${pred.confidence || 80}%</span></td>
                    `;
                });
            } else {
                console.log('WARNING: 예측 테이블 없거나 데이터 없음');
            }
            
            // 개선 제안
            if (summary.improvement_suggestions && summary.improvement_suggestions.length > 0) {
                const improvementSection = document.getElementById('improvementSection');
                const improvementList = document.getElementById('improvementList');
                
                if (improvementSection && improvementList) {
                    improvementList.innerHTML = '';
                    summary.improvement_suggestions.forEach(suggestion => {
                        const li = document.createElement('li');
                        li.textContent = suggestion;
                        improvementList.appendChild(li);
                    });
                    improvementSection.style.display = 'block';
                    console.log(`개선 제안 ${summary.improvement_suggestions.length}개 표시`);
                }
            }
            
            results.style.display = 'block';
            console.log('=== 결과 표시 완료 ===');
            
        } catch (err) {
            console.error(`결과 표시 오류: ${err.message}`);
            showError('결과 표시 중 오류가 발생했습니다.');
        }
    }

    // 에러 표시 함수
    function showError(message) {
        console.error(`에러 표시: ${message}`);
        
        const error = document.getElementById('error');
        if (!error) return;
        
        error.innerHTML = `
            <div class="error-message">
                <h4>❌ 처리 실패</h4>
                <p>${message}</p>
                <p style="font-size: 14px; margin-top: 15px;">
                    💡 <a href="/grade-management" style="color: white; text-decoration: underline;">
                    더 많은 과목을 추가</a>하면 정확도가 향상됩니다.
                </p>
                <button onclick="location.reload() " style="
                    background: white; color: #dc3545; border: 1px solid white;
                    padding: 8px 16px; border-radius: 4px; margin-top: 10px; cursor: pointer;
                ">
                    🔄 페이지 새로고침
                </button>
            </div>
        `;
        error.style.display = 'block';
    }

    // 유틸리티 함수들
    function getModelDisplayName(modelType) {
        if (!modelType) return 'AI';
        if (modelType.includes('deep_learning')) return 'DNN';
        if (modelType.includes('statistical')) return 'STAT';
        if (modelType.includes('basic')) return 'BASIC';
        return 'AI';
    }

    function getGradeBadgeClass(grade) {
        if (!grade) return 'grade-B';
        if (grade.startsWith('A')) return 'grade-A';
        if (grade.startsWith('B')) return 'grade-B';
        if (grade.startsWith('C')) return 'grade-C';
        if (grade.startsWith('D')) return 'grade-D';
        return 'grade-F';
    }

    function getConfidenceBadgeClass(confidence) {
        if (confidence >= 85) return 'confidence-high';
        if (confidence >= 70) return 'confidence-medium';
        return 'confidence-low';
    }

    </script>
</body>
</html>)";
    res.set_content(html, "text/html");
});


svr.Post("/predict-grades-separate-process", [](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.set_header("Content-Type", "application/json; charset=utf-8");
    
    if (id.empty()) {
        json errorResponse = {
            {"success", false},
            {"error", "로그인이 필요합니다"}
        };
        res.status = 401;
        res.set_content(errorResponse.dump(), "application/json");
        return;
    }
    
    try {
        std::cout << "[INFO] Deeplaerning Start: " << id << std::endl;
        std::string result = runPythonScript(id);
        
        if (result.empty()) {
            throw std::runtime_error("Python 스크립트 실행 결과 없음");
        }
        
        json result_json;
        try {
            result_json = json::parse(result);
        } catch (const json::parse_error& e) {
            throw std::runtime_error("JSON 파싱 실패");
        }
        

        if (!result_json.contains("success") || !result_json["success"].get<bool>()) {
            std::string error_msg = "딥러닝 예측 실패";
            if (result_json.contains("error")) {
                error_msg = result_json["error"].get<std::string>();
            }
            throw std::runtime_error(error_msg);
        }
        
        json response = {
            {"success", true},
            {"result", result_json},
            {"message", "딥러닝 예측 완료 (별도 프로세스)"},
            {"execution_method", "separate_process"}
        };
        
        res.set_content(response.dump(), "application/json");
        
    } catch (const std::exception& e) {
        
        try {
            json fallback_result = performBasicPrediction(id);
            
            if (!fallback_result["success"].get<bool>()) {
                std::cout << "[ERROR] 폴백 예측도 실패" << std::endl;
                throw std::runtime_error("모든 예측 방법 실패");
            }
            
            fallback_result["model_type"] = "fallback_separate_process";
            fallback_result["note"] = "딥러닝 실패로 통계 분석 사용: " + std::string(e.what());
            
            json response = {
                {"success", true},
                {"result", fallback_result},
                {"message", "기본 예측 완료 (폴백)"},
                {"execution_method", "fallback"}
            };
            
            res.set_content(response.dump(), "application/json");
            
        } catch (...) {
            json errorResponse = {
                {"success", false},
                {"error", "예측 실패: " + std::string(e.what())},
                {"execution_method", "failed"},
                {"debug_info", "Python 스크립트 및 폴백 모두 실패"}
            };
            res.status = 500;
            res.set_content(errorResponse.dump(), "application/json");
        }
    }
});

svr.Options("/predict-grades-separate-process", [](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.status = 200;
});

    svr.Options("/save-grade", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 200;
    });

    svr.listen("0.0.0.0", 8080);
    
    sqlite3_close(db);
    return 0;
}