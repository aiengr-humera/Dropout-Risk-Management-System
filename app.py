
# Dropout Risk Management System — StudentForecastPortal by HNAIResearcher
# FastAPI backend: serves predictions, model versions, metrics, and WebSocket stream

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio, joblib, numpy as np, time, os, random
from datetime import datetime

app = FastAPI(title="Dropout Risk Management System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models and scaler on startup
FEATURES = [
    "attendance_rate","assignment_completion","quiz_avg_score","previous_gpa",
    "failed_subjects","late_submissions","lms_logins_per_week","library_visits",
    "office_hours_attended","study_group_participation","financial_stress",
    "commute_distance_km","part_time_job","family_dependents","internet_access_quality",
    "mental_health_score","peer_relationship_score","motivation_score",
    "first_generation_student","gender"
]

scaler  = joblib.load("models/scaler.pkl")
models  = {
    "v1_logistic_regression": joblib.load("models/v1_logistic_regression.pkl"),
    "v2_random_forest":       joblib.load("models/v2_random_forest.pkl"),
    "v3_gradient_boosting":   joblib.load("models/v3_gradient_boosting.pkl"),
}
model_metadata = {
    "v1_logistic_regression": {"accuracy":0.975,"f1_score":0.9801,"auc_roc":0.9977,"algorithm":"Logistic Regression"},
    "v2_random_forest":       {"accuracy":0.905,"f1_score":0.9224,"auc_roc":0.9761,"algorithm":"Random Forest"},
    "v3_gradient_boosting":   {"accuracy":0.920,"f1_score":0.9355,"auc_roc":0.9789,"algorithm":"Gradient Boosting"},
}

# System state
state = {
    "active_model": "v3_gradient_boosting",
    "sensitivity":  0.5,
}

# Pakistani student names
STUDENT_NAMES = [
    "Ahmed Raza","Fatima Malik","Usman Ali","Ayesha Khan","Bilal Ahmed",
    "Zara Hussain","Omar Farooq","Sana Iqbal","Hassan Nawaz","Nida Jameel",
    "Tariq Mehmood","Amna Sheikh","Saad Butt","Hira Baig","Faisal Chaudhry"
]

def generate_student():
    return {
        "name": random.choice(STUDENT_NAMES),
        "attendance_rate":          random.uniform(20, 100),
        "assignment_completion":    random.uniform(10, 100),
        "quiz_avg_score":           random.uniform(20, 100),
        "previous_gpa":             random.uniform(0.5, 4.0),
        "failed_subjects":          random.randint(0, 5),
        "late_submissions":         random.uniform(0, 100),
        "lms_logins_per_week":      random.uniform(0, 30),
        "library_visits":           random.randint(0, 20),
        "office_hours_attended":    random.randint(0, 15),
        "study_group_participation":random.randint(0, 10),
        "financial_stress":         random.randint(1, 5),
        "commute_distance_km":      random.uniform(0, 100),
        "part_time_job":            random.randint(0, 1),
        "family_dependents":        random.randint(0, 7),
        "internet_access_quality":  random.randint(1, 5),
        "mental_health_score":      random.uniform(1, 10),
        "peer_relationship_score":  random.uniform(1, 10),
        "motivation_score":         random.uniform(1, 10),
        "first_generation_student": random.randint(0, 1),
        "gender":                   random.randint(0, 1),
    }

def predict_risk(student_data):
    features = [student_data[f] for f in FEATURES]
    scaled   = scaler.transform([features])
    model    = models[state["active_model"]]
    start    = time.time()
    prob     = model.predict_proba(scaled)[0][1]
    latency  = round((time.time() - start) * 1000, 2)
    risk     = prob
    label    = "high" if risk > state["sensitivity"] else "low" if risk < 0.3 else "moderate"
    intervention = (
        "Schedule urgent academic counseling session" if label == "high"
        else "Send motivational check-in message" if label == "moderate"
        else "Student is on track — continue monitoring"
    )
    return {
        "risk_score":    round(risk, 4),
        "risk_label":    label,
        "intervention":  intervention,
        "latency_ms":    latency,
        "model_used":    state["active_model"],
        "timestamp":     datetime.now().isoformat(),
    }

# --- REST Endpoints ---

@app.get("/")
def root():
    return {"system": "Dropout Risk Management System", "status": "running"}

@app.get("/versions")
def get_versions():
    result = []
    for k, v in model_metadata.items():
        result.append({
            "id": k,
            "algorithm": v["algorithm"],
            "accuracy":  v["accuracy"],
            "f1_score":  v["f1_score"],
            "auc_roc":   v["auc_roc"],
            "active":    k == state["active_model"]
        })
    return {"versions": result}

@app.post("/versions/switch/{model_id}")
def switch_model(model_id: str):
    if model_id not in models:
        return {"error": "Model not found"}
    state["active_model"] = model_id
    return {"message": f"Switched to {model_id}", "active_model": model_id}

@app.post("/sensitivity/{value}")
def set_sensitivity(value: float):
    state["sensitivity"] = max(0.1, min(0.9, value))
    return {"sensitivity": state["sensitivity"]}

@app.get("/metrics")
def get_metrics():
    active = state["active_model"]
    meta   = model_metadata[active]
    return {
        "active_model": active,
        "accuracy":     meta["accuracy"],
        "f1_score":     meta["f1_score"],
        "auc_roc":      meta["auc_roc"],
        "sensitivity":  state["sensitivity"],
        "total_models": len(models),
    }

@app.post("/predict")
def predict(student: dict):
    return predict_risk(student)

# --- WebSocket Endpoint ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            student = generate_student()
            result  = predict_risk(student)
            result["student_name"] = student["name"]
            result["features"] = {
                "attendance":   round(student["attendance_rate"], 1),
                "assignments":  round(student["assignment_completion"], 1),
                "quiz_score":   round(student["quiz_avg_score"], 1),
                "gpa":          round(student["previous_gpa"], 2),
                "motivation":   round(student["motivation_score"], 1),
                "mental_health":round(student["mental_health_score"], 1),
            }
            await websocket.send_json(result)
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        pass
