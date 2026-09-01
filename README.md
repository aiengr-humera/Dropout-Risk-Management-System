# Dropout Risk Management System
## StudentForecastPortal by HNAIResearcher

### What it does
A real-time student dropout risk prediction and early intervention system built for Pakistani university context. The system continuously monitors student data across 20 features — academic, behavioral, socioeconomic, and psychological — and streams live risk predictions through a WebSocket-powered dashboard with three role-based views.

### Five reference patterns covered
1. **Live streaming + sensitivity slider** — WebSocket streams risk scores continuously, threshold slider adjusts sensitivity live via API
2. **Model versioning & rollback** — 3 trained model versions (Logistic Regression, Random Forest, Gradient Boosting) with live switching and A/B comparison
3. **Metrics endpoint** — /metrics reports accuracy, F1, AUC-ROC, and prediction latency per request
4. **Multi-stage pipeline** — 5-stage visual pipeline (Data input → Validation → Feature scaling → Risk scoring → Intervention) with per-stage animation
5. **Model leaderboard** — ranked comparison of all 3 versions by accuracy, F1, and AUC

### Three user views
- **Teacher view** — live class risk gauge, student feed, alerts with intervention suggestions
- **Admin view** — model version panel, sensitivity control, performance metrics
- **Student view** — personal risk score, trend arrow, what's affecting score, recommended action

### Tech stack
- **Model:** scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- **Backend:** FastAPI + WebSocket + Uvicorn
- **Tunnel:** ngrok (browser access from Colab)
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **Data:** Synthetic Pakistani university student dataset (1000 students, 20 features)

### How to run
1. Open `Dropout Risk Management System.ipynb` in Google Colab
2. Run all cells in order (1 through 6)
3. Copy the ngrok URL printed in Cell 6
4. Update `BASE` in `dashboard.html` with the new URL
5. Download and open `dashboard.html` in your browser
6. Visit the ngrok URL once to clear the browser warning, then refresh the dashboard

### Model performance
| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 97.5% | 0.9801 | 0.9977 |
| Gradient Boosting | 92.0% | 0.9355 | 0.9789 |
| Random Forest | 90.5% | 0.9224 | 0.9761 |

### By
Humera Noor Ahmad — AI Researcher & Educator  
HNAIResearcher | Faisalabad, Pakistan
