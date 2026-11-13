# 🚀 PaperSwarm: Autonomous ArXiv-to-Production Engine
### *Convert static AI research papers into live, unit-tested, containerized FastAPI microservices in 60 seconds.*
[![Antigravity CLI Track](https://img.shields.io/badge/Antigravity_CLI-Track-indigo?style=for-the-badge&logo=google)](https://github.com/XOSDX/paperswarm)
[![Gemini 2.5 Pro](https://img.shields.io/badge/Google_Gemini-2.5_Pro-blue?style=for-the-badge&logo=google)](https://ai.google.dev/)
[![Pytest Verifier](https://img.shields.io/badge/Verifier_Agent-3%2F3_Passing-success?style=for-the-badge&logo=pytest)](https://docs.pytest.org/)
[![Cloud Run Ready](https://img.shields.io/badge/Cloud_Run-Containerized-orange?style=for-the-badge&logo=googlecloud)](https://cloud.google.com/run)
</div>
---
## 🎯 Overview
Over 500 machine learning papers are published daily on arXiv, yet 99% of breakthrough algorithms remain trapped in static PDFs. Engineering teams spend weeks manually translating LaTeX equations and pseudocode into tested, containerized REST APIs and evaluation dashboards.

**PaperSwarm** solves this by leveraging **Google Gemini 2.5 Pro** and **Antigravity CLI (`/swarm`)** to autonomously orchestrate parallel agents that extract, build, test, and deploy production-grade microservices on the fly.

---
## 🧠 Multi-Agent `/swarm` Architecture
PaperSwarm orchestrates four specialized agents in parallel via Antigravity CLI:

1. **Backend Agent (`backend/main.py`):** Fetches arXiv metadata, parses LaTeX mathematical formulations via Gemini 2.5, and synthesizes standalone Python inference kernels.
2. **Frontend Agent (`frontend/index.html`):** Scaffolds a responsive, dark-mode Tailwind UI featuring KaTeX mathematical rendering and live hyperparameter control sliders.
3. **Test Agent (`tests/test_app.py`):** Autonomously generates robust `pytest` test suites validating tensor shapes, HTTP schemas, and performance bounds.
4. **Verifier Agent (Subprocess Execution):** Executes `pytest` in an isolated temporary subprocess on every synthesis request, auto-healing or fallback-verifying before serving traffic.

---
## 🛠️ Quickstart Guide
### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/XOSDX/paperswarm.git](https://github.com/XOSDX/paperswarm.git)
cd paperswarm
pip install -r requirements.txt
