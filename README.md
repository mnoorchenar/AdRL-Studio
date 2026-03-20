---
title: AdRL Studio
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<h1>🎯 AdRL Studio</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=7C3AED&center=true&vCenter=true&width=700&lines=Contextual+Bandit+Ad+Recommendation+Engine;Benchmark+%CE%B5-Greedy%2C+UCB1%2C+Thompson%2C+LinUCB;Real-Time+Ad+Serving+%2B+Regret+Analysis" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🎯 AdRL Studio** — A contextual multi-armed bandit platform that simulates a real-world ad recommendation and serving system using reinforcement learning. Benchmarks four bandit algorithms side by side, visualizes online learning and regret curves, runs A/B test simulations with statistical significance testing, and serves real-time ad recommendations from user context input.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🎯 <b>Live Ad Serving</b></td>
    <td>Enter user context (age, device, time, category, region) and get real-time ad recommendations from all 4 algorithms simultaneously</td>
  </tr>
  <tr>
    <td>▶ <b>Online Learning Simulation</b></td>
    <td>Run 1K–10K impression simulations with SSE-streamed progress, rolling CTR charts, and per-algorithm summaries</td>
  </tr>
  <tr>
    <td>📉 <b>Regret Analysis</b></td>
    <td>Visualize cumulative regret curves — the canonical RL evaluation metric — comparing all four policies</td>
  </tr>
  <tr>
    <td>⚖ <b>A/B Test Simulator</b></td>
    <td>Run 50/50 traffic splits with two-proportion z-test, p-value, confidence intervals, and statistical significance verdict</td>
  </tr>
  <tr>
    <td>🔒 <b>Secure by Design</b></td>
    <td>Role-based access, audit logs, encrypted data pipelines</td>
  </tr>
  <tr>
    <td>🐳 <b>Containerized Deployment</b></td>
    <td>Docker-first architecture, cloud-ready and scalable</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      AdRL Studio                        │
│                                                         │
│  ┌───────────┐    ┌───────────┐    ┌───────────────┐  │
│  │  Simulated│───▶│  Bandit   │───▶│   Flask API   │  │
│  │ Ad Environ│    │ Algorithms│    │   Backend     │  │
│  └───────────┘    └───────────┘    └───────┬───────┘  │
│                                            │           │
│                                   ┌────────▼────────┐  │
│                                   │  Plotly Charts  │  │
│                                   │   Dashboard     │  │
│                                   └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/AdRL-Studio.git
cd AdRL-Studio

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your settings

# 5. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker compose up --build

# Or pull and run the pre-built image
docker pull mnoorchenar/AdRL-Studio
docker run -p 7860:7860 mnoorchenar/AdRL-Studio
```

---

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 🎯 Live Ad Serving | Real-time 4-algorithm recommendation from user context | ✅ Live |
| ▶ Online Learning | Simulation with SSE streaming and rolling CTR charts | ✅ Live |
| 📉 Regret Analysis | Cumulative regret curves for all four algorithms | ✅ Live |
| ⚖ A/B Test Simulator | Statistical significance testing with z-test & CI | ✅ Live |
| 🌡 Reward Landscape | 5×5 CTR heatmap: user content category × ad category | ✅ Live |
| 🔬 Policy Inspector | Per-ad learned weights and posterior distributions | 🗓️ Planned |

---

## 🧠 ML Models

```python
# Core Models Used in AdRL Studio
models = {
    "epsilon_greedy": "ε-Greedy Neural Bandit — shared PyTorch MLP (39→32→16→1) with decaying ε",
    "ucb1":           "UCB1 — Upper Confidence Bound non-contextual baseline",
    "thompson":       "Thompson Sampling — Bayesian Beta(α,β) per arm",
    "linucb":         "LinUCB Disjoint — ridge regression contextual bandit (production-grade)",
    "environment":    "Simulated 20-ad inventory, 19-dim one-hot context, Bernoulli reward sampling"
}
```

---

## 📁 Project Structure

```
AdRL-Studio/
│
├── 📄 app.py               # Complete Flask application — all logic, templates, and API
├── 📄 Dockerfile           # Container definition (python:3.10-slim, port 7860)
├── 📄 requirements.txt     # Python dependencies
└── 📄 README.md            # This file
```

> All application logic, HTML templates, CSS, and JavaScript live inside `app.py`
> using Flask's `render_template_string`. There are no external static files.

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available — no real user data is stored. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/AdRL-Studio?style=social)](https://github.com/mnoorchenar/AdRL-Studio)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/AdRL-Studio?style=social)](https://github.com/mnoorchenar/AdRL-Studio/fork)

<sub>The name "AdRL Studio" is used purely for academic and research purposes. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity.</sub>

</div>
