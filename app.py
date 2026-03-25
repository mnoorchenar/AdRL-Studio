"""
AdRL Studio — Contextual Bandit Ad Recommendation Engine

This application implements and benchmarks four reinforcement learning
contextual bandit algorithms for ad recommendation: (1) ε-Greedy Neural
Bandit using a shared PyTorch MLP, (2) UCB1 (Upper Confidence Bound),
a non-contextual baseline, (3) Thompson Sampling with Beta distribution
priors, and (4) LinUCB Disjoint Model, the industry-standard contextual
bandit used in production ad systems. The simulated environment features
20 ads across 5 categories and 5 user context features (age group, device,
time of day, content category, region) encoded as a 19-dimensional one-hot
vector. True click-through rates are determined by hidden weight vectors
initialized at startup (seed=42). Algorithms observe only bandit feedback
— the reward for the chosen arm only — and must balance exploration
vs. exploitation to minimize cumulative regret.
"""

import json
import math
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, Response, jsonify, render_template_string, request
from scipy import stats

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Environment constants
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

AGE_GROUPS    = ["young_adult", "adult", "senior"]
DEVICES       = ["mobile", "desktop", "tablet"]
TIMES_OF_DAY  = ["morning", "afternoon", "evening", "night"]
CONTENT_CATS  = ["tech", "sports", "lifestyle", "news", "entertainment"]
REGIONS       = ["north_america", "europe", "asia", "other"]
CONTEXT_DIM   = len(AGE_GROUPS) + len(DEVICES) + len(TIMES_OF_DAY) + len(CONTENT_CATS) + len(REGIONS)  # 19

N_ADS = 20
AD_IDS = [f"ad_{i:02d}" for i in range(1, 21)]
# Category mapping
AD_CAT_MAP = {}
for i, ad in enumerate(AD_IDS):
    cats = ["Tech","Fashion","Finance","Food","Travel"]
    AD_CAT_MAP[ad] = cats[i // 4]

AD_FORMATS = {
    "ad_01":"banner","ad_02":"video","ad_03":"native","ad_04":"banner",
    "ad_05":"banner","ad_06":"video","ad_07":"banner","ad_08":"native",
    "ad_09":"native","ad_10":"banner","ad_11":"video","ad_12":"native",
    "ad_13":"banner","ad_14":"native","ad_15":"banner","ad_16":"video",
    "ad_17":"video","ad_18":"banner","ad_19":"native","ad_20":"video",
}
AD_BIDS = {
    "ad_01":2.50,"ad_02":3.00,"ad_03":3.50,"ad_04":4.00,
    "ad_05":1.50,"ad_06":2.00,"ad_07":2.50,"ad_08":3.00,
    "ad_09":3.00,"ad_10":3.50,"ad_11":4.00,"ad_12":5.00,
    "ad_13":1.00,"ad_14":1.50,"ad_15":2.00,"ad_16":2.50,
    "ad_17":2.00,"ad_18":2.50,"ad_19":3.00,"ad_20":3.50,
}

# Hidden true CTR weights — fixed at startup, never exposed to algorithms
_TRUE_WEIGHTS = np.random.randn(N_ADS, CONTEXT_DIM) * 0.3

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def true_ctr(ad_idx, ctx):
    return float(np.clip(_sigmoid(ctx @ _TRUE_WEIGHTS[ad_idx]), 0.02, 0.25))

def encode_context(age, device, tod, content, region):
    vec = np.zeros(CONTEXT_DIM, dtype=np.float32)
    offset = 0
    vec[offset + AGE_GROUPS.index(age)] = 1.0;    offset += len(AGE_GROUPS)
    vec[offset + DEVICES.index(device)] = 1.0;    offset += len(DEVICES)
    vec[offset + TIMES_OF_DAY.index(tod)] = 1.0;  offset += len(TIMES_OF_DAY)
    vec[offset + CONTENT_CATS.index(content)] = 1.0; offset += len(CONTENT_CATS)
    vec[offset + REGIONS.index(region)] = 1.0
    return vec

def sample_random_context():
    return encode_context(
        np.random.choice(AGE_GROUPS), np.random.choice(DEVICES),
        np.random.choice(TIMES_OF_DAY), np.random.choice(CONTENT_CATS),
        np.random.choice(REGIONS),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm classes
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonGreedyNeuralBandit:
    NAME  = "ε-Greedy"
    COLOR = "#f59e0b"

    def __init__(self, epsilon=0.15, epsilon_min=0.01, decay=0.995, lr=0.01):
        self.epsilon_0   = epsilon
        self.epsilon_min = epsilon_min
        self.decay       = decay
        self.lr          = lr
        self.reset()

    def reset(self):
        self.t = 0
        self.n_updates = 0
        self.model = nn.Sequential(
            nn.Linear(CONTEXT_DIM + N_ADS, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),  nn.Sigmoid(),
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion  = nn.MSELoss()

    def _inp(self, ctx, ad_idx):
        oh = np.zeros(N_ADS, dtype=np.float32); oh[ad_idx] = 1.0
        return torch.FloatTensor(np.concatenate([ctx, oh]))

    def _pred(self, ctx, ad_idx):
        self.model.eval()
        with torch.no_grad():
            return self.model(self._inp(ctx, ad_idx)).item()

    def select(self, ctx):
        eps = max(self.epsilon_min, self.epsilon_0 * (self.decay ** self.t))
        if np.random.rand() < eps:
            return int(np.random.randint(N_ADS))
        ctx_rep = np.tile(ctx, (N_ADS, 1))
        ad_eye  = np.eye(N_ADS, dtype=np.float32)
        batch   = torch.FloatTensor(np.hstack([ctx_rep, ad_eye]))
        self.model.eval()
        with torch.no_grad():
            scores = self.model(batch).squeeze().numpy()
        return int(np.argmax(scores))

    def predict_ctr(self, ctx, ad_idx):
        return self._pred(ctx, ad_idx)

    def update(self, ctx, action, reward):
        self.model.train()
        x = self._inp(ctx, action).unsqueeze(0)
        y = torch.FloatTensor([[float(reward)]])
        self.optimizer.zero_grad()
        self.criterion(self.model(x), y).backward()
        self.optimizer.step()
        self.t += 1
        self.n_updates += 1


class UCB1Bandit:
    NAME  = "UCB1"
    COLOR = "#10b981"

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_a       = np.zeros(N_ADS)
        self.R_a       = np.zeros(N_ADS)
        self.t         = 0
        self._init_idx = 0
        self.n_updates = 0

    def select(self, ctx):
        if self._init_idx < N_ADS:
            return self._init_idx
        mu    = self.R_a / np.maximum(self.n_a, 1)
        bonus = np.sqrt(2.0 * np.log(max(self.t, 1)) / np.maximum(self.n_a, 1))
        return int(np.argmax(mu + bonus))

    def predict_ctr(self, ctx, ad_idx):
        if self.n_a[ad_idx] == 0:
            return 0.0
        return float(self.R_a[ad_idx] / self.n_a[ad_idx])

    def update(self, ctx, action, reward):
        if self._init_idx < N_ADS:
            self._init_idx += 1
        self.n_a[action] += 1
        self.R_a[action] += reward
        self.t += 1
        self.n_updates += 1


class ThompsonSamplingBandit:
    NAME  = "Thompson"
    COLOR = "#3b82f6"

    def __init__(self):
        self.reset()

    def reset(self):
        self.alpha   = np.ones(N_ADS)
        self.beta_p  = np.ones(N_ADS)
        self.n_updates = 0

    def select(self, ctx):
        return int(np.argmax(np.random.beta(self.alpha, self.beta_p)))

    def predict_ctr(self, ctx, ad_idx):
        return float(self.alpha[ad_idx] / (self.alpha[ad_idx] + self.beta_p[ad_idx]))

    def update(self, ctx, action, reward):
        if reward == 1:
            self.alpha[action]  += 1
        else:
            self.beta_p[action] += 1
        self.n_updates += 1


class LinUCBBandit:
    NAME  = "LinUCB"
    COLOR = "#ef4444"

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.reset()

    def reset(self):
        d = CONTEXT_DIM
        self.A     = [np.identity(d) for _ in range(N_ADS)]
        self.A_inv = [np.identity(d) for _ in range(N_ADS)]
        self.b     = [np.zeros(d)    for _ in range(N_ADS)]
        self.n_updates = 0

    def _ucb_score(self, ctx, ad_idx):
        A_inv = self.A_inv[ad_idx]
        theta = A_inv @ self.b[ad_idx]
        x     = ctx
        return float(theta @ x + self.alpha * math.sqrt(max(float(x @ A_inv @ x), 0.0)))

    def select(self, ctx):
        return int(np.argmax([self._ucb_score(ctx, a) for a in range(N_ADS)]))

    def predict_ctr(self, ctx, ad_idx):
        return float((self.A_inv[ad_idx] @ self.b[ad_idx]) @ ctx)

    def update(self, ctx, action, reward):
        x   = ctx
        Ai  = self.A_inv[action]
        Aix = Ai @ x
        self.A_inv[action] = Ai - np.outer(Aix, Aix) / (1.0 + x @ Aix)
        self.A[action]    += np.outer(x, x)
        self.b[action]    += reward * x
        self.n_updates    += 1


# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
ALGO_KEYS = ["epsilon_greedy", "ucb1", "thompson", "linucb"]
ALGO_CLASSES = {
    "epsilon_greedy": EpsilonGreedyNeuralBandit,
    "ucb1":           UCB1Bandit,
    "thompson":       ThompsonSamplingBandit,
    "linucb":         LinUCBBandit,
}
ALGO_DISPLAY = {
    "epsilon_greedy": "ε-Greedy", "ucb1": "UCB1",
    "thompson": "Thompson",       "linucb": "LinUCB",
}
ALGO_COLORS = {
    "epsilon_greedy": "#f59e0b", "ucb1": "#10b981",
    "thompson": "#3b82f6",       "linucb": "#ef4444",
}

algorithms = {k: cls() for k, cls in ALGO_CLASSES.items()}

sim_lock  = threading.Lock()
sim_state = {"running": False, "step": 0, "total": 0, "last_results": None}

# ─────────────────────────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>AdRL Studio</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Segoe UI',sans-serif;background:#0f0f1a;color:#e2e8f0;display:flex;height:100vh;overflow:hidden;}
/* Sidebar */
#sidebar{width:240px;min-width:240px;background:#1a1a2e;display:flex;flex-direction:column;padding:0;border-right:1px solid #2d2d4e;}
#sidebar-header{padding:24px 20px 16px;border-bottom:1px solid #2d2d4e;}
#sidebar-header h1{font-size:1.2rem;font-weight:700;color:#fff;letter-spacing:.5px;}
#sidebar-header p{font-size:.72rem;color:#7c3aed;margin-top:4px;}
#nav{padding:12px 0;flex:1;}
.nav-item{display:flex;align-items:center;gap:10px;padding:11px 20px;cursor:pointer;color:#94a3b8;font-size:.85rem;transition:all .2s;border-left:3px solid transparent;}
.nav-item:hover{background:#252545;color:#e2e8f0;}
.nav-item.active{background:#1e1b4b;color:#a78bfa;border-left:3px solid #7c3aed;}
.nav-icon{font-size:1.1rem;width:20px;text-align:center;}
/* Main */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;}
#topbar{height:52px;background:#1a1a2e;border-bottom:1px solid #2d2d4e;display:flex;align-items:center;padding:0 24px;gap:12px;}
#topbar-title{font-size:1rem;font-weight:600;color:#fff;}
#status-dot{width:10px;height:10px;border-radius:50%;background:#22c55e;margin-left:auto;}
#status-dot.running{background:#eab308;animation:pulse 1s infinite;}
#status-label{font-size:.78rem;color:#94a3b8;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}
#content{flex:1;overflow-y:auto;padding:24px;}
/* Cards */
.card{background:#16213e;border-radius:10px;padding:20px;margin-bottom:18px;border:1px solid #2d2d4e;}
.card-title{font-size:.9rem;font-weight:600;color:#a78bfa;margin-bottom:14px;text-transform:uppercase;letter-spacing:.8px;}
/* Grid */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
/* Form controls */
.form-row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin-bottom:16px;}
.form-group{display:flex;flex-direction:column;gap:5px;min-width:150px;}
label{font-size:.78rem;color:#94a3b8;font-weight:500;}
select,input[type=range]{background:#0f0f1a;border:1px solid #2d2d4e;color:#e2e8f0;border-radius:6px;padding:7px 10px;font-size:.82rem;outline:none;}
select:focus{border-color:#7c3aed;}
input[type=range]{padding:0;height:4px;accent-color:#7c3aed;width:100%;}
.range-row{display:flex;justify-content:space-between;font-size:.75rem;color:#64748b;margin-top:2px;}
/* Buttons */
.btn{background:#7c3aed;color:#fff;border:none;border-radius:7px;padding:9px 20px;font-size:.85rem;font-weight:600;cursor:pointer;transition:background .2s;}
.btn:hover{background:#6d28d9;}
.btn:disabled{background:#374151;cursor:not-allowed;}
/* Algo cards */
.algo-card{background:#0f0f1a;border-radius:8px;padding:14px;border:1px solid #2d2d4e;}
.algo-name{font-size:.8rem;font-weight:700;margin-bottom:6px;}
.algo-ad{font-size:1.05rem;font-weight:600;color:#fff;margin-bottom:2px;}
.algo-meta{font-size:.75rem;color:#94a3b8;}
.algo-score{font-size:.8rem;margin-top:6px;}
/* Table */
table{width:100%;border-collapse:collapse;font-size:.82rem;}
th{background:#0f0f1a;color:#94a3b8;padding:8px 12px;text-align:left;font-weight:600;border-bottom:1px solid #2d2d4e;}
td{padding:8px 12px;border-bottom:1px solid #1e293b;color:#e2e8f0;}
tr:last-child td{border-bottom:none;}
/* Progress bar */
.progress-bar{background:#1e293b;border-radius:4px;height:8px;overflow:hidden;margin:10px 0;}
.progress-fill{height:100%;background:#7c3aed;transition:width .3s;border-radius:4px;}
/* Tabs hidden by default */
.tab-pane{display:none;}
.tab-pane.active{display:block;}
/* Stat box */
.stat-box{background:#0f0f1a;border-radius:8px;padding:12px;text-align:center;}
.stat-val{font-size:1.4rem;font-weight:700;color:#a78bfa;}
.stat-lbl{font-size:.72rem;color:#64748b;margin-top:2px;}
/* Verdict */
.verdict-sig{color:#22c55e;font-weight:700;}
.verdict-ns{color:#ef4444;font-weight:700;}
/* Lift row */
.lift-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;}
.lift-box{flex:1;min-width:120px;background:#0f0f1a;border-radius:8px;padding:12px;text-align:center;}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">
    <h1>&#127916; AdRL Studio</h1>
    <p>Contextual Bandit Ad Engine</p>
  </div>
  <nav id="nav">
    <div class="nav-item active" onclick="showTab('live')" id="nav-live">
      <span class="nav-icon">&#127919;</span><span>Live Ad Serving</span>
    </div>
    <div class="nav-item" onclick="showTab('simulation')" id="nav-simulation">
      <span class="nav-icon">&#9654;</span><span>Online Learning</span>
    </div>
    <div class="nav-item" onclick="showTab('regret')" id="nav-regret">
      <span class="nav-icon">&#128200;</span><span>Regret Analysis</span>
    </div>
    <div class="nav-item" onclick="showTab('abtest')" id="nav-abtest">
      <span class="nav-icon">&#9878;</span><span>A/B Test Simulator</span>
    </div>
    <div class="nav-item" onclick="showTab('heatmap')" id="nav-heatmap">
      <span class="nav-icon">&#127777;</span><span>Reward Landscape</span>
    </div>
  </nav>
</div>

<div id="main">
  <div id="topbar">
    <span id="topbar-title">Live Ad Serving</span>
    <div id="status-dot"></div>
    <span id="status-label">Model Ready</span>
  </div>

  <div id="content">

    <!-- TAB 1: Live Ad Serving -->
    <div class="tab-pane active" id="tab-live">
      <div class="card">
        <div class="card-title">&#127891; User Context</div>
        <div class="form-row">
          <div class="form-group">
            <label>Age Group</label>
            <select id="ctx-age">
              <option value="young_adult">Young Adult (18–34)</option>
              <option value="adult" selected>Adult (35–54)</option>
              <option value="senior">Senior (55+)</option>
            </select>
          </div>
          <div class="form-group">
            <label>Device</label>
            <select id="ctx-device">
              <option value="mobile">Mobile</option>
              <option value="desktop" selected>Desktop</option>
              <option value="tablet">Tablet</option>
            </select>
          </div>
          <div class="form-group">
            <label>Time of Day</label>
            <select id="ctx-tod">
              <option value="morning">Morning (6–12)</option>
              <option value="afternoon" selected>Afternoon (12–18)</option>
              <option value="evening">Evening (18–24)</option>
              <option value="night">Night (0–6)</option>
            </select>
          </div>
          <div class="form-group">
            <label>Content Category</label>
            <select id="ctx-content">
              <option value="tech" selected>Tech</option>
              <option value="sports">Sports</option>
              <option value="lifestyle">Lifestyle</option>
              <option value="news">News</option>
              <option value="entertainment">Entertainment</option>
            </select>
          </div>
          <div class="form-group">
            <label>Region</label>
            <select id="ctx-region">
              <option value="north_america" selected>North America</option>
              <option value="europe">Europe</option>
              <option value="asia">Asia</option>
              <option value="other">Other</option>
            </select>
          </div>
          <div class="form-group" style="justify-content:flex-end;">
            <button class="btn" onclick="getRecommendations()">&#128269; Get Recommendations</button>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">&#127917; Algorithm Recommendations</div>
        <div class="grid-4" id="rec-grid">
          <div class="algo-card"><div class="algo-name" style="color:#f59e0b">ε-Greedy</div><div class="algo-ad" id="r-eg-ad">—</div><div class="algo-meta" id="r-eg-meta">—</div><div class="algo-score" id="r-eg-score">—</div></div>
          <div class="algo-card"><div class="algo-name" style="color:#10b981">UCB1</div><div class="algo-ad" id="r-ucb-ad">—</div><div class="algo-meta" id="r-ucb-meta">—</div><div class="algo-score" id="r-ucb-score">—</div></div>
          <div class="algo-card"><div class="algo-name" style="color:#3b82f6">Thompson</div><div class="algo-ad" id="r-ts-ad">—</div><div class="algo-meta" id="r-ts-meta">—</div><div class="algo-score" id="r-ts-score">—</div></div>
          <div class="algo-card"><div class="algo-name" style="color:#ef4444">LinUCB</div><div class="algo-ad" id="r-lu-ad">—</div><div class="algo-meta" id="r-lu-meta">—</div><div class="algo-score" id="r-lu-score">—</div></div>
        </div>
      </div>
    </div>

    <!-- TAB 2: Online Learning Simulation -->
    <div class="tab-pane" id="tab-simulation">
      <div class="card">
        <div class="card-title">&#9881; Simulation Settings</div>
        <div class="form-row">
          <div class="form-group" style="flex:1;max-width:300px;">
            <label>Impressions: <span id="n-val">3000</span></label>
            <input type="range" id="n-impressions" min="1000" max="10000" step="500" value="3000"
                   oninput="document.getElementById('n-val').textContent=this.value"/>
            <div class="range-row"><span>1,000</span><span>10,000</span></div>
          </div>
          <div class="form-group" style="flex:1;max-width:300px;">
            <label>Random Seed: <span id="seed-val">42</span></label>
            <input type="range" id="sim-seed" min="1" max="100" step="1" value="42"
                   oninput="document.getElementById('seed-val').textContent=this.value"/>
            <div class="range-row"><span>1</span><span>100</span></div>
          </div>
          <div class="form-group" style="justify-content:flex-end;">
            <button class="btn" id="run-sim-btn" onclick="runSimulation()">&#9654; Run Simulation</button>
          </div>
        </div>
        <div class="progress-bar" id="sim-progress-bar" style="display:none;">
          <div class="progress-fill" id="sim-progress-fill" style="width:0%;"></div>
        </div>
        <div id="sim-progress-text" style="font-size:.78rem;color:#94a3b8;"></div>
      </div>
      <div class="card">
        <div class="card-title">&#128200; Rolling CTR (100-impression window)</div>
        <div id="sim-chart" style="height:320px;"></div>
      </div>
      <div class="card">
        <div class="card-title">&#128202; Simulation Summary</div>
        <div id="sim-table-container"><p style="color:#64748b;font-size:.82rem;">Run a simulation to see results.</p></div>
      </div>
    </div>

    <!-- TAB 3: Regret Analysis -->
    <div class="tab-pane" id="tab-regret">
      <div class="card">
        <div class="card-title">&#128201; Cumulative Regret Comparison</div>
        <p style="font-size:.78rem;color:#64748b;margin-bottom:12px;">
          Cumulative regret measures the total reward missed vs. always picking the oracle best arm.
          Lower is better. LinUCB and Thompson typically achieve sub-linear regret.
        </p>
        <div id="regret-chart" style="height:340px;"></div>
      </div>
      <div class="card">
        <div class="card-title">&#128203; Regret Summary</div>
        <div id="regret-table-container"><p style="color:#64748b;font-size:.82rem;">Run a simulation first (Online Learning tab).</p></div>
      </div>
      <div style="text-align:right;margin-top:-8px;">
        <button class="btn" onclick="loadRegret()" style="font-size:.78rem;padding:7px 14px;">&#8635; Refresh Regret Data</button>
      </div>
    </div>

    <!-- TAB 4: A/B Test Simulator -->
    <div class="tab-pane" id="tab-abtest">
      <div class="card">
        <div class="card-title">&#9878; A/B Test Settings</div>
        <div class="form-row">
          <div class="form-group">
            <label>Policy A</label>
            <select id="ab-policy-a">
              <option value="linucb" selected>LinUCB</option>
              <option value="epsilon_greedy">ε-Greedy</option>
              <option value="ucb1">UCB1</option>
              <option value="thompson">Thompson</option>
            </select>
          </div>
          <div class="form-group">
            <label>Policy B</label>
            <select id="ab-policy-b">
              <option value="ucb1" selected>UCB1</option>
              <option value="epsilon_greedy">ε-Greedy</option>
              <option value="thompson">Thompson</option>
              <option value="linucb">LinUCB</option>
            </select>
          </div>
          <div class="form-group" style="flex:1;max-width:280px;">
            <label>Impressions: <span id="ab-n-val">5000</span></label>
            <input type="range" id="ab-impressions" min="1000" max="20000" step="1000" value="5000"
                   oninput="document.getElementById('ab-n-val').textContent=this.value"/>
            <div class="range-row"><span>1,000</span><span>20,000</span></div>
          </div>
          <div class="form-group" style="justify-content:flex-end;">
            <button class="btn" id="run-ab-btn" onclick="runABTest()">&#9878; Run A/B Test</button>
          </div>
        </div>
      </div>
      <div id="ab-results" style="display:none;">
        <div class="card">
          <div class="card-title">&#128202; A/B Test Results</div>
          <div class="lift-row">
            <div class="lift-box"><div class="stat-val" id="ab-ctr-a">—</div><div class="stat-lbl" id="ab-lbl-a">Policy A CTR</div></div>
            <div class="lift-box"><div class="stat-val" id="ab-ctr-b">—</div><div class="stat-lbl" id="ab-lbl-b">Policy B CTR</div></div>
            <div class="lift-box"><div class="stat-val" id="ab-lift">—</div><div class="stat-lbl">Absolute Lift</div></div>
            <div class="lift-box"><div class="stat-val" id="ab-lift-rel">—</div><div class="stat-lbl">Relative Lift</div></div>
          </div>
          <div class="lift-row">
            <div class="lift-box"><div class="stat-val" id="ab-z">—</div><div class="stat-lbl">Z-Statistic</div></div>
            <div class="lift-box"><div class="stat-val" id="ab-p">—</div><div class="stat-lbl">P-Value</div></div>
            <div class="lift-box"><div class="stat-val" id="ab-ci">—</div><div class="stat-lbl">95% CI (Lift)</div></div>
            <div class="lift-box" style="flex:2;"><div class="stat-val" id="ab-verdict">—</div><div class="stat-lbl">Verdict</div></div>
          </div>
          <div id="ab-chart" style="height:280px;margin-top:8px;"></div>
        </div>
      </div>
    </div>

    <!-- TAB 5: Reward Landscape -->
    <div class="tab-pane" id="tab-heatmap">
      <div class="card">
        <div class="card-title">&#127777; Reward Landscape Settings</div>
        <div class="form-row">
          <div class="form-group">
            <label>Algorithm</label>
            <select id="hm-algo">
              <option value="linucb" selected>LinUCB</option>
              <option value="epsilon_greedy">ε-Greedy</option>
              <option value="ucb1">UCB1</option>
              <option value="thompson">Thompson</option>
            </select>
          </div>
          <div class="form-group" style="justify-content:flex-end;">
            <button class="btn" onclick="loadHeatmap()">&#8635; Refresh Heatmap</button>
          </div>
        </div>
        <p style="font-size:.76rem;color:#64748b;">Estimated CTR for each user content category × ad category pair. Context held at: adult, desktop, afternoon, north_america.</p>
      </div>
      <div class="card">
        <div class="card-title">&#128200; Estimated CTR Heatmap</div>
        <div id="heatmap-chart" style="height:380px;"></div>
      </div>
    </div>

  </div><!-- /content -->
</div><!-- /main -->

<script>
// ── Tab switching ────────────────────────────────────────────────────────────
const TAB_TITLES = {
  live:'Live Ad Serving', simulation:'Online Learning Simulation',
  regret:'Regret Analysis', abtest:'A/B Test Simulator', heatmap:'Reward Landscape'
};
function showTab(name) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  document.getElementById('topbar-title').textContent = TAB_TITLES[name];
}

// ── Status polling ───────────────────────────────────────────────────────────
function pollStatus() {
  fetch('/api/status').then(r => r.json()).then(d => {
    const dot = document.getElementById('status-dot');
    const lbl = document.getElementById('status-label');
    if (d.running) {
      dot.className = 'running'; dot.style.background = '#eab308';
      lbl.textContent = 'Simulation Running (' + d.step + '/' + d.total + ')';
    } else {
      dot.className = ''; dot.style.background = '#22c55e';
      lbl.textContent = 'Model Ready';
    }
  }).catch(() => {});
}
setInterval(pollStatus, 2000);

// ── Tab 1: Recommendations ───────────────────────────────────────────────────
async function getRecommendations() {
  const body = {
    age:     document.getElementById('ctx-age').value,
    device:  document.getElementById('ctx-device').value,
    tod:     document.getElementById('ctx-tod').value,
    content: document.getElementById('ctx-content').value,
    region:  document.getElementById('ctx-region').value,
  };
  const r = await fetch('/api/recommend', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  const d = await r.json();
  const keys = ['epsilon_greedy','ucb1','thompson','linucb'];
  const ids  = ['eg','ucb','ts','lu'];
  keys.forEach((k, i) => {
    const rec = d[k];
    document.getElementById('r-' + ids[i] + '-ad').textContent  = rec.ad_id + ' (' + rec.category + ')';
    document.getElementById('r-' + ids[i] + '-meta').textContent = rec.format + ' | $' + rec.bid.toFixed(2);
    document.getElementById('r-' + ids[i] + '-score').textContent = 'Est. CTR: ' + (rec.score * 100).toFixed(2) + '%';
  });
}

// ── Tab 2: Simulation ────────────────────────────────────────────────────────
let simRollingData = {};

async function runSimulation() {
  const n    = parseInt(document.getElementById('n-impressions').value);
  const seed = parseInt(document.getElementById('sim-seed').value);
  const btn  = document.getElementById('run-sim-btn');
  const bar  = document.getElementById('sim-progress-bar');
  const fill = document.getElementById('sim-progress-fill');
  const txt  = document.getElementById('sim-progress-text');

  btn.disabled = true;
  bar.style.display = 'block';
  fill.style.width = '0%';
  txt.textContent = 'Starting simulation…';
  simRollingData = {epsilon_greedy:[], ucb1:[], thompson:[], linucb:[], steps:[]};

  try {
    const resp = await fetch('/api/simulate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({n_impressions: n, seed: seed})
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const payload = JSON.parse(line.slice(5).trim());
        const pct = Math.round(payload.step / payload.total * 100);
        fill.style.width = pct + '%';
        txt.textContent = 'Step ' + payload.step + ' / ' + payload.total;
        if (payload.done) {
          renderSimCharts(payload);
          renderSimTable(payload);
          btn.disabled = false;
          txt.textContent = 'Simulation complete — ' + payload.n_impressions + ' impressions.';
        }
      }
    }
  } catch(e) {
    txt.textContent = 'Error: ' + e.message;
    btn.disabled = false;
  }
}

function renderSimCharts(d) {
  const traces = [
    {x: d.steps, y: d.rolling_ctr.epsilon_greedy, name:'ε-Greedy',  line:{color:'#f59e0b'}},
    {x: d.steps, y: d.rolling_ctr.ucb1,           name:'UCB1',      line:{color:'#10b981'}},
    {x: d.steps, y: d.rolling_ctr.thompson,        name:'Thompson',  line:{color:'#3b82f6'}},
    {x: d.steps, y: d.rolling_ctr.linucb,          name:'LinUCB',   line:{color:'#ef4444'}},
  ];
  Plotly.react('sim-chart', traces, {
    template:'plotly_dark', paper_bgcolor:'#16213e', plot_bgcolor:'#0f0f1a',
    margin:{t:10,b:40,l:50,r:10}, autosize:true,
    xaxis:{title:'Impression', color:'#94a3b8', gridcolor:'#1e293b'},
    yaxis:{title:'Rolling CTR', color:'#94a3b8', gridcolor:'#1e293b'},
    legend:{bgcolor:'#16213e', font:{color:'#e2e8f0'}},
  }, {responsive:true});
}

function renderSimTable(d) {
  const keys = ['epsilon_greedy','ucb1','thompson','linucb'];
  const names = {'epsilon_greedy':'ε-Greedy','ucb1':'UCB1','thompson':'Thompson','linucb':'LinUCB'};
  const colors = {'epsilon_greedy':'#f59e0b','ucb1':'#10b981','thompson':'#3b82f6','linucb':'#ef4444'};
  let html = '<table><thead><tr><th>Algorithm</th><th>Final CTR</th><th>Total Reward</th><th>Policy Updates</th></tr></thead><tbody>';
  keys.forEach(k => {
    html += '<tr><td style="color:' + colors[k] + ';font-weight:600;">' + names[k] + '</td>'
      + '<td>' + (d.final_ctr[k] * 100).toFixed(2) + '%</td>'
      + '<td>' + d.total_reward[k] + '</td>'
      + '<td>' + d.n_updates[k] + '</td></tr>';
  });
  html += '</tbody></table>';
  document.getElementById('sim-table-container').innerHTML = html;
}

// ── Tab 3: Regret ────────────────────────────────────────────────────────────
async function loadRegret() {
  const r = await fetch('/api/regret');
  if (!r.ok) { alert('Run a simulation first.'); return; }
  const d = await r.json();
  if (!d.steps || d.steps.length === 0) { alert('No simulation data yet.'); return; }
  const traces = [
    {x:d.steps, y:d.cumulative_regret.epsilon_greedy, name:'ε-Greedy', line:{color:'#f59e0b'}},
    {x:d.steps, y:d.cumulative_regret.ucb1,           name:'UCB1',     line:{color:'#10b981'}},
    {x:d.steps, y:d.cumulative_regret.thompson,        name:'Thompson', line:{color:'#3b82f6'}},
    {x:d.steps, y:d.cumulative_regret.linucb,          name:'LinUCB',  line:{color:'#ef4444'}},
  ];
  Plotly.react('regret-chart', traces, {
    template:'plotly_dark', paper_bgcolor:'#16213e', plot_bgcolor:'#0f0f1a',
    margin:{t:10,b:40,l:50,r:10}, autosize:true,
    xaxis:{title:'Impression', color:'#94a3b8', gridcolor:'#1e293b'},
    yaxis:{title:'Cumulative Regret', color:'#94a3b8', gridcolor:'#1e293b'},
    legend:{bgcolor:'#16213e', font:{color:'#e2e8f0'}},
  }, {responsive:true});

  const keys = ['epsilon_greedy','ucb1','thompson','linucb'];
  const names = {'epsilon_greedy':'ε-Greedy','ucb1':'UCB1','thompson':'Thompson','linucb':'LinUCB'};
  const colors = {'epsilon_greedy':'#f59e0b','ucb1':'#10b981','thompson':'#3b82f6','linucb':'#ef4444'};
  let html = '<table><thead><tr><th>Algorithm</th><th>Final Cumulative Regret</th><th>Avg Per-Step Regret</th></tr></thead><tbody>';
  keys.forEach(k => {
    html += '<tr><td style="color:' + colors[k] + ';font-weight:600;">' + names[k] + '</td>'
      + '<td>' + d.final_regret[k].toFixed(2) + '</td>'
      + '<td>' + d.avg_regret[k].toFixed(4) + '</td></tr>';
  });
  html += '</tbody></table>';
  document.getElementById('regret-table-container').innerHTML = html;
}

// ── Tab 4: A/B Test ──────────────────────────────────────────────────────────
async function runABTest() {
  const pA = document.getElementById('ab-policy-a').value;
  const pB = document.getElementById('ab-policy-b').value;
  const n  = parseInt(document.getElementById('ab-impressions').value);
  if (pA === pB) { alert('Please select two different policies.'); return; }
  const btn = document.getElementById('run-ab-btn');
  btn.disabled = true; btn.textContent = 'Running…';
  try {
    const r = await fetch('/api/abtest', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({policy_a: pA, policy_b: pB, n_impressions: n})
    });
    const d = await r.json();
    const names = {epsilon_greedy:'ε-Greedy', ucb1:'UCB1', thompson:'Thompson', linucb:'LinUCB'};
    document.getElementById('ab-results').style.display = 'block';
    document.getElementById('ab-lbl-a').textContent = names[pA] + ' CTR';
    document.getElementById('ab-lbl-b').textContent = names[pB] + ' CTR';
    document.getElementById('ab-ctr-a').textContent = (d.ctr_a * 100).toFixed(2) + '%';
    document.getElementById('ab-ctr-b').textContent = (d.ctr_b * 100).toFixed(2) + '%';
    document.getElementById('ab-lift').textContent   = (d.lift_abs * 100).toFixed(3) + '%';
    document.getElementById('ab-lift-rel').textContent = (d.lift_rel * 100).toFixed(1) + '%';
    document.getElementById('ab-z').textContent     = d.z_stat.toFixed(3);
    document.getElementById('ab-p').textContent     = d.p_value.toFixed(4);
    document.getElementById('ab-ci').textContent    = '[' + (d.ci_low*100).toFixed(3) + '%, ' + (d.ci_high*100).toFixed(3) + '%]';
    const vEl = document.getElementById('ab-verdict');
    if (d.significant) {
      vEl.textContent = '✅ Significant (p<0.05)'; vEl.className = 'stat-val verdict-sig';
    } else {
      vEl.textContent = '❌ Not Significant'; vEl.className = 'stat-val verdict-ns';
    }
    // Bar chart with error bars
    const ctrA = d.ctr_a, ctrB = d.ctr_b;
    const seA = Math.sqrt(ctrA*(1-ctrA)/d.n_a), seB = Math.sqrt(ctrB*(1-ctrB)/d.n_b);
    const traceAB = {
      x:[names[pA], names[pB]], y:[ctrA, ctrB],
      type:'bar', marker:{color:['#7c3aed','#0ea5e9']},
      error_y:{type:'data', array:[1.96*seA, 1.96*seB], visible:true, color:'#e2e8f0'},
      text:[(ctrA*100).toFixed(2)+'%', (ctrB*100).toFixed(2)+'%'],
      textposition:'outside',
    };
    Plotly.react('ab-chart', [traceAB], {
      template:'plotly_dark', paper_bgcolor:'#16213e', plot_bgcolor:'#0f0f1a',
      margin:{t:20,b:40,l:50,r:10}, autosize:true, showlegend:false,
      yaxis:{title:'CTR', color:'#94a3b8', gridcolor:'#1e293b'},
    }, {responsive:true});
  } catch(e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = '⚖ Run A/B Test';
  }
}

// ── Tab 5: Heatmap ───────────────────────────────────────────────────────────
async function loadHeatmap() {
  const algo = document.getElementById('hm-algo').value;
  const r = await fetch('/api/heatmap', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({algorithm: algo})
  });
  const d = await r.json();
  const trace = {
    z: d.matrix, x: d.ad_cats, y: d.content_cats,
    type:'heatmap', colorscale:'Viridis',
    hoverongaps:false,
    colorbar:{title:'Est. CTR', tickfont:{color:'#e2e8f0'}, titlefont:{color:'#e2e8f0'}},
    text: d.matrix.map(row => row.map(v => (v*100).toFixed(2)+'%')),
    texttemplate:'%{text}', textfont:{color:'#fff', size:11},
  };
  const names = {epsilon_greedy:'ε-Greedy', ucb1:'UCB1', thompson:'Thompson', linucb:'LinUCB'};
  Plotly.react('heatmap-chart', [trace], {
    template:'plotly_dark', paper_bgcolor:'#16213e', plot_bgcolor:'#0f0f1a',
    margin:{t:30,b:60,l:120,r:10}, autosize:true,
    title:{text:'Estimated CTR — ' + names[algo], font:{color:'#e2e8f0', size:13}},
    xaxis:{title:'Ad Category', color:'#94a3b8'},
    yaxis:{title:'User Content Category', color:'#94a3b8'},
  }, {responsive:true});
}

// Auto-load heatmap once the page is fully ready
document.addEventListener('DOMContentLoaded', function() { loadHeatmap(); });
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(TEMPLATE)


@app.route('/api/status')
def api_status():
    with sim_lock:
        return jsonify({
            "running": sim_state["running"],
            "step":    sim_state["step"],
            "total":   sim_state["total"],
        })


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json(force=True)
    try:
        ctx = encode_context(
            data['age'], data['device'], data['tod'],
            data['content'], data['region']
        )
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    result = {}
    for key, algo in algorithms.items():
        ad_idx = algo.select(ctx)
        score  = algo.predict_ctr(ctx, ad_idx)
        ad_id  = AD_IDS[ad_idx]
        result[key] = {
            "ad_id":    ad_id,
            "category": AD_CAT_MAP[ad_id],
            "format":   AD_FORMATS[ad_id],
            "bid":      AD_BIDS[ad_id],
            "score":    round(score, 4),
        }
    return jsonify(result)


@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    data = request.get_json(force=True)
    n_impressions = int(data.get('n_impressions', 3000))
    seed          = int(data.get('seed', 42))
    n_impressions = max(1000, min(10000, n_impressions))

    def generate():
        # Reset all algorithm states
        for algo in algorithms.values():
            algo.reset()

        np.random.seed(seed)

        with sim_lock:
            sim_state['running'] = True
            sim_state['step']    = 0
            sim_state['total']   = n_impressions

        rewards    = {k: [] for k in ALGO_KEYS}
        checkpoint_interval = 50

        # Per-checkpoint rolling window (last 100 impressions)
        rolling_window = 100
        rolling_ctr_series = {k: [] for k in ALGO_KEYS}
        steps_series = []

        # Incremental cumulative regret (avoids O(n²) post-loop recomputation)
        cum_regret       = {k: 0.0 for k in ALGO_KEYS}
        cum_regret_series = {k: [] for k in ALGO_KEYS}

        for t in range(n_impressions):
            ctx = sample_random_context()

            # Vectorized oracle best arm
            all_ctrs   = np.clip(_sigmoid(_TRUE_WEIGHTS @ ctx), 0.02, 0.25)
            oracle_idx = int(np.argmax(all_ctrs))
            oracle_r   = int(np.random.rand() < all_ctrs[oracle_idx])

            # Each algorithm selects, receives reward, updates
            for k, algo in algorithms.items():
                act = algo.select(ctx)
                r   = int(np.random.rand() < all_ctrs[act])
                algo.update(ctx, act, r)
                rewards[k].append(r)
                cum_regret[k] += oracle_r - r

            # Checkpoint every `checkpoint_interval` steps
            if (t + 1) % checkpoint_interval == 0 or t == n_impressions - 1:
                steps_series.append(t + 1)
                for k in ALGO_KEYS:
                    start = max(0, len(rewards[k]) - rolling_window)
                    window = rewards[k][start:]
                    rolling_ctr_series[k].append(round(sum(window) / len(window), 4))
                    cum_regret_series[k].append(round(cum_regret[k], 4))

                with sim_lock:
                    sim_state['step'] = t + 1

                payload = {
                    "step":  t + 1,
                    "total": n_impressions,
                    "done":  False,
                }
                yield f"data: {json.dumps(payload)}\n\n"

        # Final payload with full series
        final_ctr  = {k: round(sum(rewards[k]) / len(rewards[k]), 4) for k in ALGO_KEYS}
        total_rew  = {k: int(sum(rewards[k])) for k in ALGO_KEYS}
        n_upd      = {k: algorithms[k].n_updates for k in ALGO_KEYS}

        # Store for /api/regret
        with sim_lock:
            sim_state['running'] = False
            sim_state['last_results'] = {
                'steps': steps_series,
                'cumulative_regret': cum_regret_series,
                'final_regret': {k: cum_regret_series[k][-1] for k in ALGO_KEYS},
                'avg_regret': {k: round(cum_regret_series[k][-1] / n_impressions, 5) for k in ALGO_KEYS},
            }

        final_payload = {
            "done":          True,
            "step":          n_impressions,
            "total":         n_impressions,
            "n_impressions": n_impressions,
            "steps":         steps_series,
            "rolling_ctr":   rolling_ctr_series,
            "final_ctr":     final_ctr,
            "total_reward":  total_rew,
            "n_updates":     n_upd,
        }
        yield f"data: {json.dumps(final_payload)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/regret')
def api_regret():
    with sim_lock:
        results = sim_state.get('last_results')
    if results is None:
        return jsonify({"error": "No simulation results available. Run a simulation first."}), 404
    return jsonify(results)


@app.route('/api/abtest', methods=['POST'])
def api_abtest():
    data   = request.get_json(force=True)
    key_a  = data.get('policy_a', 'linucb')
    key_b  = data.get('policy_b', 'ucb1')
    n_tot  = int(data.get('n_impressions', 5000))
    n_tot  = max(1000, min(20000, n_tot))

    if key_a not in ALGO_CLASSES or key_b not in ALGO_CLASSES:
        return jsonify({"error": "Invalid policy key"}), 400
    if key_a == key_b:
        return jsonify({"error": "Policy A and B must differ"}), 400

    algo_a = ALGO_CLASSES[key_a]()
    algo_b = ALGO_CLASSES[key_b]()
    n_each = n_tot // 2
    np.random.seed(1)

    r_a, r_b = [], []
    for _ in range(n_each):
        ctx  = sample_random_context()
        act  = algo_a.select(ctx)
        rew  = int(np.random.rand() < true_ctr(act, ctx))
        algo_a.update(ctx, act, rew)
        r_a.append(rew)

    for _ in range(n_each):
        ctx  = sample_random_context()
        act  = algo_b.select(ctx)
        rew  = int(np.random.rand() < true_ctr(act, ctx))
        algo_b.update(ctx, act, rew)
        r_b.append(rew)

    n1, n2   = len(r_a), len(r_b)
    p1, p2   = sum(r_a) / n1, sum(r_b) / n2
    p_pool   = (sum(r_a) + sum(r_b)) / (n1 + n2)
    se       = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if p_pool not in (0, 1) else 1e-9
    z        = (p1 - p2) / se
    p_value  = float(2 * (1 - stats.norm.cdf(abs(z))))
    se_diff  = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    ci_low   = (p1 - p2) - 1.96 * se_diff
    ci_high  = (p1 - p2) + 1.96 * se_diff

    return jsonify({
        "ctr_a":      round(p1, 5),
        "ctr_b":      round(p2, 5),
        "n_a":        n1,
        "n_b":        n2,
        "lift_abs":   round(p1 - p2, 5),
        "lift_rel":   round((p1 - p2) / max(p2, 1e-9), 5),
        "z_stat":     round(z, 4),
        "p_value":    round(p_value, 5),
        "ci_low":     round(ci_low, 5),
        "ci_high":    round(ci_high, 5),
        "significant": p_value < 0.05,
    })


@app.route('/api/heatmap', methods=['POST'])
def api_heatmap():
    data     = request.get_json(force=True)
    algo_key = data.get('algorithm', 'linucb')
    if algo_key not in algorithms:
        return jsonify({"error": "Invalid algorithm"}), 400

    algo       = algorithms[algo_key]
    ad_cats    = ["Tech", "Fashion", "Finance", "Food", "Travel"]
    matrix     = []

    for content in CONTENT_CATS:
        row = []
        for ad_cat in ad_cats:
            # Representative ad: first ad of this category
            ad_idx_for_cat = ad_cats.index(ad_cat) * 4  # ad_01, ad_05, ad_09, ad_13, ad_17
            ctx = encode_context("adult", "desktop", "afternoon", content, "north_america")
            score = algo.predict_ctr(ctx, ad_idx_for_cat)
            row.append(round(float(score), 5))
        matrix.append(row)

    return jsonify({
        "matrix":       matrix,
        "content_cats": CONTENT_CATS,
        "ad_cats":      ad_cats,
        "algorithm":    algo_key,
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)
