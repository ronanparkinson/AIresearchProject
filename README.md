# 🚀 AI-Driven Cloud Pipeline Automation using Reinforcement Learning

## 🧠 Overview

This project explores the use of reinforcement learning (**PPO**) for proactive autoscaling in cloud environments.

Traditional rule-based autoscaling approaches react to system load, often leading to inefficient resource usage and latency spikes. In contrast, this project investigates whether a reinforcement learning agent can learn optimal scaling strategies under dynamic and noisy workloads.

The system is evaluated using real-world workload traces from the **Google Cluster dataset**, with a focus on **cost-performance trade-offs** and **system stability**.

---

## ⚠️ Problem Statement

Cloud-based systems commonly rely on reactive autoscaling policies, which can result in:

- 📉 **Over-provisioning** → increased infrastructure cost  
- ⚡ **Under-provisioning** → degraded performance and latency spikes  
- 🔄 **Instability** under fluctuating workloads  

This project investigates whether reinforcement learning can improve scaling decisions by learning proactive policies that balance **performance**, **cost**, and **system reliability**.

---

## 🧪 Approach

- 🧩 Modelled autoscaling as a reinforcement learning problem  
- 🤖 Implemented a **PPO (Proximal Policy Optimisation)** agent using Stable-Baselines3  
- 🎯 Designed a custom reward function incorporating:
  - 💰 Resource cost
  - ⏱️ Latency penalties
  - ⚖️ Scaling stability

- 📊 Simulated workloads using the Google Cluster trace dataset  
- 🔍 Compared the RL-based approach against a rule-based autoscaling baseline  

---

## 🏗️ System Design

The system simulates a cloud environment with dynamic workload behaviour.

### 🔑 Key Components:

- 🖥️ **Environment**  
  Simulates CPU usage, scaling decisions, and system response to workload changes  

- 🤖 **RL Agent (PPO)**  
  Learns optimal scaling actions based on observed system state  

- 📦 **Workload Input**  
  Derived from Google Cluster trace dataset to reflect real-world conditions  

- 📏 **Baseline Policy**  
  Rule-based autoscaling strategy used for comparison  

### ⚙️ Real-World Considerations

- 🌊 Noisy and non-stationary workloads  
- ⏳ Delayed reward signals  
- ⚖️ Trade-offs between performance and cost  

---

## 📈 Results

The reinforcement learning approach demonstrated:

- 💸 Improved cost efficiency compared to rule-based scaling  
- 🔄 Better stability under fluctuating workloads  
- 🧠 Enhanced handling of noisy and non-stationary conditions  

### 📊 Evaluation Metrics:

- Cost efficiency  
- Latency impact  
- Scaling stability  

---

## 🛠️ Tech Stack

- 🐍 Python  
- 🤖 Stable-Baselines3 (PPO)  
- 🔢 NumPy / Pandas  
- 📉 Matplotlib (visualisation)  
- ☁️ Google Cluster Dataset  

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/ronanparkinson/AIresearchProject.git
cd AIresearchProject
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python ppoTraining.py
```

4. Run evaluation:

```bash
python runEvaluation.py
```

---

🧠 Key Learnings
Reinforcement learning performance is highly sensitive to reward function design
Delayed rewards and noisy environments introduce training instability
Real-world systems require balancing cost, performance, and reliability
Simulating realistic workloads is critical for meaningful evaluation
---

🔮 Future Work
☁️ Integrate with real cloud orchestration systems (e.g., Kubernetes)
🎯 Improve reward shaping and training stability
🔗 Extend to multi-service or multi-cluster scaling scenarios
📡 Incorporate anomaly detection and predictive workload modelling

---

👤 Author

Ronan Parkinson
