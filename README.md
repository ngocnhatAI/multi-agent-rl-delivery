# 🚚 Delivery Optimization with Multi-Agent Reinforcement Learning (MARL)

This project simulates a delivery system powered by **Multi-Agent Reinforcement Learning (MARL)**. Each agent represents a delivery unit that learns to coordinate with others to optimize delivery time, distance, and overall efficiency.

## ⚙️ How to Run

1. Make sure the script is executable:
```bash
chmod +x run.sh
````

2. Run the project:

```bash
./run.sh
```

This script will launch the training or evaluation loop (depending on your configuration inside the code).

---

## 🧠 MARL Overview

* Each agent learns to make decisions such as which package to pick up and which route to take.
* Agents are trained using **MAPPO** algorithm.
* The environment is custom-built with frameworks **Gymnasium**.


## 📦 Requirements

* Python 3.x
* Dependencies: `torch`, `numpy`, `gym`, `pettingzoo`, etc.