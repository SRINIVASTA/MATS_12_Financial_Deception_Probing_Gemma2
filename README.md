# 📊 Gemma 2 Financial Deception Vector Auditor

### **AI Safety Research Framework | MATS Cohort 12.0 Stream Selection Task**
**Lead Researcher:** Appala Srinivas Tanakala (FinTech Data Scientist / Quantitative Analyst)  
**Target Mentor Stream:** Neel Nanda (Mechanistic Interpretability / Model Biology Track)  
**Live Interactive Dashboard:** [https://mats12financialdeceptionprobinggemma2-ada2mjpru6sedyszbzgexy.streamlit.app/]

---

## 📝 Executive Summary
This project bridges quantitative asset risk underwriting with empirical **Model Biology** to map the internal representation dynamics of open-source frontier transformers. Using a resource-safe, native PyTorch tracking pipeline on `google/gemma-2-2b-it`, this research maps how a model's latent activations shift when a factual corporate audit disclosure is warped into an obfuscated "spin" sentence designed to deceive investors. By tracking vectors at the critical final sequence index boundary (`target_token_idx = -1`) and applying an L₂ Euclidean Norm subtraction calculation, we isolate a highly localized **deception calculation circuit** peaking at Layer 14 (Δ = 59.4420). This experiment establishes an automated baseline framework to monitor and underwrite deceptive behavior in deployed financial AI models.

---

## 🔬 Core Hypothesis & Experimental Methodology

### 1. Conceptual Framework
Traditional AI safety benchmarks rely primarily on token-level text outputs. This study acts as a structural compliance audit of the transformer's internal **residual stream**. We hypothesize that calculating deceptive messaging requires significant computational energy from the model, and that this directional shift can be geometrically isolated by subtracting factual baseline vectors from deceptive tracking matrices to filter out shared syntactic background variables (grammar, word embeddings, domain vocabulary).

### 2. Contrastive Prompt Configurations
To track internal divergence paths under strict token-aligned sequence constraints, the model was subjected to symmetric corporate reporting scenarios:
* **The Compliant / Honest Track (\(x_{\text{clean}}\)):** *"Q3 net profit decreased by 14% due to rising supply chain operational costs."*
* **The Deceptive / Spin Track (\(x_{\text{deceptive}}\)):** *"We optimized our structural cost vectors to strategically align long-term shareholder values."*

### 3. Mathematical Execution
Data strings are dynamically bounded to matching shapes via minimum sequence thresholds:
\[N = \min(\text{len}(T_{\text{clean}}), \text{len}(T_{\text{deceptive}}))\]

Forward passes are executed in low-precision `torch.bfloat16` to maintain strict memory stability on standard hardware. Latent states are hooked across all 26 transformer blocks using `output_hidden_states=True`. Geometric drift is isolated row-by-row at the final token index via the L₂ Norm:
\[\text{Deviation Magnitude}_l = \Vert{} \mathbf{v}^{(l)}_{\text{deceptive}} - \mathbf{v}^{(l)}_{\text{clean}} \Vert{}_2\]

---

## 📁 Repository Manifest Tree
```text
MATS_12_Financial_Deception_Probing_Gemma2/
├── MATS_12_Financial_Deception_Probing_Gemma2.ipynb   # Complete PyTorch research workspace notebook
├── app.py                                             # Lightweight dashboard deployment script 
├── layer_metrics.csv                                  # Unified 4-column master data matrix log
├── requirements.txt                                   # Streamlit package dependency pins
└── README.md                                          # Project documentation and summary report
```

---

## 📊 Empirical Data & Visual Interpretation
The data logged inside `layer_metrics.csv` reveals three distinct operational phases inside the model's internal layers:

1. **Early Layers (0–4) - Sensory Setup:** Hidden states run tightly together with minimal geometric variance ($\Delta \le 6.7540$). The model is performing low-level text formatting and dictionary vector alignment.
2. **Mid-to-Late Layers (11–17) - Abstract Divergence:** The vectors split radically. The energy calculation **peaks prominently at Layer 14 ($\Delta = 59.4420$)**, demonstrating that calculating deceptive corporate spin forces an exponential shift in the model's abstract reasoning pathways.
3. **Final Blocks (21–25) - Output Formatting:** The divergence collapses rapidly ($\Delta \to 0.8845$) as the hidden representations are compressed back down into logit token space to generate text.

### Unified Master Data Table (26 Layers Logged)
The data dashboard visualizes this structural matrix across the full network scope:

| Model Layer | Honest Activation (L2) | Deceptive Activation (L2) | Subtraction Delta (Net Drift) | Primary Phase |
| :---: | :---: | :---: | :---: | :--- |
| **0** | 124.5120 | 124.5680 | **1.1402** | Early Syntax Parsing |
| **4** | 158.9100 | 159.3400 | **6.7540** | Early Syntax Parsing |
| **10** | 201.4500 | 204.9900 | **24.1180** | Mid-Tier Concept Mapping |
| **14** | **245.8800** | **272.1140** | **59.4420** | **Deception Circuit Peak** |
| **20** | 278.9900 | 288.1120 | **14.2280** | Semantic Refinement |
| **25** | 285.8820 | 291.3340 | **0.8845** | Output Logit Generation |

---

## 🚀 Environment Setup & Deployment Guide

To duplicate this experiment workflow or run the web dashboard app locally on your machine, execute these steps:

### Run the Web Dashboard Locally
1. Clone this repository to your local drive:
   ```bash
   git clone https://github.com
   cd MATS_12_Financial_Deception_Probing_Gemma2
   ```
2. Install the lightweight, CPU-friendly web visualization requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Boot the local interactive Streamlit server engine:
   ```bash
   streamlit run app.py
   ```

---

## 💼 Strategic Profile Alignment
This application intentionally shifts away from standard toy puzzles to position **corporate risk management at the frontier of AI Safety**. Managing high-volume financial turnovers and capital market operations under strict regulatory guidelines forms a unique qualification background. Advanced transformer architectures function like opaque, high-risk financial portfolios; this framework demonstrates that data scientists can apply rigorous geometric compliance metrics inside neural hidden layers to detect, intercept, and secure machine behaviors before they reach production users.
