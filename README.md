# Cost-Effectiveness Model for CAD Diagnostic Strategies

This repository implements a Python-based cost-effectiveness model comparing **FFRCT** (Fractional Flow Reserve from CT) and **Stress Testing** as diagnostic strategies for **coronary artery disease (CAD)** in diverse rural populations across the U.S.

## 🧠 Overview

The model simulates the long-term health and economic outcomes of patients undergoing different diagnostic pathways, integrating:
- A **decision tree** for initial test performance and treatment allocation
- A **Markov model** to track health state transitions over a 25-year time horizon
- Detailed cost and quality-adjusted life year (QALY) calculations, including rural disutilities and transport costs

## 🔧 Structure

- `improved_model.py`: Core simulation logic with parameter definitions, decision tree, and Markov model
- `run_model.py`: Runner script that executes scenarios and saves annualized results to CSV

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- NumPy
- Pandas
- (Optional) Virtual environment setup

### Installation
```bash
git clone https://github.com/zhornbe/econ_model.git
cd econ_model
pip install -r requirements.txt  # If you create one
```

### Run the Model
```bash
python run_model.py
```

This will:
- Run cost-effectiveness simulations for all rural distance strata (`<50`, `50–100`, `>100` miles)
- Print QALYs, costs, and ICERs for each strategy
- Save annual state-level results as CSVs (e.g., `Distance_lt_50_miles_FFRCT_Annual.csv`)

## 📊 Outputs

Each run includes:
- Total discounted **QALYs** and **costs** for each strategy
- **ICERs** (incremental cost-effectiveness ratios) vs. stress testing
- Per-cycle breakdowns (age, cost, QALY) saved to disk

## 📌 Assumptions

Real-world parameters have been used where possible and are clearly cited in code comments. Key assumptions include:
- Test performance metrics from NEJM/JACC studies
- Clinical effectiveness from COURAGE, ISCHEMIA, and PROMISE trials
- CMS-derived cost estimates (CY2024)
- Rural patient disutilities and transport burdens

See inline comments in `improved_model.py` for citations and justification.

## 🛠️ Customization

You can adjust:
- Test sensitivity/specificity
- Prevalence
- Transition probabilities
- Cost inputs (diagnostic, procedural, transport)
- Utilities by health state

## 📎 Example Output
```
--- Scenario: Distance 50-100 miles ---

Strategy: FFRCT
  Total QALYs: 12.35
  Total Costs: $32,108.45

Strategy: Stress Test
  Total QALYs: 12.21
  Total Costs: $27,802.12

ICERs (vs Stress Test):
  FFRCT: $29,307.89 per QALY
```

## 📚 Citations
- Douglas PS et al., *NEJM*, 2015 (PLATFORM Study)
- Nørgaard BL et al., *JACC*, 2014
- Maron DJ et al., *NEJM*, 2020 (ISCHEMIA Trial)
- Boden WE et al., *NEJM*, 2007 (COURAGE Trial)
- CMS Medicare Fee Schedule, CY2024

## 📄 License

This project is open-sourced under the MIT License.

---

*For questions or contributions, feel free to open an issue or pull request.*
