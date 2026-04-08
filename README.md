# AraEduAgent
**AraEduAgent** is an Arabic framework for modeling student learning behavior in online learning environments. This project translates the EduAgent dataset into Modern Standard Arabic (MSA) and benchmarks multiple large language models (LLMs) on student simulation tasks across behavioral, cognitive, and assessment metrics.

---
## Table of Contents
- [Overview](#overview)
- [Research Context](#research-context)
- [Models Evaluated](#models-evaluated)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Student Persona Configuration](#student-persona-configuration)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)
---
## Overview
AraEduAgent extends the EduAgent generative student simulation framework to support the Arabic language and culturally relevant educational contexts. The framework leverages LLM-based virtual student agents whose behavior is conditioned on rich demographic and psychological profiles, including academic background, motivation, focus, curiosity, and family support.
The project benchmarks five LLMs — two commercial and three open-source (including **ALLaM-7B**, an Arabic-specialized model from SDAIA) — against real student behavioral data collected from **301 students** enrolled in online courses. Evaluation covers:
- **Behavioral metrics**: gaze-based attention and engagement patterns from eye-tracking (AOI) data
- **Cognitive metrics**: focus, curiosity, compliance, and prior knowledge indicators
- **Assessment metrics**: post-lecture quiz performance and answer accuracy
---
## Research Context
This work addresses the gap in Arabic-language educational AI research by adapting the EduAgent benchmark to Modern Standard Arabic. It employs Arabic-translated course materials, video lecture transcripts with timestamps, student demographic profiles, and real gaze-tracking data to simulate and evaluate student learning behavior in an eLearning setting.
The dataset originates from a study conducted at **Umm Al-Qura University**, Makkah, Saudi Arabia, involving 301 students across diverse academic majors and educational backgrounds.

---
## Models Evaluated
| Model | Type | Provider |
|---|---|---|
| GPT-3.5 Turbo | Commercial | OpenAI |
| GPT-4o Mini | Commercial | OpenAI |
| Llama 3.1 (8B) | Open Source | Meta |
| Qwen2.5 (7B) | Open Source | Alibaba |
| ALLaM-7B Instruct | Open Source (Arabic-specialized) | SDAIA |

### ALLaM Context-Reduction Strategies
ALLaM-7B has a 4,096-token context limit that prevents direct use of the full EduAgent prompt. We evaluated three strategies to overcome this:
- **S1** — Memory-free with cognitive prior (`forget_effect=no_memory`, `KM` only, `standard_cog`)
- **S2** — LLMLingua-2 automatic compression (full memory `KM+PM+MM+CM` + XLM-RoBERTa sanitization + vocab clamping)
- **S3** — Memory-free without cognitive prior (`forget_effect=no_memory`, `KM` only, `standard`)

S3 is used in the main benchmark results; S1 achieves the lowest Focus/Engagement MAE among the three strategies. See `run_allam_cuda.py` and `analyze_results_with_allam_done.ipynb`.

---
## Dataset
The `dataset/` directory contains all data files used for simulation and evaluation:
| File | Description |
|---|---|
| `student_demo.csv` | Real student demographic data (301 students) |
| `student_demo_generated.csv` | Synthetically generated virtual student demographics for simulation |
| `student_question.csv` | Post-lecture quiz questions per course segment |
| `student_answer_item_revised.csv` | Student responses and correctness labels |
| `course_material_slide.csv` | Mappings of course slides to lecture content |
| `aoi_material_ext_slide.csv` | Areas of Interest (AOI) gaze data aligned to course slides |
| `during_behavior_slide.csv` | Gaze tracking and engagement behavior data per slide |
### Subdirectories
- `dataset/simulation/full_experiment_300_students/` — Output files from the full simulation run across 300 virtual students
- `dataset/final_results/` — Figures and tables from the primary evaluation
- `dataset/new_results/` — Updated visualizations from revised analysis runs
- `dataset/NLP PROJECT/` — Supporting materials from the NLP coursework component
---
## Project Structure
```
AraEduAgent/
│
├── simulation_experiment.ipynb       # Main LLM simulation pipeline notebook
├── analysis_results.ipynb            # Results analysis and visualization notebook
├── student_demo_config.py            # Virtual student demographic profile configurations (Arabic)
├── transcript_map.py                 # Arabic video lecture transcripts with timestamps
├── .gitignore
│
└── dataset/
    ├── student_demo.csv
    ├── student_demo_generated.csv
    ├── student_question.csv
    ├── student_answer_item_revised.csv
    ├── course_material_slide.csv
    ├── aoi_material_ext_slide.csv
    ├── during_behavior_slide.csv
    │
    ├── simulation/
    │   └── full_experiment_300_students/   # Per-student simulation outputs
    ├── final_results/                      # Primary evaluation figures and tables
    └── new_results/                        # Updated visualizations
```
---
## Student Persona Configuration
Virtual student agents are configured using rich demographic and psychological attributes defined in `student_demo_config.py`. Each student persona is described in Modern Standard Arabic across 13 dimensions:
| Attribute | Description |
|---|---|
| `age` | Age group (18–24, 25–31, 32–38, >39) |
| `gender` | Gender identity |
| `major` | Academic field (Humanities, Social Sciences, Natural Sciences, Engineering & Technology, Business & Economics, Health Sciences) |
| `education` | Highest education level (High School, Undergraduate, Master's, PhD) |
| `attitude` | Motivational attitude toward learning (enthusiastic ↔️ disengaged) |
| `exam` | Historical academic performance (high GPA ↔️ low GPA) |
| `focus` | Attention level during lectures (highly focused ↔️ distracted) |
| `curiosity` | Intellectual curiosity toward course content |
| `interest` | Interest level in the subject matter |
| `priors` | Prior knowledge and background in the course topic |
| `compliance` | Ability to keep pace with instructor teaching speed |
| `smartness` | Speed of comprehension and intellectual aptitude |
| `family` | Family academic background and educational support |
Each attribute is encoded as a binary or categorical value and rendered as a natural-language Arabic string when constructing LLM prompts.
---
## Requirements
- Python 3.8+
- Jupyter Notebook
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- OpenAI API key (for GPT-3.5 Turbo and GPT-4o Mini)
- HuggingFace access token (for Llama 3.1 and Qwen2.5)
---
## Installation
1. **Clone the repository**
```bash
git clone https://github.com/areejadel/AraEduAgent.git
cd AraEduAgent
```
2. **Install Python dependencies**
```bash
pip install pandas numpy matplotlib seaborn jupyter openai huggingface_hub
```
3. **Set up API credentials**
For OpenAI models, set your API key as an environment variable:
```bash
export OPENAI_API_KEY="your_openai_api_key"
```
For HuggingFace models (Llama 3.1, Qwen2.5), authenticate via:
```bash
huggingface-cli login
```
---
## Usage
### 1. Configure Student Personas
Review and modify virtual student demographic configurations in `student_demo_config.py` to adjust the range of simulated student profiles used in experiments.
### 2. Run the Simulation Experiment
Open and execute `simulation_experiment.ipynb` in Jupyter Notebook:
```bash
jupyter notebook simulation_experiment.ipynb
```
This notebook:
- Loads real and generated student demographics
- Constructs Arabic-language student personas using `student_demo_config.py`
- Feeds course material from `transcript_map.py` and `course_material_slide.csv` to each LLM
- Simulates student responses to post-lecture quiz questions
- Saves outputs to `dataset/simulation/`
### 3. Analyze Results
Open and execute `analysis_results.ipynb` to compute evaluation metrics and generate visualizations:
```bash
jupyter notebook analysis_results.ipynb
```
This notebook:
- Compares simulated student responses against real student answers
- Evaluates behavioral accuracy (focus, attention) against gaze-tracking data
- Produces plots and tables saved to `dataset/final_results/` and `dataset/new_results/`
---
## Results
Results from the full experiment across 300 virtual students are stored in:
- `dataset/simulation/full_experiment_300_students/` — Raw simulation outputs per model
- `dataset/final_results/` — Evaluation figures and summary tables
- `dataset/new_results/` — Refined plots from updated analysis runs
Evaluation dimensions include:
- **Assessment accuracy**: How well simulated quiz answers match real student responses
- **Behavioral fidelity**: Alignment of simulated attention/engagement with eye-tracking data
- **Cognitive plausibility**: Whether generated persona behaviors are consistent with configured psychological traits
---
## Citation
If you use AraEduAgent in your research, please cite AraEduAgent paper:
```bibtex
@article{AraEduAgent,
  title   = {AraEduAgent: Arabic EduAgent Dataset for Peer-Attention Modeling in eLearning},
  author  = {Areej A. Bawazir, Asmaa K. Alshumrani, Reem A. Qaid, Walaa S. Alfahmi, Mourad Mars},
  journal = {arXiv preprint arXiv:XXXXXXX},
  year    = {2026}
}
```

If you use AraEduAgent in your research, please cite also the original EduAgent paper:
```bibtex
@article{xu2024eduagent,
  title   = {EduAgent: Generative Student Agents in Learning},
  author  = {Xu, Songlin and Zhang, Xinyu and Qin, Lianhui},
  journal = {arXiv preprint arXiv:2404.07963},
  year    = {2024}
}
```

---
## License
This project is released for academic research purposes.
