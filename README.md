# 🎓 AraEduAgent

An Arabic adaptation of the [EduAgent](https://github.com/EduAgent/EduAgent) framework for modeling student learning behavior in online learning environments. This project translates the EduAgent dataset into Modern Standard Arabic (MSA) and benchmarks multiple LLMs on student simulation tasks.

## Models Used

| Model | Type |
|---|---|
| GPT-3.5 Turbo | Commercial (OpenAI) |
| GPT-4o Mini | Commercial (OpenAI) |
| Llama 3.1 (8B) | Open Source (Meta) |
| Qwen2.5 (7B) | Open Source (Alibaba) |

## Project Structure

```
├── simulation_experiment.ipynb         # Main simulation experiment notebook
├── analysis_results.ipynb              # Results analysis and visualization
├── student_demo_config.py              # Virtual student demographic configurations
├── transcript_map.py                   # Arabic video lecture transcripts with timestamps
│
└── dataset/
    ├── student_demo.csv                # Real student demographics
    ├── student_demo_generated.csv      # Generated virtual student demographics
    ├── student_question.csv            # Post-lecture quiz questions
    ├── student_answer_item_revised.csv # Student responses and accuracy
    ├── course_material_slide.csv       # Course slide mappings
    ├── aoi_material_ext_slide.csv      # Areas of Interest (AOI) data
    ├── during_behavior_slide.csv       # Gaze tracking and engagement data
    ├── simulation/                     # Simulation experiment outputs
    ├── final_results/                  # Figures and tables
    └── new_results/                    # Updated visualizations
```

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- LLM API access (OpenAI, HuggingFace)

## Usage

1. Configure student personas in `student_demo_config.py`
2. Run simulation in `simulation_experiment.ipynb`
3. Analyze results in `analysis_results.ipynb`

## Based On

This work builds on the [EduAgent](https://github.com/EduAgent/EduAgent) framework:

```bibtex
@article{xu2024eduagent,
  title={EduAgent: Generative Student Agents in Learning},
  author={Xu, Songlin and Zhang, Xinyu and Qin, Lianhui},
  journal={arXiv preprint arXiv:2404.07963},
  year={2024}
}
```

## Authors

Areej A. Bawazir, Asmaa K. Alshumrani, Reem A. Qaid, Walaa S. Alfahmi, Mourad Mars

Department of Computer Science, Umm Al-Qura University, Makkah, Saudi Arabia
