# Quantum Test Task

This repository contains three independent tasks (algorithms, ML, and OOP) used for an evaluation exercise. Each task lives under `src/` .

**Project layout**

- Task 1 — Counting islands  [src/task1](src/task1)
    - [src/task1/main.py](src/task1/main.py)
    - implements an efficient island counting algorithm (union-find)
- Task 2 — Regression (tabular ML) [src/task2](src/task2)
    - notebook: [eda.ipynb](src/task2/eda.ipynb),
    - scripts: [train.py](src/task2/train.py), [predict.py](src/task2/predict.py)
    - requirements: [requirements.txt](src/task2/requirements.txt)
    - data in `/data` folder, predictions are saved to `/data/predictions.csv`
- Task 3 — MNIST classifier (OOP) [src/task3](src/task3) 
    - files: [main.py](src/task3/main.py), [classifier.py](src/task3/classifier.py), [interfaces.py](src/task3/interfaces.py), [models.py](src/task3/models.py)
    - A small demo entrypoint is provided in `main.py`
----