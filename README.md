# Classifying, storing and exploring flower images

This project aims to produce a flower classification pipeline
using IA libraries (TensorFlow, Pytorch), database tools (SQLAlchemy),
project management tools (Poetry).

The only use of this project is me getting used to these different tools.
Hopefully the end result can be something cool, but useless.

## Features

- Clean Python project with Poetry
- Use TensorFlow Flowers dataset
- Convert dataset to a DB with SQLAlchemy
- Train a CNN model with PyTorch on this DB
- Explore metadata and model performance with Streamlit

## Installation

```bash
git clone https://github.com/ismael-lajaaiti/flower_classifier_db
cd flower_classifier_db
poetry install
```

## Run training

```bash
poetry run python -m flower_classifer_db.train
```

## Launch dashboard

```bash
poetry run streamlit run dashboard/app.py
```
