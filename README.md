# Pac-Man Autonomous Agent Projects

Welcome to the repository for our Pac-Man autonomous agent projects! This repository contains two main practice folders: Practice_1 and Practice_2, each with their own set of files and objectives. 

This project was awarded first place in the class competition of Machine Learning I at UC3M. In collaboration with Marina Gómez Rey ([GitHub Profile](https://github.com/MarinaGRey)). Note that only our code has been updated, not all class materials.

## Overview

### Project 1
The aim of Project 1 was to develop an autonomous Pac-Man agent capable of efficiently playing the game and maximizing its score. This involved:
- Selecting and refining various algorithms.
- Creating and testing two distinct models:
  - **Model 1:** Focused on quickly eating ghosts.
  - **Model 2:** Concentrated on maximizing score by including pac-dots.

The **J48 decision tree algorithm** was identified as the most effective for predicting Pac-Man's next move. In contrast, regression models revealed that **M5** was best for predicting future scores. Despite Model 1 having better accuracy, Model 2, which included pac-dots, offered superior overall results in terms of gameplay fluency and scoring.

### Project 2
In Practice 2, the objective was to implement a Q-learning algorithm to enable an autonomous Pac-Man agent to navigate various maze configurations and maximize rewards. Key components included:
- Designing a state representation with direction and distance attributes.
- Setting up a Q-table with 16 states and action columns.
- Adjusting learning parameters like alpha, epsilon, and discount factors.

Significant improvements were achieved by:
- Simplifying the reward function to prioritize eating pac-dots and penalizing wall collisions.
- Developing methods for updating the Q-table, calculating rewards, and determining Pac-Man’s position and nearest elements.

This approach proved more effective than previous models, allowing Pac-Man to optimally eat both ghosts and pac-dots, resulting in higher scores.

## Repository Structure

This repository contains the following folders and files:

### Practice_1
- **different_models/**: Folder containing different model files.
- **Practice 1.docx**: Document detailing the objectives, methods, and results of Project 1.
- **bustersAgents.py**: Python script related to the Pac-Man agent for Project 1.
- **dots_training.arff**: ARFF file used for training models in Project 1.

### Practice_2
- **Practice 2.docx**: Document detailing the objectives, methods, and results of Project 2.
- **bustersAgents.py**: Python script related to the Pac-Man agent for Project 2.
- **qtable.txt**: File containing the Q-table used for Q-learning in Project 2.

