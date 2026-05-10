# AI Lab Project Overview

## Student Information
- **Name**: Saad Zaidi
- **Enrollment**: K23-0874
- **Institution**: FAST-NUCES Karachi

---

## Project 1: A* 8-Puzzle Solver

### Overview
An **interactive graphical pathfinding application** that solves the classic 8-puzzle problem using the A* search algorithm. The application visualizes the solution process, heuristic search space exploration, and step-by-step puzzle transformations in real-time using Pygame.

### Problem Description
The 8-puzzle is a sliding puzzle with:
- A 3×3 grid containing tiles numbered 1-8 and one empty space (0)
- Goal: Arrange tiles in order (1,2,3,4,5,6,7,8,0) from left to right, top to bottom
- Valid moves: Slide adjacent tiles into the empty space

### Key Features

#### 1. **Algorithm Implementation**
- **A* Search Algorithm**: Combines actual path cost (g) with heuristic estimate (h)
  - Formula: `f(n) = g(n) + h(n)`
  - Explores minimum number of states while guaranteeing optimal solution
- **Solvability Check**: Uses inversion counting—puzzles have even/odd parity that determines solvability

#### 2. **Heuristic Functions**
Two heuristics available for comparison:

| Heuristic | Formula | Description |
|-----------|---------|-------------|
| **Manhattan Distance** | Sum of |row_diff| + |col_diff| for each tile | More informed; better performance |
| **Misplaced Tiles** | Count of tiles not in goal position | Less informed; explores more states |

#### 3. **Core Components**

**Puzzle Logic Functions**:
- `is_solvable(state)`: Counts inversions to determine if puzzle is solvable
- `get_neighbors(state)`: Generates valid moves by sliding tiles
- `manhattan(state)` / `misplaced(state)`: Heuristic evaluation functions
- `astar(start, heuristic_fn)`: Main A* algorithm implementation

**Visualization**:
- Current puzzle board (main 3×3 grid display)
- Solution timeline: Scrollable panel showing all steps from initial to goal state
- Progress indicator: Visual bar tracking solution progress
- State exploration display: Shows nodes explored during search

#### 4. **Interactive Controls**
The application provides a rich GUI with:
- **Random Generate**: Create new random solvable puzzle
- **Solve**: Run A* algorithm with selected heuristic
- **Play/Step**: Animate solution or advance one step at a time
- **Reset**: Return to initial state
- **Switch Heuristic**: Toggle between Manhattan and Misplaced heuristics
- **Toggle Explored States**: Visualize search space exploration

#### 5. **How It Works** (Step-by-Step)

1. **Initialization**
   - Random solvable puzzle generated using shuffle + solvability check
   - A* queue initialized with starting state and heuristic value

2. **Search Process**
   ```
   While open queue not empty:
     - Pop state with lowest f(n) = g(n) + h(n)
     - If state = goal → return solution path
     - Generate all neighbors (up to 4 valid moves)
     - Add unvisited neighbors to queue with updated costs
   ```

3. **Visualization**
   - Each step displayed with tile positions
   - Timeline shows progression through solution
   - Explored nodes highlight search space coverage

4. **Optimality**
   - A* guarantees shortest solution path
   - Manhattan heuristic is **admissible** (never overestimates)
   - Fewer nodes explored than uninformed searches (BFS/DFS)

### Performance Metrics
- **Nodes Explored**: Displayed after solving
- **Moves Required**: Optimal solution length
- **Search Time**: Real-time animation speed control

### Technical Stack
- **Language**: Python
- **GUI Framework**: Pygame
- **Data Structure**: Min-Heap (heapq) for priority queue
- **Complexity**: O(n^d) where n=branching factor, d=depth

---

## Project 2: Heart Disease Classification

### Overview
A **comprehensive machine learning classification pipeline** that predicts heart disease presence/absence using multiple algorithms. Includes exploratory data analysis, model training, hyperparameter tuning, and detailed performance evaluation.

### Dataset
- **Source**: UCI Machine Learning Repository (Cleveland Heart Disease)
- **Samples**: 303 patients
- **Features**: 13 medical indicators
- **Target**: Binary classification (0=No disease, 1=Disease present)

### Medical Features
| Feature | Description |
|---------|-------------|
| `age` | Patient age in years |
| `sex` | Gender (0=female, 1=male) |
| `cp` | Chest pain type (typical angina, atypical, etc.) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia type (3=normal, 6=fixed defect, 7=reversible) |

### Pipeline Architecture

#### **Phase 1: Data Preparation**
```
Raw Data → Missing Value Imputation → Feature Scaling → Train/Test Split
```

- **Missing Values**: Median imputation per column
- **Train/Test Split**: 80/20 stratified split (preserves class balance)
- **Normalization**: StandardScaler (zero mean, unit variance)
  - Essential for distance-based (KNN) and regularization-sensitive (logistic regression) models

#### **Phase 2: Model Training**
Four different classifiers trained and compared:

| Model | Type | Key Characteristics |
|-------|------|-------------------|
| **Logistic Regression** | Linear classifier | Fast, interpretable, baseline model |
| **Decision Tree** | Tree-based | Captures non-linear relationships, prone to overfitting |
| **K-Nearest Neighbors (KNN)** | Instance-based | No training phase, lazy learner, sensitive to feature scaling |
| **Random Forest** | Ensemble | Multiple decision trees, reduces overfitting, best performance |

#### **Phase 3: Hyperparameter Tuning**
Grid Search over Random Forest parameters:
```python
Parameters Tuned:
- n_estimators: [50, 100, 200]      # Number of trees
- max_depth: [None, 5, 10]          # Tree depth limits
- min_samples_split: [2, 5]         # Minimum samples to split node
- max_features: ['sqrt', 'log2']    # Features per split
```

**Strategy**: 5-fold cross-validation to find optimal combination

#### **Phase 4: Evaluation Metrics**

**Classification Metrics**:
- **Accuracy**: (TP + TN) / Total — Overall correctness
- **Precision**: TP / (TP + FP) — Of predicted positive, how many correct
- **Recall**: TP / (TP + FN) — Of actual positive, how many caught
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) — Harmonic mean
- **ROC-AUC**: Area under receiver operating characteristic curve (0.5-1.0 scale)

**Visualizations Generated**:
1. **Class Distribution**: Shows imbalance between disease/no-disease samples
2. **Correlation Heatmap**: Feature relationships and multicollinearity
3. **Confusion Matrices**: TP/TN/FP/FN breakdown per model
4. **ROC Curves**: Trade-off between TPR and FPR for each model
5. **Performance Comparison**: Bar chart across all metrics and models
6. **Feature Importance**: Top contributing features from Random Forest

### How It Works (Step-by-Step)

1. **Load & Explore**
   - Download dataset from UCI repository
   - Convert target to binary (1=any disease level, 0=healthy)
   - Analyze distributions and correlations

2. **Preprocess**
   - Handle missing values (median fill)
   - Normalize features for model fairness
   - Split data maintaining class distribution

3. **Train**
   - Each model trained on 80% of data
   - 5-fold cross-validation for robust performance estimation
   - Track accuracy, precision, recall, F1, ROC-AUC

4. **Tune Best Model**
   - Random Forest selected based on initial performance
   - Grid search tests 2×3×2×2 = 24 parameter combinations
   - Best combination applied to final model

5. **Evaluate & Interpret**
   - Detailed metrics comparison across all models
   - Visualizations to understand model behavior
   - Feature importance ranking
   - Final model achieves ~85%+ accuracy

### Key Results
- **Best Model**: Random Forest (after tuning)
- **Primary Metric**: ROC-AUC (handles class imbalance better than accuracy)
- **Feature Importance**: Top predictors identified for medical interpretation
- **Generalization**: Cross-validation ensures model not overfitted

### Technical Stack
- **Language**: Python (Jupyter Notebook)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (models, evaluation, tuning)
- **Visualization**: Matplotlib, Seaborn
- **Validation**: 5-fold cross-validation, train-test split

---

## Comparison & Integration

### Algorithmic Approaches
| Aspect | A* Puzzle | Heart Disease ML |
|--------|-----------|-----------------|
| **Problem Type** | Search optimization | Classification prediction |
| **Search Space** | Discrete puzzle states | Continuous feature space |
| **Goal** | Find path to target state | Estimate class probability |
| **Optimality** | Guarantees shortest path | Best effort (statistical) |
| **Uncertainty** | Deterministic | Probabilistic |

### Common Principles
1. **Heuristics**: Both use domain knowledge to guide decisions
   - Puzzle: Manhattan distance estimates remaining moves
   - ML: Feature patterns estimate disease likelihood

2. **Exploration**: Both explore solution spaces intelligently
   - Puzzle: A* explores state combinations
   - ML: Models explore feature-space patterns

3. **Validation**: Both measure effectiveness
   - Puzzle: Solution length, nodes explored
   - ML: Accuracy, precision, recall, F1, AUC

---

## How to Run

### A* Puzzle Solver
```bash
python astar_8puzzle.py
```
- Interactive GUI launches
- Click "Solve" to find solution with A* algorithm
- Use buttons to control visualization speed and heuristic

### Heart Disease Classification
```bash
jupyter notebook heart_disease_classification.ipynb
```
- Run cells sequentially to execute pipeline
- Generates plots and comparison tables
- See final metrics and feature importance

---

## Dependencies
- `pygame` — GUI for 8-puzzle visualization
- `numpy`, `pandas` — Numerical computing and data manipulation
- `scikit-learn` — Machine learning models and evaluation
- `matplotlib`, `seaborn` — Data visualization

---

## Conclusion

These projects demonstrate two fundamental AI concepts:

1. **Search-Based AI (A* Algorithm)**
   - Optimal pathfinding in discrete state spaces
   - Heuristic-guided exploration reduces computation
   - Real-time visualization of algorithm execution

2. **Statistical AI (Machine Learning)**
   - Pattern learning from data
   - Multi-model comparison and hyperparameter optimization
   - Evaluation across meaningful metrics

Together, they showcase problem-solving across different AI paradigms: deterministic optimization and probabilistic prediction.
