![Still Count Logo](https://raw.githubusercontent.com/paulagsotres/Still_Count/master/Still_count_logo.png)
# Still Count: Immobility Behavior Analysis
Analyze rodent immobility from videos with a GUI-based tool compatible with a variety of experimental settings and with no training.
## Table of Contents
- [Installation](#installation)
- [How to use](#howto)
- [License](#license)
  
## Installation

Follow these steps to set up the Still Count immobility analysis tool:

---

### 1️⃣ Open a command prompt / terminal

- **Windows:** Press `Win + R`, type `cmd`, and hit Enter.  
- **Mac:** Open **Terminal** from Applications → Utilities.  
- **Linux:** Open your favorite terminal.

### 2️⃣ Clone the repository (or download the ZIP directly from Github)

In the command prompt or terminal, if you have Git installed, run:

```bash
git clone https://github.com/paulagsotres/Still_Count.git
```

---

### 3️⃣ (Optional but recommended) Create a new environment in using Conda

A virtual environment keeps the project dependencies separate from other Python projects.
```bash
conda create -n still_count
```
Activate it:
```bash
conda activate still_count
```
You should now see (still_count) in your conda environments – this means the environment is active.


---

### 4️⃣ Install dependencies

With the environment active, open the terminal and run:
``` bash
pip install -r requirements.txt
```

