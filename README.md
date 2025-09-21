![Still Count Logo](https://raw.githubusercontent.com/paulagsotres/Still_Count/master/Still_count_logo.png)
# Still Count: Immobility Behavior Analysis
Analyze rodent immobility from videos with a GUI-based tool compatible with a variety of experimental settings and with no training.
## Table of Contents
- [Installation](#installation)
- [How to use](#howto)
- [License](#license)
  
## Installation

This software does not require GPU to run. Follow these steps to set up the Still Count immobility analysis tool:


### 1️⃣ Open a command prompt / terminal in the folder you want to store Still Count

- **Windows:** Press `Win + R`, type `cmd`, and hit Enter.  
- **Mac:** Open **Terminal** from Applications → Utilities.  
- **Linux:** Open your favorite terminal.

### 2️⃣ Clone the repository (or download the ZIP directly from Github)

In the command prompt or terminal, if you have Git installed, run:

```bash
git clone https://github.com/paulagsotres/Still_Count.git
```


### 3️⃣ (Optional but recommended) Create a new environment in using Conda

A virtual environment keeps the project dependencies separate from other Python projects. In your terminal type:
```bash
conda create -n still_count
```
Activate it:
```bash
conda activate still_count
```
You should now see (still_count) in your conda environments – this means the environment is active.



### 4️⃣ Install dependencies

With the environment active, open the terminal and run:
``` bash
cd folder\Still_Count (path to your Still Count folder)
pip install -r requirements.txt
```

### 5️⃣ Open GUI

Run the GUI by clicking "launch_stillcount.bat" file. This file can be used to open the GUI directly from now on. 

![GUI preview](https://github.com/paulagsotres/Resources/blob/main/screenshot1.png?raw=true)

----
## How to use

Watch the tutorial video here:  

[![Still Count Tutorial](https://img.youtube.com/vi/mXGWnG6s_rs/0.jpg)](https://youtu.be/mXGWnG6s_rs)

Videos that can get processed preferably should not have big ambient light changes (for example if the light gets stronger or dimmer throughout the video). The parameter of analysis should be choosen manually by the users depending of the experimental set-up and the protocol. There are 4 presets included that can give you an idea of where to start. 

**TIP!** Plot the binary size analysis in one video to visualize possible periods of freezing and THEN, choose your Immobility Event threshold. Save your working configurations to not loose them!

## Licence
https://doi.org/10.5281/zenodo.17171323

**Any bugs or errors, feel free to email me at: paulagomezsotres@gmail.com**
