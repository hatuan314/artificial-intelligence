# GraphLabs — Pure Python Scripts (VSCode)

This package provides two runnable Python scripts for **Zachary's Karate Club** analysis:
- `src/network_analysis_lab.py`
- `src/link_prediction_lab.py`

## Quick Start
1. Open this folder in VSCode.
2. (optional) Create and activate a virtual environment:
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the scripts:
   ```bash
   python src/network_analysis_lab.py
   python src/link_prediction_lab.py
   ```

## Notes
- Both scripts default to the built-in NetworkX karate graph. You can switch to loading from `data/karate.edgelist` by uncommenting the relevant lines in the code.
- Plots are created with matplotlib. Close the figure windows to continue execution when prompted.
