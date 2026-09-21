<<<<<<< HEAD
# In-gel fluorescence densitometry

A small desktop tool for ImageJ-style lane analysis of grayscale TIFF images.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

## Workflow

1. Open a grayscale TIFF.
2. Drag on the image to define the horizontal lane ROI. The profile is the mean grey value across that ROI for each x coordinate, matching ImageJ.
3. Draw a background baseline by clicking two points on the profile. The linear function `B(x)` is interpolated between them.
4. Click and drag across each peak to add an integration interval.
5. Click **Export CSV**. Each row contains the peak bounds, raw integrated density, background integrated density, and background-corrected total integrated density based on the mean-grey-value profile.

Use **Reset** to start a new measurement. TIFF stacks are reduced to their first page.
=======
# FluoroGelAnalyzer
Perform densitometry for in-gel fluorescence, background subtraction to obtain corrected AUC for quantification of gel badn species. 
>>>>>>> dc30e154a73c05b8f917c822c8b66403442d3467
