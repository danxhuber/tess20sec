# TESS 20-second cadence data

Data and scripts to reproduce results for the paper "A 20-second Cadence View of Solar-Type Stars and Their Planets with TESS: Asteroseismology of Solar Analogs and a Re-characterization of pi Men c" ([Huber et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210809109H/abstract)).

sample/: Photometric precision of 20-second cadence data (Section 2)

SYDSAP/: Custom light curves (Section 3.1)

seismo/: Asteroseismic analysis of solar analogs (Section 3.2-3.5)

transit/: Joint transit and RV fit of pi Men c (Section 4). 

Notes:

* Scripts to reproduce Figures 10-12 require the trace output of the joint transit+RV fit from [exoplanet](https://docs.exoplanet.codes/en/latest/), which can be can be recreated by running transit/fittransitrv.py
* Radial velocities from ESPRESSO and CORALIE in transit/data/rvs/ are from [Damasso et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...642A..31D/abstract), please cite the paper when using the data.
