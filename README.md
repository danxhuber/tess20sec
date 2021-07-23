# TESS 20-second cadence data

Data and scripts to reproduce results for the paper "A 20-second Cadence View of Solar-Type Stars and Their Planets with TESS: Asteroseismology of Solar Analogs and a Re-characterization of pi Men c" (Huber et al. 2021, in prep).

sample/: Photometric precision of 20-second cadence data (Section 2)

SYDSAP/: Custom light curves (Section 3.1)

seismo/: Asteroseismic analysis of solar analogs (Section 3.2-3.5)

transit/: Joint transit and RV fit of pi Men c (Section 4). 

See individual directories for scripts to reproduce figures. Note that Figure 10-12 requires the trace output of the joint transit+RV fit from exoplanet, which can be can be recreated by running transit/fittransitrv.py.
