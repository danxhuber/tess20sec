# MESA inlists

These inlists reproduce the best-fitting MESA models in the reference
sets for γ Pav, ζ Tuc and π Men that appear in Huber et al. (2021).

The inlists and `run_star_extras.f90` were developed for MESA r15140.
To run these inlists (assuming you have already set up MESA):

1. create a new `work` folder from the `astero` module: e.g. `cp -r $MESA_DIR/astero/work $HOME/tess20sec_mesa`
2. copy the contents of this folder to the new work folder: e.g. if you're in this folder, `cp -r * $HOME/tess20sec_mesa/`
3. build the work folder: `./mk`
4. edit `inlist_astero_search_controls` to select a star
5. run MESA: `./rn`

Optimal parameters are also included for the runs that excluded the
luminosity constraint.  In the full `inlist_astero_search_controls_star_name`,
you can run with these parameters by setting

    include_logL_in_chi2_spectro = .false. ! instead of .true.

and uncommenting the corresponding values of `first_FeH`, `first_Y`
and `first_mass` around line 440.
