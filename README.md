MSc Thesis: Modelling Climate Tipping Impacts of Short-Lived Climate Forcers
############################################################################

1/ Implementation:

-- MCTP-main.py computes the MCTPs using the time-integrated framework

-- MCTP-instantaneous.py computes the MCTPs using the instantaneous framework



2/ Results:

The MCTPs are provided in csv format. The files have names of the form {instantaneous/time-integrated}-{noozone/withozone}-{mean/all scenarios}-{mean/all years}.

For each framework (instantaneous or time integrated), the MCTPs are available with or without ozone adjustments, which gives four combinations (instantaneous - withozone, instantaneous - noozone, time-integrated - withozone, time-integrated - noozone).

For each of these four combinations, the MCTPs are available in four formats:

- all scenarios all years: scenario-specific and time-differentiated MCTPs (one value per forcer, per scenario, per time step)

- all scenarios mean years: scenario-specific MCTPs but averaged over time (one value per forcer, per scenario)

- mean scenarios all years: time-differentiated MCTPs but averaged across scenarios (one value per forcer, per time step)

- mean scenarios mean years: average across scenarios and over time of the MCTPs (one value per forcer)

Depending on the application, it is possible to choose one or the other of these MCTPs.
