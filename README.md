# Which method catches a sick biogas digester first?

[![tests](https://github.com/kwwaleszczynski-del/biogas-digester-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/kwwaleszczynski-del/biogas-digester-twin/actions/workflows/ci.yml)

**Interactive dashboard - replay any upset with its sensors and every method's alarms, method comparison,
confusion matrices, real farm data:
[kwwaleszczynski-del.github.io/biogas-digester-twin](https://kwwaleszczynski-del.github.io/biogas-digester-twin/)**

An agricultural biogas plant fails quietly. Volatile fatty acids build up for days or weeks before the gas
yield drops, and by the time the CHP output falls the process may need weeks to recover. The usual
safeguard is the FOS/TAC titration, done once or twice a week. This project asks **which monitoring
methods detect and classify digester upsets earliest and with the fewest false alarms, before the gas
yield drops**, and whether they can tell a process upset from a faulty instrument.

Open operating data from full-scale biogas plants with CHP are practically unavailable, and the few open
sets carry no record of what went wrong when. So the core is a **simulator with ground truth**: the IWA
Anaerobic Digestion Model No. 1 in its BSM2 implementation, verified against the published BSM2 steady state,
set up as a maize silage and cattle slurry plant of about 0.5 MWel. Seven kinds of upset are injected with
known start and known yield loss, the plant is watched through realistic sensors, and eight methods are
compared on test runs none of them saw while being tuned, each against what its own false alarms would
"detect" by chance. Open data from four small farm-scale digesters
serve for the one thing a simulator cannot provide: false alarms and plausibility on real measurements.

It is the biogas counterpart of the outage studies in this portfolio
([PV outage classification](https://github.com/kwwaleszczynski-del/pv-outage-classifier),
[wind turbine downtime and availability](https://github.com/kwwaleszczynski-del/wind-downtime-availability));
the sibling project [pl-biogas-flex](https://github.com/kwwaleszczynski-del/pl-biogas-flex) looks at the same kind of plant from the power market
side.

## Key results

All numbers are measured on the test set: 140 upsets (20 per type, each run with its own feed history)
and three normal years that no method saw while it was fitted or tuned (protocol in
[Validation](#validation)). One event rule counts both sides: an **alarm onset** is a day in alarm after a
day without. An upset is detected if an onset falls **within 14 days** of the cause; **chance** is how often
the same rule fires in a 14-day window of normal operation, where every onset is a coincidence; **excess**
is the difference. Brackets: 95 % bootstrap intervals over test runs.

**1. Scored against chance, the ranking changes.** A detector that is in alarm a third of the time
"detects" a third of all upsets by coincidence. The first version of this study counted any alarm start
within 60 days of the cause, without a chance level; there the CUSUM looked perfect (100 %), and EWMA and
the expert rules nearly so. Within 14 days and against chance:

| method (lab 10 % missing) | process upsets alarmed within 14 d | chance | excess | sensor faults | median delay [d] | false alarms per year | days in alarm | ammonia: alarmed before loss (chance) | ammonia: median lead [d] | named right |
|---|---|---|---|---|---|---|---|---|---|---|
| FOS/TAC > 0.3 | 27 % (21-33 %) | 0 % | +27 % | 0 % | 4 | 0.0 (0.0-0.1) | 0 % | 0 % (0 %) | - |  |
| FOS/TAC +30 % (plant-specific trend) | 48 % (44-53 %) | 9 % | +39 % | 0 % | 5 | 2.1 (1.7-2.6) | 6 % | 5 % (8 %) | 2 |  |
| EWMA on specific methane yield | 89 % (83-94 %) | 29 % | +60 % | 23 % | 5 | 7.8 (7.0-8.5) | 39 % | 90 % (26 %) | 3 |  |
| CUSUM on specific methane yield | 29 % (20-38 %) | 14 % | +15 % | 13 % | 5 | 3.5 (2.8-4.2) | 76 % | 35 % (12 %) | 5 |  |
| PCA T2/SPE on online signals | 54 % (50-59 %) | 2 % | +52 % | 8 % | 2 | 0.6 (0.2-1.0) | 0.2 % | 0 % (2 %) | - |  |
| Expert rules | 82 % (75-89 %) | 29 % | +53 % | 55 % | 2 | 8.0 (7.1-8.8) | 42 % | 90 % (26 %) | 6 | 95 % |
| Gradient boosting | **92 %** (86-97 %) | 6 % | **+86 %** | **88 %** | 2 | 2.1 (1.4-2.8) | 2 % | **100 %** (6 %) | **7** | **98 %** |
| EKF twin | 89 % (83-95 %) | 15 % | +74 % | 68 % | 3 | 4.1 (3.4-5.0) | 10 % | 55 % (13 %) | 2 | 93 % |

- **The yield charts are mostly alarm habit.** The specific methane yield wanders because the silage dry
  matter the operator books drifts between clamps. The EWMA is in alarm on 39 % of normal days and gets 29 %
  of the upsets by chance. The CUSUM, not restarted after a signal, sits in alarm on 76 % of normal days, so
  an upset rarely starts a new alarm: 29 % detected against 14 % chance (restarted after every signal, as in
  the first version, it raised a new alarm about every ten days instead). The expert rules inherit the
  EWMA's habit (42 % of days in alarm, 8 false alarms a year).
- **The lab routine is quiet and narrow.** FOS/TAC > 0.3 never fires by chance but sees only shock loads
  and overloads (27 % of process upsets within 14 days); the plant-specific trend rule sees 48 % at 2.1
  false alarms a year. Neither sees a heating failure, a poorer silage or an instrument, and neither warns
  ahead of an ammonia loss (0 and 1 of 20).
- **PCA is the quietest online method** (0.6 false alarms a year, in alarm on 0.2 % of normal days) and
  catches heating failures, shock loads and overloads (median 1-8 days), but not ammonia or a poorer silage
  within two weeks (1 and 0 of 20; it sees 84 % of all process upsets within 60 days).

**2. At the same false-alarm rate.** Moving each detector's threshold until it raises at most 2 false
alarms a year (read off its curve on the test runs - a fair comparison of separation, not a threshold a
plant could set in advance):

| method | alarmed within 14 d | chance | excess | sensor faults | false alarms per year | days in alarm |
|---|---|---|---|---|---|---|
| Gradient boosting | 94 % | 3 % | +91 % | 88 % | 0.8 | 1 % |
| Expert rules | 79 % | 8 % | +71 % | 50 % | 1.8 | 3 % |
| EKF twin | 76 % | 6 % | +70 % | 65 % | 1.9 | 2 % |
| PCA T2/SPE | 74 % | 7 % | +67 % | 28 % | 1.9 | 2 % |
| EWMA yield | 66 % | 9 % | +57 % | 2 % | 2.0 | 4 % |
| FOS/TAC +30 % | 45 % | 7 % | +38 % | 0 % | 1.6 | 4 % |
| FOS/TAC > 0.3 | 42 % | 7 % | +35 % | 0 % | 1.6 | 5 % |
| CUSUM yield | 23 % | 8 % | +15 % | 5 % | 1.9 | 22 % |

Held to this rate, the expert rules and the EWMA lose their alarm habit and keep or raise their excess;
the CUSUM does not recover. The absolute lab rule reaches 42 % only with its limit lowered to 0.23.

![trade-off](docs/img/tradeoff.png)

**3. Early warning pays for one upset type: ammonia.** Ammonia inhibition from an N-rich co-substrate builds
up for a median 12.5 days before the 7-day methane falls 5 % short of the healthy twin, and costs a median
17 % of the methane over the following three months. Gradient boosting alarms before the loss in all 20
cases (chance 6 %), a median 7 days ahead; the expert rules and the EWMA in 18 of 20, but their chance is
26 %; the EKF twin in 11 of 20 (chance 13 %), 2 days ahead; FOS/TAC and PCA not at all. The other process
upsets have no loss to warn of in the same sense:

- a **poorer silage** loses yield from the first day it is fed (the "loss" 4-5 days later is the 7-day mean
  catching up) and costs a median 19 % of the methane, so the useful number is how fast it is found;
- **overload, shock load and heating failure are mild in this plant**: they dip the 7-day methane for a few
  days (median lowest 95 %, 94 % and 91 % of the healthy twin) and cost at most 1.5 % of the methane over 90
  days (medians below 0.1 %, below 0.1 % and 0.6 %); half the overloads never reach the 5 % line. They are
  the textbook upsets that stress the process (shock loads and overloads raise the acids to a median peak
  of 3.6 and 2.7 g/L) and they test detection, not prevention of loss. A heating failure shows on the temperature probe, and every method that reads it alarms within 1-2
  days.

![lead time](docs/img/lead_time.png)

**4. Gradient boosting does not need the feed log, but it needs examples.** The harmless recipe changes of
normal operation now overlap the upsets in size, and gradient boosting without any feed-log feature does as
well as with it (96 % vs 92 % within 14 days, 98 % named right both), so its result does not come from
reading the upset off the feed sizes. It does come from having seen every upset type in training:
retrained without one type and tested on it, it misses **all overloads and all poorer silages, 18 of 20
CH4 analyser drifts and 19 of 20 stuck flow meters**, and names the ammonia and heating upsets it still
notices a "substrate change" (16 of 18 and 16 of 17). A real plant has no labelled history of its rare
failures.

**5. The EKF twin needs no examples of failures and separates instrument from process,** with 99 % of its
detections on the right side of the process/sensor line and 93 % named right, but it knows the plant too
well here: its model is lumped from the true plant. With nine of its parameters off by about 20 % and
re-zeroed on each run's first weeks, its VFA estimate crosses its limit far more often: 11 false alarms a
year instead of 4.1, and an excess of +51 % (42-59 %) instead of +74 % (68-80 %). A filter for a real plant
needs its model calibrated to that plant.

![confusion](docs/img/confusion.png)

**6. Without the lab.** Losing half or all of the FOS/TAC samples leaves the online methods and gradient
boosting unchanged (95 % within 14 days with no lab at all), costs the EKF twin naming accuracy (93 % ->
78 % named right with no lab), and leaves the lab rules blind.

![robustness](docs/img/robustness.png)

**7. On real farm data the yield charts are not usable as they are.** On four small manure-fed digesters
(below) the EWMA on gas per kg manure, calibrated on each digester's own first eight weeks, is in alarm on
37-85 % of the days; the CUSUM goes into alarm once or twice and stays there (83-99 % of the days). Gas
output there is dominated by how the digester is operated, the same failure the simulation shows. The one
rule that transfers is the data-quality check: 11 of the 12 alarm onsets of the frozen-or-backwards meter
rule on the two digesters with an operator log start within three days of a note about a repair, a closed
valve or the meter.

An example: the ammonia upset of test run 101 - true methane against the healthy twin, the sensors, the
FOS/TAC samples and every method's alarms:

![replay](docs/img/replay_ammonia.png)

## The twin

### ADM1 with the BSM2 implementation details, verified

`src/biogas_twin/adm1.py` is an own implementation of ADM1 (Batstone et al. 2002) with the implementation
choices of the Benchmark Simulation Model No. 2 (Rosen & Jeppsson 2006): 26 liquid states, three gas-phase
states, Hill-type pH inhibition, the acid-base system solved algebraically (charge balance by Newton-Raphson
at every call, the "DAE" variant), the temperature-dependent constants and the gas outlet
$`q_{gas} = k_p (P_{gas} - P_{atm})`$. Integrated with constant BSM2 influent from a state far away (biomass
halved, solubles doubled), it settles on the published BSM2 steady state within a **relative deviation of
2e-7** for every state, the pH, the ions and the gas flow (a test keeps it below 1e-5):

| BSM2 steady state | published | this implementation |
|---|---|---|
| pH | 7.46554 | 7.46554 |
| acetate S_ac [kg COD/m3] | 0.197630 | 0.197630 |
| inorganic nitrogen S_IN [kmol/m3] | 0.130230 | 0.130230 |
| methane in the gas phase [kg COD/m3] | 1.625607 | 1.625607 |
| gas flow at atmospheric pressure [m3/d] | 2955.70 | 2955.70 |

Two additions are switched off in the BSM2 set and on for the agricultural plant: kinetic rates scale as
$`k(T) = k(T_{ref})\, e^{\theta (T - T_{ref})}`$, and the free-ammonia inhibition of acetoclastic
methanogens keeps ADM1's form $`I = 1/(1 + S_{NH_3}/K_I)`$ with an optional Hill exponent (left at 1).

### The plant (assumptions)

| | value | why |
|---|---|---|
| digester | one CSTR, 3000 m3 liquid, 600 m3 gas, 40 °C | main digester of a typical 0.5 MWel maize/slurry plant |
| feed | 26 t/d maize silage + 30 t/d cattle slurry, once a day | HRT 54 d, OLR 3.45 kg oDM/(m3 d) |
| hydrolysis | first order, 0.25 1/d | rate-limiting step for silage |
| uptake of propionate, acetate | 1.2 and 4.5 1/d (BSM2: 13 and 8) | BSM2 kinetics are for sewage sludge; lowered so that the steady-state acids (about 0.7 g/L acetic, 0.12 g/L propionic acid) and the room before overload match agricultural plants |
| temperature factor θ | 0.035 1/K hydrolysis, acidogenesis, decay; 0.069 1/K acetogenesis, methanogenesis | methanogenic rates halve per 10 K below the set point |
| ammonia inhibition | ADM1 default, $`K_I`$ = 0.0018 kmol N/m3 | gives the "inhibited steady state" with little spare methanogenic capacity that ammonia-rich agricultural plants run at |

The resulting plant: pH 7.6, 53.5 % CH4, 6,000 Nm3 biogas per day (3,220 Nm3 CH4, about 540 kWel at 40 %
electrical efficiency), 0.31 Nm3 CH4 per kg oDM fed, TAN 2.4 g/L, FOS 1,840 mg/L, TAC 11,000 mg CaCO3/L,
FOS/TAC 0.17.

### Substrates (assumptions, typical values)

Weender-type composition per kg dry matter, chosen inside the ranges given in the FNR *Leitfaden Biogas*
and the KTBL standard yields; mapped to ADM1 inputs directly as carbohydrates, proteins, lipids, sugars and
acids (common practice for agricultural ADM1 applications, e.g. Lübken et al. 2007, Koch et al. 2010), with
a substrate-specific anaerobic degradability splitting off the inert part.

| per kg DM | maize silage | cattle slurry | poultry manure (N-rich, ammonia scenario) | sugar beet (shock load) |
|---|---|---|---|---|
| dry matter [% FM] | 33 | 9 | 45 | 23 |
| organic DM [% DM] | 95.5 | 80 | 72 | 92 |
| crude protein | 0.085 | 0.15 | 0.26 | 0.05 |
| crude fat | 0.032 | 0.035 | 0.025 | 0.005 |
| free sugars | 0.010 | - | - | 0.68 |
| silage acids / VFA | 0.065 | 0.045 | 0.02 | - |
| anaerobic degradability | 0.90 | 0.50 | 0.65 | 0.95 |
| ammonium N [kg/t FM] | 0.4 | 2.6 | 6.0 | 0.1 |

COD factors: carbohydrate 1.185, protein 1.50, lipid 2.90, organic acids 1.067, sucrose 1.123 kg COD/kg.
Fed alone to the plant's digester (HRT 54 days), the maize silage gives 0.36 Nm3 CH4 per kg oDM (KTBL:
0.355 for maize silage with good silage management) and the cattle slurry 0.20; a test keeps the mix
between the two.

### What the operator sees

| signal | how | imperfections |
|---|---|---|
| biogas flow | hourly, Nm3/h dry | 2 % noise, a fixed calibration error of ±2 %, 1 % dropouts, outages of 2-36 h (about 2 per quarter) |
| CH4 content | hourly | 0.3 %-point noise, slow drift re-zeroed by a monthly calibration |
| pH | hourly, in the digester | 0.02 noise, electrode drift recalibrated monthly |
| temperature | hourly | 0.1 K noise |
| FOS/TAC | lab, Monday and Thursday | titration volumes ±0.1 mL (A) and ±0.05 mL (B); 10 % of samples missing |
| ammonium N | lab, Monday | ±5 % |
| feed log | daily, t per substrate | ±2 % weighing; oDM booked with the nominal silage dry matter (true DM drifts between clamps, 31-35 %) |

FOS/TAC is not a random number added to the VFA: the Nordmann titration is simulated on the ADM1 acid-base
state of the sample (20 mL titrated with 0.1 N H2SO4 to pH 5.0, volume A, and on to pH 4.4, volume B):

```math
\mathrm{TAC} = 250\,A \ [\mathrm{mg\ CaCO_3/L}], \qquad
\mathrm{FOS} = (1.66\,B - 0.15)\cdot 500 \ [\mathrm{mg\ HAc/L}]
```

so FOS carries its real interference from bicarbonate and ammonia, and TAC moves with the ammonium.

### Upsets and ground truth

| upset | injected | severity drawn per run |
|---|---|---|
| organic overload | more maize silage, ramped in over 1-7 days, kept | x1.4 to x2.0 |
| acidification (shock load) | sugar beet on top of the recipe for 2-4 days | 40-90 t/d |
| ammonia inhibition | poultry manure replaces part of the silage (same oDM), ramped over 1-14 days | 40-90 % of the silage oDM |
| heating failure | temperature falls for 5-14 days, then recovers at 0.5 K/d | 0.5-1.0 K/d |
| substrate change | a new silage of lower degradability, kept | x0.65 to x0.88 |
| CH4 analyser drift | linear drift, until the end | ±0.08 to ±0.3 %-points per day |
| stuck gas flow meter | the hourly reading freezes | 7-44 days |

The ground truth of each run is its class, the start $`t_0`$ and the **start of yield loss**: the first day on
which the true methane production, averaged over the trailing 7 days, falls more than 5 % below what a
**healthy twin** produces from the same feed. The healthy twin is the same ADM1 plant without the upset
mechanism - no inhibition terms, the normal temperature, the nominal silage quality - so it is the methane
potential of what was actually fed. With $`r(t) = Q_{CH_4}(t) / Q_{CH_4}^{healthy}(t)`$ normalised by its mean
over the 14 days before $`t_0`$:

```math
t_{loss} = \min \Big\{ t \ge t_0 : \tfrac{1}{7} \sum_{s=t-6}^{t} r(s) < 0.95 \Big\}
```

Sensor faults never cause yield loss; the methods must notice them without calling them a process upset.

**Harmless recipe changes are part of normal operation.** Every 6-17 weeks the operator changes the
recipe: 5-45 % of the silage oDM as poultry manure for 3 days to 4 weeks (the larger the share, the
shorter: share times days at most 3.2), 5-50 t/d of sugar beet for 1-2 days, or 8-50 % more silage for 2-6
weeks. Their sizes overlap the lower end of the upsets (40-90 % poultry manure, 40-90 t/d beet, silage x1.4
to x2.0), so neither a person nor a model can tell an upset from a harmless change by the size of the feed
change alone; what differs is how long it lasts and what the process does. None of them crosses the
yield-loss line: against an uninhibited twin fed the same, the 7-day methane of the five normal years
never falls below 95.9 % (a poultry share of 45 % for 10 days would cross it, hence the share-times-days
cap). Without such changes every poultry manure
entry in the feed log would mean "ammonia upset coming", and a method could name the upset from the feed
log before the process shows anything - which is how a first version of this study looked. A second version
added small changes (5-15 % poultry manure, 5-15 t/d beet, +8-20 % silage); a review found that ranges
disjoint from the upsets still let a model separate them by size, so the ranges now overlap, and gradient
boosting is also run without any feed-log feature to measure how much it leans on the log.

## Methods

| method | uses | alarm | names the cause |
|---|---|---|---|
| **FOS/TAC > 0.3** | lab | latest result above 0.3 (a common practice threshold) | no |
| **FOS/TAC +30 %** | lab | latest result 30 % above the median of the 8 before (plant-specific trend) | no |
| **EWMA yield** | gas, CH4, feed log | EWMA of the standardised 7-day specific methane yield outside ±L sigma | no |
| **CUSUM yield** | gas, CH4, feed log | two-sided tabular CUSUM of the same, not restarted after a signal | no |
| **PCA T2/SPE** | gas, CH4, pH, temperature, feed log | Hotelling T2 or SPE above the 99.5 % training quantile on 2 days | no |
| **Expert rules** | everything | the morning checks of an operator, top to bottom | yes |
| **Gradient boosting** | everything | P(normal) < 0.5 on 2 days | yes |
| **EKF twin** | everything | estimated VFA, methanogenic activity, substrate quality, analyser bias | yes |

**Specific methane yield** (Nm3 CH4 per kg oDM fed) from the measured gas, CH4 content and the feed log:
$`\mathrm{SMY}_t = \sum_{s=t-6}^{t} Q_s\, c_s \,/\, \sum_{s=t-6}^{t} m^{oDM}_s`$. Standardised with the mean and
standard deviation of two normal training years, $`z_t`$; the EWMA $`e_t = \lambda z_t + (1-\lambda) e_{t-1}`$
($`\lambda = 0.2`$) alarms outside $`\pm L \sqrt{\lambda / (2-\lambda)}`$ with $`L = 3`$; the CUSUM sums
$`C^{\pm}_t = \max(0, C^{\pm}_{t-1} \pm z_t - k)`$ with $`k = 0.5`$ and alarms above $`h = 5`$. It is not
restarted after a signal, so a lasting shift is one alarm (a restart after every signal, as in the first
version, raised a new alarm every few days of the same shift).

Every detector also has a continuous **score** scaled so that its published threshold is 1 (for example
$`|e_t|`$ over its limit, $`\max(C^+_t, C^-_t)/h`$, the larger of T2 and SPE over their limits,
$`(1 - P_{normal})/0.5`$ for gradient boosting); moving the threshold on the score traces each method's
false-alarm/detection curve, which is how all detectors are compared at one common false-alarm rate.

**PCA** on eight standardised daily variables (specific yield, gas per oDM, CH4, pH, temperature and the
7-day changes of CH4, pH and yield), components up to 90 % of the variance, fitted on the normal training
years (MacGregor & Kourti 1995):

```math
T^2 = \sum_{i=1}^{k} \frac{t_i^2}{\lambda_i}, \qquad \mathrm{SPE} = \lVert z - P P^{\top} z \rVert^2
```

**Expert rules**, checked top to bottom: frozen hourly flow (zero spread over a day) -> stuck meter;
temperature 1.5 K below the set point -> heating; otherwise an upset is flagged by the EWMA, the relative
FOS/TAC rule or a pH drop of 0.08, and named from the feed log (sugar beet -> shock load, poultry manure or
TAN above 3.2 g/L -> ammonia, oDM 20 % above the last month -> overload), else a CH4 jump of 2 %-points with
steady pH and FOS/TAC -> analyser drift, else a yield drop without VFA signs -> substrate.

**Gradient boosting** (`HistGradientBoostingClassifier`, balanced classes) on 60 daily features: levels,
3/7/14-day changes, 7-day spread and deviation from the last month of the yield, gas, CH4, pH and temperature;
the spread of the hourly flow; the latest FOS, TAC, FOS/TAC, TAN, their age and trend; the feed composition
of the last 7 and 28 days. Trained on 2 normal years and 8 runs of every upset type, with fixed iterations
(no early stopping) and one thread, so a refit is bit-identical. A check variant drops the six feed-log
features (what and how much was fed, the oDM fed relative to last month) and keeps the rest.

**EKF twin**: an extended Kalman filter on a reduced model of AM2 type (Bernard et al. 2001) with a hydrolysis
step, whose parameters come from lumping the ADM1 plant at its nominal steady state (no fitting to any upset):

```math
\begin{aligned}
\dot X_T &= D(s\,X_{T,in} - X_T) - k_h f_h(T) X_T + r_{dec} X_2 \\
\dot S_2 &= D(S_{2,in} - S_2) + (1-f_{H_2})(1-Y_1)\big(k_h f_h(T) X_T + D S_{1,in}\big) - r_2 \\
\dot X_2 &= Y_2\, r_2 - (D + k_d) X_2, \qquad r_2 = e^{m}\, k_2 f_m(T) \frac{S_2}{K_2 + S_2} X_2\, I_{pH} \\
\dot C &= D(C_{in} - C) + c_1 r_{ac} + c_2 r_2 - q_{CO_2}/V, \qquad \dot Z = D(Z_{in} - Z) + n_N k_h f_h(T) X_T
\end{aligned}
```

with $`X_T`$ particulate degradable COD, $`S_2`$ VFA, $`X_2`$ methanogens, $`C`$ inorganic carbon and $`Z`$
alkalinity; pH from the charge balance, CO2 to the gas by transfer, CH4 from the H2 and acetate routes. The
filter also estimates three slowly varying unknowns: $`m`$, the log activity of methanogenesis (inhibition by
ammonia, pH or overload pushes it down), $`s`$, the degradability of the feed relative to the recipe, and
$`b`$ with its drift rate, the bias of the CH4 analyser. Measurements: daily biogas, CH4 and pH; FOS and TAC on
lab days through the same titration model. Alarms: VFA estimate above 1.6 times nominal or $`m < -0.3`$;
$`m < -0.15`$ without VFA or $`s < 0.93`$ (substrate); $`|b| > 1.5`$ %-points (analyser); frozen flow; low
temperature. The cause of a VFA alarm is named from the feed log as in the rules.

The filter knows the plant better than it would in practice: its parameters are lumped from the true
plant, it uses the true plant's titration model and the nominal substrate composition, so its only model
error is structural (the reduced model). A check variant scales nine of its rate, yield and transfer
parameters by random factors $`e^{N(0,\,0.2^2)}`$ (about ±20 %) and re-zeroes $`m`$, $`s`$, $`b`$ and the
nominal VFA level on days 14-35 of each run, as one would commission a filter on a new plant.

## Validation

- **Training**: 2 normal years (seeds 1001-1002) and 8 runs of every upset type (seeds 0-7): EWMA, CUSUM
  and PCA in-control statistics and the gradient boosting model. The rule and EKF thresholds are set by
  hand; the code was debugged on a smoke run that included one run per type with seed 100, so seed 100 is
  not in the test set.
- **Test**: 3 normal years (seeds 2001-2003) and 20 runs of every upset type (seeds 101-120), 140 upsets.
  Every run has its own feed history (feed and silage variation, harmless recipe changes) and its own
  sensor noise; it starts from the steady state, runs 90-96 days of normal operation and 90 days after the
  upset. Runs are therefore independent, and intervals come from a bootstrap over runs (1000 resamples
  within each upset type).
- **One event rule**: an **alarm onset** is a day in alarm after a day without. **Detected**: an onset
  within 14 days from $`t_0`$ (an alarm already running at $`t_0`$ does not count). **False alarms**: onsets
  per year on upset-free days after a 35-day warm-up, i.e. the normal years and the history before $`t_0`$
  of every test run; **days in alarm**: the share of those days a method is in alarm.
- **Chance**: the same detection rule applied at every possible fake $`t_0`$ on the upset-free days - the
  share of 14-day windows of normal operation that contain an onset. A method in alarm half the time
  "detects" many upsets by coincidence; **excess = detected - chance** is what it finds beyond its own alarm
  habit.
- **Common false-alarm rate**: each score threshold moved until the method raises at most 2 false alarms a
  year on the test runs, read off its curve (a fair comparison of separation, not a threshold a plant could
  set in advance).
- **Lead time** $`t_{loss} - t_{alarm}`$ only where the upset costs a **material** share of methane - at
  least 2 % of the healthy twin's production over the 90 days after $`t_0`$ - and the loss builds up after
  the cause: the ammonia upsets. Chance for "alarmed before the loss" uses windows of the same length.
- **Classification**: the class named in the first three days of the first alarm, against the truth.
- **Missing lab**: every test run observed again with the same online data and 50 % and 100 % of the lab
  samples removed.
- **Unseen upset**: gradient boosting retrained seven times, each time without one upset type, tested on
  that type.

## Real farm data

The Mendeley set (D. Mulat, ILRI, 2025) holds a year of daily records from four small farm-scale
digesters fed with 20 kg of cattle manure and 30 kg of water a day (two flexible bags, two fixed domes):
the gas meter reading, feed, air and digester temperature (flexible bags only) and operator comments. The
pH, VFA, TAN and FOS/TAC columns are empty and CH4 was measured once, so only the gas-based detectors and
the data-quality rule apply. They are calibrated on each digester's first eight weeks after start-up, as
one would commission them on a new plant.

| digester | EWMA: alarm onsets / year (days in alarm) | CUSUM: alarm onsets / year (days in alarm) | meter rule: alarm onsets (near an operator note) |
|---|---|---|---|
| R1 flexible bag | 8.2 (58 %) | 1.2 (95 %) | 4 (4) |
| R2 flexible bag | 5.8 (85 %) | 1.2 (99 %) | 8 (7) |
| R3 fixed dome | 5.1 (81 %) | 1.3 (91 %) | 4 (no log) |
| R4 fixed dome | 6.4 (37 %) | 2.6 (83 %) | 4 (no log) |

Plausibility: the twin assumes methanogenic rates fall by half per 10 K. In these unheated digesters, at
18-27 °C, the weekly gas per kg manure does not rise with temperature: the fitted sensitivity is
-0.10 1/K (95 % interval -0.17 to -0.03) for R1 and -0.05 (-0.11 to 0.01) for R2, with weeks near operator
notes left out. Operation (gas use, valve handling, start-up after repairs) swamps the temperature effect,
so the data can neither confirm nor refute the assumption.

![real data](docs/img/real_data.png)

## Limitations

- **The plant is a model.** ADM1 with plausible agricultural parameters is a well-established description,
  but the kinetics, the inhibition constants and the substrate characterisation are assumptions, not a
  calibration to one real plant. The ranking of methods should transfer; the exact days and rates will not.
- **One digester, one recipe.** Real plants have a main and a secondary digester, mixing problems, foaming,
  floating layers, trace-element deficiencies, sulphide and desulphurisation - none of which is simulated.
- **The upsets are the textbook ones**, each alone; real incidents overlap (a new silage clamp in winter
  with a cold snap). Three of the five process upsets (overload, shock load, heating failure) are mild in
  this plant and cost under 1.5 % of the methane; only ammonia and a poorer silage cost a material share.
- **The feed log helps the naming.** Rules and the EKF name overload, shock load and ammonia partly from
  what was fed. That is realistic - the operator knows the feed - but it means the classification of those
  upsets is easier than detecting them. Gradient boosting detects and names as well without the feed log.
- **Gradient boosting has seen every upset type in training**, with severities from the same ranges; its
  numbers are an optimistic bound. The unseen-type experiment shows what remains without that help.
- **The EKF twin knows the plant** (parameters lumped from the true plant, its titration model, the
  nominal substrate); the perturbed variant shows the cost of a 20 % model error.
- **The common-rate comparison reads thresholds off the test curves**: it compares how well the methods
  separate upsets from normal operation; a plant setting a threshold in advance would land near, not on,
  those points.
- **The real data cannot validate detection.** The open farm-scale set has no pH, VFA or FOS/TAC and no
  documented process upsets; it checks false alarms and plausibility only.
- **Lab error is idealised** as noise on the titration volumes; real FOS/TAC also depends on sample
  handling, degassing and the analyst.

## Possible version 2 (not built)

Demand-driven feeding for a flexible CHP: the sibling project
[pl-biogas-flex](https://github.com/kwwaleszczynski-del/pl-biogas-flex) keeps the gas production constant and
shifts only the power output, with extra engine capacity and gas storage. When a plant also shifts its gas
production towards the high-price hours, the feed varies day to day by design. Every yield-based detector here
assumes a steady recipe; with flexible feeding the expected gas must come from a model of the feed response, which
is exactly what the EKF twin provides. A v2 would derive feeding schedules from that project's dispatch, replay them
through this twin and check which methods keep their false-alarm rate.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                                    # 15 tests, offline, about a minute
python -m biogas_twin verify                 # ADM1 against the BSM2 steady state
python -m biogas_twin fetch                  # the Mendeley farm data, 12 MB, CC BY 4.0, checksum verified
python -m biogas_twin study --workers 6      # 201 simulations + 429 evaluations, about an hour
python -m biogas_twin real                   # false alarms on the farm data
python -m biogas_twin report                 # results/, docs/img, docs/data.js
```

```
src/biogas_twin/
  adm1.py        ADM1 with the BSM2 implementation details; the BSM2 reference case
  substrates.py  substrate composition -> ADM1 influent
  plant.py       the agricultural plant, daily feeding, hourly outputs
  scenarios.py   upsets, the healthy twin, the yield-loss ground truth
  sensors.py     online sensors, the FOS/TAC titration model, lab, feed log, the daily table
  methods.py     FOS/TAC rules, EWMA/CUSUM, PCA, expert rules, gradient boosting
  am2.py         the reduced model, its calibration from ADM1, the EKF and its alarms
  evaluate.py    alarm onsets, detection against chance, false alarms, lead time, common-rate curves,
                 bootstrap intervals, confusion
  study.py       the campaign; realdata.py the farm-data check; report.py figures and dashboard data
tests/           15 tests: BSM2 verification, plant plausibility, titration, scenarios, methods, scores,
                 benign-change limits, a bit-identical gradient boosting refit
```

## Data, sources and licence

Code: MIT. The simulator needs no data. Farm data, downloaded on demand and not part of this repository
(`results/` and the dashboard hold derived figures only: alarm counts, weekly means):

- D. Mulat (ILRI), *Process monitoring and weather datasets on farm-scale biogas digesters*, Mendeley Data
  V1, 2025, [doi:10.17632/gk3f363sfg.1](https://doi.org/10.17632/gk3f363sfg.1), CC BY 4.0. Changes: dates
  repaired where the spreadsheet swapped day and month, the cumulative gas meter differenced per day,
  impossible readings and temperatures (0 °C, 55 °C) set to missing, weekly means for display.

References:

- Batstone D.J. et al. (2002). *Anaerobic Digestion Model No. 1 (ADM1)*. IWA Scientific and Technical
  Report No. 13. IWA Publishing.
- Rosen C., Jeppsson U. (2006). *Aspects on ADM1 implementation within the BSM2 framework*. Department of
  Industrial Electrical Engineering and Automation, Lund University.
- Rosen C., Vrecko D., Gernaey K.V., Pons M.N., Jeppsson U. (2006). Implementing ADM1 for plant-wide
  benchmark simulations in Matlab/Simulink. *Water Science and Technology* 54(4), 11-19.
- Bernard O., Hadj-Sadok Z., Dochain D., Genovesi A., Steyer J.-P. (2001). Dynamical model development and
  parameter identification for an anaerobic wastewater treatment process. *Biotechnology and
  Bioengineering* 75(4), 424-438.
- Lübken M., Wichern M., Schlattmann M., Gronauer A., Horn H. (2007). Modelling the energy balance of an
  anaerobic digester fed with cattle manure and renewable energy crops. *Water Research* 41, 4085-4096.
- Koch K., Lübken M., Gehring T., Wichern M., Horn H. (2010). Biogas from grass silage - measurements and
  modeling with ADM1. *Bioresource Technology* 101, 8158-8165.
- Chen Y., Cheng J.J., Creamer K.S. (2008). Inhibition of anaerobic digestion process: a review.
  *Bioresource Technology* 99, 4044-4064.
- Nordmann W. (1977). Die Überwachung der Schlammfaulung. *KA-Informationen für das Betriebspersonal*,
  Beilage zur Korrespondenz Abwasser 3/77.
- MacGregor J.F., Kourti T. (1995). Statistical process control of multivariate processes. *Control
  Engineering Practice* 3(3), 403-414.
- FNR (2016). *Leitfaden Biogas - Von der Gewinnung zur Nutzung*. Fachagentur Nachwachsende Rohstoffe.
- KTBL (2013). *Faustzahlen Biogas*, 3rd ed. Kuratorium für Technik und Bauwesen in der Landwirtschaft.

The published BSM2 steady-state values used in the verification test are the ones tabulated by Rosen &
Jeppsson (2006), as also used in the test suites of open ADM1 implementations (QSDsan/EXPOsan, PyADM1).
