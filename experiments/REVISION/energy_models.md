# Locked empirical energy-accounting models

**Status:** pre-computation specification, 2026-09-01. These coefficients and
equations are fixed before `analyze_energy_extensions.py` is run. They are
literature-parameterized accounting checks, not a calibration to the study's routes
or to a proposed deployment fleet.

## Route accounting equation

For each fixed time-oriented H2-v3 plan, let `D_D` be aggregate drone flight distance,
`D_T` hybrid-truck distance, and `L_0` the matched LKH truck-only route length. For a
published drone distance intensity `q_D` and ground-vehicle distance intensity `q_T`
(both MJ/km), the check is

```
E_hybrid = q_D D_D + q_T D_T
E_truck_only = q_T L_0
energy saving (%) = 100 (1 - E_hybrid / E_truck_only).
```

A plan is `faster_and_lower_energy` only when its nominal makespan is below `L_0`
and `E_hybrid < E_truck_only`. Solver replicates are averaged within each instance
before instances are resampled for the reported fraction and 95% interval.

The geometric unit is set to 1 km, truck speed to 1 km per model-time unit, and drone
speed to `alpha` km per model-time unit. Every drone operation carries one package on
launch-to-customer travel and returns empty on customer-to-rendezvous travel. The
energy *ratio* is invariant to any common positive conversion of model distance to km,
so the 1-km convention supplies units rather than a claim about district size.

These equations deliberately omit a fixed takeoff/landing surcharge, hovering and
truck idling. Those omissions are evaluated separately by the generalized boundary
grid in the same analysis and must remain in the manuscript's limitations.

## Rodrigues et al. (2022), Patterns — very-small-package quadcopter and vans

Source: T. A. Rodrigues et al., “Drone flight data reveal energy and greenhouse gas
emissions savings for very small package delivery,” *Patterns* 3, 100569 (2022),
[doi:10.1016/j.patter.2022.100569](https://doi.org/10.1016/j.patter.2022.100569).

Use the version-of-record Table 3 base-case distance intensities:

| Vehicle | `q` (MJ/km) |
|---|---:|
| Small quadcopter drone | 0.08 |
| Small electric van | 1.65 |
| Small diesel van | 4.90 |

The drone coefficient describes a DJI Matrice 100 carrying 0.5 kg, cruising at 12
m/s and 100 m, over a 4-km round trip with a loaded outbound and unloaded return leg.
The table includes the stated charging/transmission losses for electric vehicles. The
paper is based on 188 flights. This final-paper coefficient replaces the earlier
preprint's 0.05 MJ/km value; the preprint is not used.

Two arms are reported separately:

- `rodrigues_quadcopter_electric_van`: (`q_D`, `q_T`) = (0.08, 1.65)
- `rodrigues_quadcopter_diesel_van`: (`q_D`, `q_T`) = (0.08, 4.90)

## Stolaroff et al. (2018), Nature Communications — small quadcopter and Class-4 truck

Source: J. K. Stolaroff et al., “Energy use and life cycle greenhouse gas emissions
of drones for commercial package delivery,” *Nature Communications* 9, 409 (2018),
[doi:10.1038/s41467-017-02411-5](https://doi.org/10.1038/s41467-017-02411-5),
including the published correction.

Use the article's direct average small-quadcopter estimate of 32 J/m = 0.032 MJ/km
(Methods, “Emissions from electricity”) and its reviewed Class-4 parcel-vehicle final
energy intensities (Methods, “Ground vehicle parameters”):

| Vehicle | `q` (MJ/km) |
|---|---:|
| Small quadcopter drone | 0.032 |
| Class-4 electric parcel truck | 2.44 |
| Class-4 diesel parcel truck | 7.30 |

The drone estimate is derived from measured flight segments and a calibrated physical
model; the paper reports 1,073 outdoor flight segments and 0–7 m/s winds. The truck
intensities are vehicle-distance—not package-delivery or life-cycle—values. Two arms
are reported separately:

- `stolaroff_quadcopter_class4_ev`: (`q_D`, `q_T`) = (0.032, 2.44)
- `stolaroff_quadcopter_class4_diesel`: (`q_D`, `q_T`) = (0.032, 7.30)

## Interpretation boundary

The four arms are not averaged. They mix different drone payloads, ground-vehicle
classes, system boundaries and years, so agreement is a robustness check and
disagreement is a scope finding. They do not include battery reserve, wind-specific
re-routing, traffic, takeoff/landing energy beyond what is embedded in a published
distance average, warehousing, battery manufacture or other life-cycle impacts.
Accordingly, the output may support only conditional statements of the form “under
the Rodrigues electric-van parameterization,” never a universal claim that hybrid
truck–drone delivery is greener.
