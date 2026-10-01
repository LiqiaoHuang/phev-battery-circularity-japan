# PHEV battery circularity in Japan

Python code and selected model inputs for evaluating a domestic circular strategy for plug-in hybrid electric vehicle (PHEV) batteries in Japan.

The model tracks vehicle and battery flows from 2022 to 2050, including overseas vehicle service, domestic battery capture, second-life stationary storage, recycling and primary material make-up. It compares a business-as-usual scenario with a domestic circular strategy and links the resulting activities to environmental and social impact calculations.

The fleet model runs using the files in this repository. Impact calculations require separately supplied characterised factors from the corresponding licensed product systems. The ecoinvent and PSILCA databases and their numerical factor libraries are **not included**.

## Quick start

Use Python 3.10 or later. Run these commands from the repository folder.

```bash
python -m pip install -r requirements.txt
python run_model.py
```

To run the base case and six distinct intervention settings, use

```bash
python run_model.py --case all
```

No commercial database is needed for this step. The calculation produces five annual physical-flow tables and two impact-activity tables in `results/`, which is created when the model runs. BAU stationary storage uses new LFP modules sized to provide the same cumulative delivered storage service as the second-life system in the Circular scenario.

## Code and inputs

| File | Contents |
| --- | --- |
| `run_model.py` | Input loading and model execution |
| `fleet_model.py` | Battery degradation, vehicle routing, stationary service and material recovery |
| `activities.py` | Physical-to-impact activity conversion and A0–A6 replacement assumptions |
| `electricity.py` | AR6 and Japanese JMIP2 electricity pathway processing |
| `impact_calculation.py` | Impact aggregation, scenario comparison and electricity adjustments |
| `data/parameters.json` | Battery, degradation, routing and intervention assumptions |
| `data/manufacturing_schedule.csv` | Annual manufacturing trajectory from Supplementary Table S1 |
| `data/scenarios.csv` | Base case and intervention settings |
| `data/replacement_scenarios.csv` | A0–A6 replacement assumptions from Supplementary Table S8 |
| `data/electricity_scenarios.csv` | Selected electricity models, scenarios and source regions from Supplementary Table S2 |
| `data/activity_mapping.csv` | Source columns, countries, reference units, conversion multipliers, credit signs and electricity weights |
| `data/social_indicators.csv` | The 91 indicators retained for the main social analysis |

Future manufacturing values are model assumptions, not observed sales. Rates, depth of discharge and state of health are fractions. Capacity is in kWh, `T50_mean` is in cycles and `dSoH_per_cycle` is capacity fade per cycle. Recovered products are expressed in kg per pack, not elemental metal masses. Service-life distributions are specified in `fleet_model.py` and Supplementary Note 2.

Output column names identify packs, modules, kg, kWh, km and pack-days. Stock and average columns must not be summed as flows. Repeated 0.5 settings in the intervention sweeps are represented by the single base case.

## Activity units and system boundaries

Environmental battery production uses 86.6 kg per NMC pack and 27 kg per LFP module. Environmental electricity factors are per kWh. Social battery factors are per pack or module, while social electricity factors are per 1000 kWh. Recycling and disposal use pack or module reference quantities. Salt production and avoided products use kg.

Recycling credits have negative activity quantities. Social credit factors use absolute magnitudes, following the study convention. Environmental factors retain their LCIA signs. Supplied factors must not already include credit signs or pack-to-mass multipliers.

Environmental hydrometallurgical product systems include upstream black mass preparation. Separate black mass activities are therefore added only in the social calculation, whose factor boundaries differ. `Battery_Production_packs` denotes the battery production supply chain, including upstream materials, rather than final assembly alone. Country identifies the foreground activity location, not the complete geographical distribution of supply-chain impacts. The identifier `Avoided_vergin` is retained for factor matching.

## Impact calculations

Supply an authorised factor CSV with these columns.

```text
factor_key,activity_unit,indicator,factor
```

For environmental factors, `factor_key` is the component identifier in `activity_mapping.csv`. For social factors, it is `social_factor_key`. Units must match `environmental_unit` or `social_unit` exactly.

Social factors are characterised medium risk hours (mrh) per reference activity and must not be multiplied by monetary reference amounts again. Environmental factors must match the study's foreground boundaries and chosen LCIA method. The main prospective environmental calculation uses 18 ReCiPe midpoint indicators.

```bash
python impact_calculation.py --assessment social --factors PATH_TO_SOCIAL_FACTORS.csv
python impact_calculation.py --assessment environment --factors PATH_TO_ENVIRONMENT_FACTORS.csv
```

These commands produce component contributions and scenario comparisons. Social results include the sum over the 91 retained indicators. The main social calculation holds factors constant over time. The environmental command above uses a static grid. For the prospective environmental calculation, supply the electricity inputs described below.

The code accepts characterised exports from matching foreground product systems. It does not construct or solve those systems in openLCA. Access to a commercial database alone does not supply the study-specific factor exports.

## Electricity pathways

Obtain the AR6 Scenario Database version 1.1 ISO3 and R10 CSV downloads from the [AR6 Scenario Explorer](https://data.ece.iiasa.ac.at/ar6/). The dataset record and access terms are documented by [Byers et al. (2022)](https://doi.org/10.5281/zenodo.7197970). Place the two files named in `electricity_scenarios.csv` in one directory.

```bash
python electricity.py --source-dir PATH_TO_AR6_DIRECTORY
```

Japan uses the [JMIP 2 Net Zero CDR dataset](https://doi.org/10.5281/zenodo.8383675) by Sugiyama (2024). Add its downloaded CSV and an authorised Japanese technology-factor CSV.

```bash
python electricity.py --source-dir PATH_TO_AR6_DIRECTORY --japan-source PATH_TO_JMIP2.csv --japan-technology-factors PATH_TO_JAPAN_TECHNOLOGY_FACTORS.csv
```

Japanese technology factors have columns `technology`, `indicator` and `factor_per_kwh`. Technology identifiers are defined in `electricity.py`. The code weights matched technologies at each original model-year, takes the indicator-specific median across available models and then interpolates. Missing whole model-years are omitted rather than assigned zeros. The generated `japan_model_factors.csv` records matched generation coverage.

Foreign technology factors have columns `country`, `technology`, `indicator` and `factor_per_kwh`. Countries use ISO3 codes. Match technologies to `annual_electricity_shares.csv`. For multiple providers, use their weighted mean factor per kWh. Convert factors expressed per MJ to per kWh using 3.6 MJ per kWh.

```bash
python impact_calculation.py --assessment environment --factors PATH_TO_ENVIRONMENT_FACTORS.csv --grid-shares results/annual_electricity_shares.csv --technology-factors PATH_TO_FOREIGN_TECHNOLOGY_FACTORS.csv --japan-model-factors results/japan_model_factors.csv
```

Japan uses annual absolute factors. Overseas explicit electricity uses the 2022 purchased-electricity factor multiplied by an indicator-specific generation index. Chinese and Korean embedded-electricity adjustments use recovery-weighted indices and assumed fractions of 0.40 for hydrometallurgy and 0.55 for materials. These are approximate scenario adjustments, not complete future inventories. Supplementary Note 4 describes the pathways and geographical proxies.

## Scenario and sensitivity analysis

The study compares predefined scenarios and sensitivities rather than applying formal multiobjective optimisation.

The default activity tables use A0. To generate another replacement case without rerunning the fleet model, use a separate output directory.

```bash
python activities.py --flows results --alternative A3 --output results/A3
```

Then add `--flows results/A3` to the impact command. A0 and A1 share the same incremental battery boundary. The remaining cases adjust new NMC or LFP production, LFP disposal proxies, NMC recovery credits and replacement electricity. Korean recovery removals use replacement-vehicle retirement timing. A4 reduces mobility service and is not service equivalent. Whole-vehicle production and import transport are excluded from these adjustments.

The `--case all` option varies export suppression, domestic capture and storage acceptance one at a time. The parameter files and Supplementary Notes 1 and 2 describe the assumptions. Embedded-electricity fractions are exposed in `activity_mapping.csv`.

## Availability and licence

The code is released under the [MIT License](LICENSE). The included data files document selected study inputs, assumptions and mappings. The code licence does not grant rights to external datasets, commercial databases or numerical factor exports.

Access to [ecoinvent](https://ecoinvent.org/) and [PSILCA](https://psilca.net/) is governed by their providers' licences. Original AR6 and JMIP2 data should be obtained from the sources linked above. This repository does not contain commercial inventories, numerical factor libraries, private analysis archives or precomputed impact results.

Methods, assumptions and supplementary tables should be read together with the accompanying manuscript and Supplementary Information.
