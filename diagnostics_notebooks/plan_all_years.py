"""Plan usable + search years for every region in one go (reads only a few columns, so it is quick) and print a comparison.

    python plan_all_years.py PEN DES IGP WGR ICR NEI HIM

Writes data/study_years/{REGION}.json for each region. To adjust a region, edit data/study_years/overrides.json
(see spatial_split.plan_years) and run this again, then run_regions.py.
"""
import sys
import spatial_split as ss

regions = sys.argv[1:] or ['HIM', 'PEN', 'DES', 'IGP', 'WGR', 'ICR', 'NEI']
for r in regions:
    try:
        ss.plan_years(r, n_spread=8, stress_years={'HIM': [2003, 2001]}.get(r))
    except Exception as e:
        print(f'[{r}] FAILED: {e}')
