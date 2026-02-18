# preprocess the data
# python preprocess.py --input "data/nam.xlsx" --output "data/nam.xlsx"
# python preprocess.py --input "data/ldt.xlsx" --output "data/ldt.xlsx"

# split relations completely
# python quickfix.py --input "data/nam.xlsx" --output "data/nam_split_relations.xlsx"
# python quickfix.py --input "data/ldt.xlsx" --output "data/ldt_split_relations.xlsx"

# combine relations into one big aggregate
# python quickfixaggregatefeature.py --input "data/nam.xlsx" --output "data/nam_aggregated.xlsx"
# python quickfixaggregatefeature.py --input "data/ldt.xlsx" --output "data/ldt_aggregated.xlsx"

# split into similarity and associate buckets
# python quickfixaggregatewithinfunc_perc.py --input "data/nam.xlsx" --output "data/nam_func_v_perc.xlsx"
# python quickfixaggregatewithinfunc_perc.py --input "data/ldt.xlsx" --output "D:/Bootstraps/ldt_func_v_perc.xlsx"

# do rho extraction...
# "normal"
# ./run_split.ps1

# aggregate
# ./run_aggregated.ps1

# buckets
./run_func_v_perc.ps1

# do rho analysis
# python rho_analysis.py --rhos "D:/split_priming/rhos.json"

# python rho_analysis.py --rhos "D:/aggregate_priming/rhos.json"

# python rho_analysis.py --rhos "D:/bucket_priming/rhos.json" --graphs "D:/bucket_priming/rho_graphs"

python rho_analysis.py --rhos "D:/func_v_perc/rhos.json" --graphs "D:/func_v_perc/graphs/" --a "perceptual" --b "functional"