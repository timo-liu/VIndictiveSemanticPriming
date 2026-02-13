#./run.ps1
python rho_analysis.py # computes rho graphs
python rsa.py --output "D:/WithoutCategory/rsa_meta.csv" --dir "D:/redone_heatmaps" # computes rsa
python .\rsa_rho_analysis.py  --data "D:/WithoutCategory/rsa_meta.csv" --graphs "redone_rsa_graphs"
python dissect.py --input "rsa_rho_dump.pkl" --collapse_tasks --collapse_isis