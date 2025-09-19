CLI Commands Used In ArgoPipeline :

python batch_runner.py -i ../data/2025/09 -o ../output/2025/09 --log ../output/2025/09/process_log.csv

python year_runner.py -y 2025 -d ../data -o ../output

python csv_to_parquet.py

python csv_to_parquet.py -y 2025 -o ../output -p ../parquet

python validate_parquet.py -y 2025 -o ../output -p ../parquet

python duckdb_loader.py -y 2025 -p ../parquet -d ../argo.duckdb

python pipeline/validate_duckdb.py

python vector_ingest.py -y 2025 -p ../parquet

python validate_chroma.py

python rag_pipeline.py -q "Show me salinity profiles near the equator in March 2025" -y 2025 -m 03 --top_k 30 --out ./pipeline_outputs/rag_equator_mar2025.json

python pipeline/query_validator.py

python pipeline/data_validator.py

python pipeline/rag_test_suite.py 