run-all:
ifeq ($(CI),1)
	python src/data_prep.py --rows 800 --seed 42 --out data/fragrance_data.csv
	python src/train.py --data data/fragrance_data.csv --out_dir artifacts --estimator xgb --ci-mode
else
	python src/data_prep.py --rows 20000 --seed 42 --out data/fragrance_data.csv
	python src/train.py --data data/fragrance_data.csv --out_dir artifacts --estimator xgb
endif
	python src/make_predictions.py --data data/fragrance_data.csv --model artifacts/champion_model.pkl --out artifacts/predictions.csv
	python src/evaluate.py --pred artifacts/predictions.csv --data data/fragrance_data.csv --outdir reports
	python src/generate_pngs.py --pred artifacts/predictions.csv --outdir reports/figures
