.PHONY: run-all run-all-shap clean

# Always enforce deterministic behavior
export PYTHONHASHSEED=42
export MPLBACKEND=Agg

ifeq ($(CI),1)
# Use smaller dataset for CI smoke test
DATA_CMD=echo "Using fixed dataset: data/fragrance_data.csv"
TRAIN_CMD=python src/train.py --data data/fragrance_data.csv --out_dir artifacts --estimator xgb --ci-mode
else
# Use full dataset locally (reproducible with fixed seed)
DATA_CMD=echo "Using fixed dataset: data/fragrance_data.csv"
TRAIN_CMD=python src/train.py --data data/fragrance_data.csv --out_dir artifacts --estimator xgb
endif

run-all:
	$(DATA_CMD)
	$(TRAIN_CMD)
	python src/make_predictions.py --data data/fragrance_data.csv --model artifacts/champion_model.pkl --out artifacts/predictions_test.csv --index artifacts/test_index.csv
	python src/evaluate.py --pred artifacts/predictions_test.csv --data data/fragrance_data.csv --outdir reports
	python src/generate_pngs.py --pred artifacts/predictions_test.csv --outdir reports/figures
	python src/fairness_eval.py --data data/fragrance_data.csv --pred artifacts/predictions_test.csv --outdir reports

run-all-shap:
	$(DATA_CMD)
	$(TRAIN_CMD)
	python src/make_predictions.py --data data/fragrance_data.csv --model artifacts/champion_model.pkl --out artifacts/predictions_test.csv --index artifacts/test_index.csv
	python src/evaluate.py --pred artifacts/predictions_test.csv --data data/fragrance_data.csv --outdir reports
	python src/generate_pngs.py --pred artifacts/predictions_test.csv --outdir reports/figures --with-shap
	python src/fairness_eval.py --data data/fragrance_data.csv --pred artifacts/predictions_test.csv --outdir reports

clean:
	rm -f artifacts/*.csv artifacts/*.pkl artifacts/*.json
	rm -f reports/*.csv
	rm -f reports/figures/*.png
	@echo "Cleaned artifacts and reports"
