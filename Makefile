PY=python

# Generate synthetic data
run-data:
	$(PY) src/data_prep.py

# Train model and save artifact
run-train:
	$(PY) src/train.py --data data/fragrance_data.csv --out_dir artifacts --estimator xgb

# Make predictions using trained model
run-predict:
	$(PY) src/make_predictions.py --data data/fragrance_data.csv --out artifacts/predictions.csv

# Evaluate model performance
run-eval:
	$(PY) src/evaluate.py --pred artifacts/predictions.csv --data data/fragrance_data.csv --outdir reports

# Generate figures (ROC, PR, Lift, SHAP)
run-figs:
	$(PY) src/generate_pngs.py

# Run full pipeline end-to-end
run-all: run-data run-train run-predict run-eval run-figs
	@echo "Full ML pipeline executed successfully!"
