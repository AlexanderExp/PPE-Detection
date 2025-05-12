.PHONY: exps tune

# текущие эксперименты
exps:
	./run_experiments.sh

# запуск тюнинга через Optuna
tune:
	dvc exp run tune

prune:
	dvc exp run prune
