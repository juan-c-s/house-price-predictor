`Y_pred = model.fit(X,Y)`

`evaluator = Evaluator (precision, f1, recall, auc_score)`

`model.evaluate(Y_pred, evaluator)`

`class Evaluator:`

`class Estimator:`

`def **init** (self, columns, estimator_name, output_label, evaluator: Evaluator, *args, **kwargs)`

`train`

`evaluate(`

`evaluate (Y_pred, Y, evaluator: Evaluator)` o `evaluate(Y_pred,Y)`

`class ScikitLearnEstimator (Estimator):`

`class DeepLearningEstimator(Estimator):`

`scikit_estimator = ScikitLearnEstimator(args, kwargs)`

`deep_estimator = DeepLearningEstimator(args,kwargs)`

`estimators: list[Estimator] = [`

`scikit_estimator,`

`deep_estimator,`

`]`

`for estimator in estimator:`

`metrics[estimator.estimator_name] = estimator.evaluate(Y,Y_pred)`

`visualize_metrics(metrics)`