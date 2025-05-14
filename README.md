# House Price Predictor
https://github.com/juan-c-s/house-price-predictor

Notion: 
https://www.notion.so/TODO-1f27e87b1e4880b69eeed353b3fb93ee?pvs=4

## EDA (Estadística, fundamentos, Álgebra) jueves Mayo 15

1. Feature Engineering : Cuales columnas escoger
    1. matriz correlación
    2. Kolmogrov: que tan parecido son a la distribucion normal
    3. distribución
2. Outliers: Qué datos pueden estar raros
3. Balance de datos general

## Modelo (Estadística, Álgebra)

1. Selección de modelos 
2. Sampling
    1. Stratified sampling (por clase)
    2. Separar el dataset
3. Interpretación resultados
4. Benchmark de modelos
5. cross-validation con los modelos

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

## Data Analytics | Visualizar la info (Estadística) Outputs y evaluación final

- Dados los resultados y concluir:
    - visualizar clusters
    - Métricas de predicción
- Testear modelo final por feature

## Teoría de ejecución ideal (Almacenamiento)

- Cómo el proyecto se ejecturaría a gran escala? (Usar AWS, Scrapear datos de Zillow, SageMaker)
- Data extraction
    - Zillow (batches)
    - scheduled Jobs (Cron-jobs)
        - updating dataset
        - fine-tuning the model

### Para el jueves 15 de Mayo

1. EDA ⇒ Prieto, Jorge
2. Modelo ⇒ Jorge, Juan Camilo, Prieto
    1. Evaluar dado unos datos (Y, Y_pred) 
    2. Escoger modelos
3. Data Analytics + Teoría de Almacenamiento ⇒ Daniel
4. Orchestrador: Revisar PR's, calidad ⇒ Juan Camilo