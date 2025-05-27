# Documentación del Proyecto de Predicción de Precios de Viviendas

## Descripción General del Proyecto
Este proyecto implementa una solución de aprendizaje automático para predecir precios de viviendas basado en diversas características. La implementación incluye un preprocesamiento integral de datos, comparación de múltiples modelos, seguimiento de experimentos con MLflow, y un pipeline de datos escalable utilizando AWS EMR y S3 para almacenamiento y procesamiento de datos.

## Descripción General de la Arquitectura

### Pipeline de Datos
1. **Ingesta de Datos**
   - Ingesta de datos sin procesar en la zona raw de S3
   - Validación y limpieza de datos en cluster EMR
   - Datos procesados almacenados en zonas trusted/refined

2. **Capa de Procesamiento**
   - Cluster EMR con Spark para procesamiento distribuido
   - PySpark para transformaciones de datos y entrenamiento de modelos
   - Particionamiento de datos para procesamiento eficiente

3. **Visualización Frontend**
   - Aplicación web Dash para exploración interactiva de datos
   - DuckDB para rendimiento eficiente de consultas
   - Implementación de carga diferida para grandes conjuntos de datos

## Procesamiento de Datos e Ingeniería de Características

### Características Utilizadas
- **Características Numéricas**: 
  - área
  - baños
  - pisos
  - estacionamiento
  - dormitorios

- **Características Binarias**:
  - aire acondicionado (sí/no)
  - área preferencial (sí/no)

- **Características Nominales**:
  - estado del amueblado (amueblado/semi-amueblado/sin amueblar)

### Pipeline de Preprocesamiento
1. **Limpieza de Datos**:
   - Imputación de valores faltantes
     - Estrategia de mediana para características numéricas
     - Estrategia de moda para características categóricas
   - Escalado de características usando StandardScaler
   - Codificación one-hot para características nominales

2. **División de Datos**:
   - Conjunto de entrenamiento: 80%
   - Conjunto de prueba: 20%
   - Estado aleatorio: 42 para reproducibilidad

## Implementación del Modelo

### Modelos Evaluados
1. **Regresión Lineal**
   - Modelo base para comparación
   - Resultados interpretables simples

2. **Random Forest Regressor**
   - Enfoque de aprendizaje conjunto
   - Mejor manejo de relaciones no lineales

3. **XGBoost Regressor**
   - Gradient boosting avanzado
   - Hiperparámetros:
     - n_estimators: 100
     - random_state: 42

### Métricas de Evaluación
- Puntuación R² (Coeficiente de Determinación)
- Error Cuadrático Medio (RMSE)
- Error Absoluto Medio (MAE)
- Error Porcentual Absoluto Medio (MAPE)
- Error Máximo

## Resultados del Análisis de Datos

### Distribución de Precios
- **Rango**: 1,750,000 a 13,300,000
- **Asimetría**: 1.21 (distribución sesgada a la derecha)
- **Visualización de Distribución**: El análisis de histograma muestra concentración de precios en rangos más bajos

### Importancia de Características
- El análisis de importancia de características de XGBoost revela determinantes clave de precios
- Visualización usando gráficos de barras seaborn
- Ayuda a identificar las características más influyentes en la predicción de precios

## Implementación AWS

### Arquitectura del Data Lake en S3
- **Zona Raw**: Ingesta inicial de datos
  - Archivos CSV/Parquet originales
  - Almacenamiento de datos sin procesar
  
- **Zona Trusted**: Datos validados
  - Verificaciones de calidad de datos
  - Conjuntos de datos limpios
  
- **Zona Refined**: Datos listos para análisis
  - Características procesadas
  - Predicciones del modelo
  - Resultados agregados

### Configuración del Cluster EMR
- **Nodo Master**: Gestión y coordinación del cluster
- **Nodos Worker**: Procesamiento distribuido
- **Scripts de Bootstrap**: Inicialización y configuración del cluster (instalación de numpy)
- **Integración con Jupyter**: Entorno de desarrollo interactivo

### Integración DuckDB
- **Implementación de Carga Diferida**:
  - Consulta eficiente de datos en S3
  - Huella mínima de memoria
  - Carga dinámica de datos basada en interacción
  
- **Integración Frontend con Dash**:
  - Visualización de datos
  - Visualización paginada de datos
  - Filtrado y ordenamiento interactivo
  - Consulta directa a zona refined de S3
  ![Aplicación Dash](evidence/Dash-app.jpg)

## Evidencia de Implementación AWS

1. **Configuración de Buckets S3 y Zonas de Datos**
   - Configuración de zonas Raw, Trusted y Refined
   - Scripts de bootstrap para configuración del cluster
![Configuración de Buckets S3](evidence/buckets.jpg)

2. **Configuración del Cluster EMR**
   - Configuración del cluster y ajustes de seguridad
   - Especificaciones de red e instancias
![Configuración EMR](evidence/emr-setup.jpg)

3. **Instancias EC2 para EMR**
   - Nodos master y worker
   - Capacidad de procesamiento distribuido
![Instancias EC2](evidence/ec2-instances.jpg)

4. **Entorno de Desarrollo PySpark**
   - Integración con Jupyter notebook
   - Procesamiento interactivo de datos
![Notebook EMR Jupyter](evidence/jupyter.jpg)

5. **Ejecución del Pipeline de Datos**
   - Operaciones de escritura en S3
   - Monitoreo y depuración del pipeline
![Error de escritura en bucket](evidence/jupyter2.jpg)

6. **Integración de Athena (Data Warehouse)**
    - Configuración de catalogación automática de datos con **AWS Glue** para entender la estructura de datos
    - Listo para futuras consultas SQL y reportes

## Estructura del Proyecto
```
house-price-predictor/
├── data/
│   ├── raw/          # Archivos de datos iniciales
│   └── staged/       # Conjuntos de datos procesados
├── src/
│   └── house_price_predictor/
│       ├── model/    # Implementaciones de modelos ML
│       │   ├── estimators.py
│       │   └── evaluator.py
│       └── dash_app/ # Visualización frontend
├── mlruns/          # Seguimiento de experimentos MLflow
└── notebooks/       # Notebooks de desarrollo
```

## Conclusión
El proyecto implementa exitosamente un sistema integral de predicción de precios de viviendas con:
- Preprocesamiento robusto de datos e ingeniería de características
- Múltiples implementaciones y comparaciones de modelos
- Infraestructura AWS escalable usando EMR y S3
- Acceso eficiente a datos usando DuckDB
- Visualización interactiva a través de Dash
- Integración con prácticas modernas de MLOps

El sistema está diseñado para precisión y escalabilidad, con potencial para mejoras futuras y despliegue en producción. La combinación de servicios AWS, DuckDB y Dash crea un pipeline de datos potente y eficiente para predicciones de precios de viviendas en tiempo real.
