# House Price Predictor

Un sistema de predicción de precios de viviendas utilizando machine learning y AWS.

## Requisitos Previos

- Python 3.12 o superior
- Credenciales de AWS
- [uv](https://github.com/astral-sh/uv) para gestión de dependencias (recomendado)

## Configuración del Entorno

1. Clonar el repositorio:
```bash
git clone https://github.com/juan-c-s/house-price-predictor
cd house-price-predictor
```
2. Correr cualquier comando con UV, esto te generará el ambiente virtual.
```bash
uv run python
```

## Configuración de Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto basado en `.env.example`:

```env
# AWS Credentials
AWS_ACCESS_KEY_ID=tu_access_key_id
AWS_SECRET_ACCESS_KEY=tu_secret_access_key
AWS_DEFAULT_REGION=us-east-1
```

## Estructura del Proyecto

```
house-price-predictor/
├── data/                    # Datos locales (si se necesitan)
├── src/
│   └── house_price_predictor/
│       ├── model/          # Implementaciones de modelos ML
│       └── dash_app/       # Aplicación web Dash
├── notebooks/              # Notebooks de desarrollo
└── mlruns/                # Seguimiento de experimentos MLflow
```


## Uso de la Aplicación Dash

La aplicación Dash proporciona una interfaz web para visualizar predicciones y métricas.

### Ejecutar la Aplicación

Antes de ejecutar la aplicación:
1. Asegúrate de tener el archivo `.env` configurado correctamente (ver sección "Configuración de Variables de Entorno")
2. Las variables de entorno son necesarias para:
   - Conexión con AWS S3

1. **Desde la raíz del proyecto** (recomendado):
```bash
make dash-app
```

La aplicación estará disponible en `http://localhost:8050`

La aplicación mostrará:
- Tabla de métricas de predicción

## Desarrollo

# Notas de Desarrollo

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
