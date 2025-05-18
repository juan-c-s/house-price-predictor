from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_percentage_error, max_error, mean_absolute_error
from sklearn.base import BaseEstimator
import numpy as np


class Estimator:
    """
    Clase abstracta que define la interfaz para los estimadores.
    Todas las clases concretas de estimadores deben heredar de esta clase.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.estimator_name = "Estimador Base" # Nombre por defecto

    def train(self, X, y):
        """
        Entrena el modelo con los datos de entrenamiento.
        Este método debe ser implementado por las subclases.
        """
        raise NotImplementedError("El método train debe ser implementado por la subclase")

    def predict(self, X):
        """
        Realiza predicciones con el modelo entrenado.
        Este método debe ser implementado por las subclases.
        """
        raise NotImplementedError("El método predict debe ser implementado por la subclase")

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evalúa el rendimiento del modelo comparando las predicciones con los valores reales.
        Calcula varias métricas de evaluación relevantes.
        """
        print(f"Evaluando el modelo: {self.estimator_name}") # Añadido para claridad

        # If predictions are continuous, use regression metrics
        if np.issubdtype(y_pred.dtype, np.floating) or np.issubdtype(y_pred.dtype, np.complexfloating):
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            max_err = max_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            return {
                "r2": r2,
                "rmse": rmse,
                "mse": mse,
                "mae": mae,
                "max_error": max_err,
                "mape": mape
            }
        # If predictions are integer and match the set of true labels, use classification metrics
        elif np.issubdtype(y_pred.dtype, np.integer) or np.issubdtype(y_pred.dtype, np.bool_):
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                # Multiclass classification
                accuracy = accuracy_score(y_true, np.argmax(y_pred, axis=1))
                precision = precision_score(y_true, np.argmax(y_pred, axis=1), average='macro', zero_division=0)
                recall = recall_score(y_true, np.argmax(y_pred, axis=1), average='macro', zero_division=0)
                f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average='macro', zero_division=0)
                return {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            else:
                # Binary or multiclass classification
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                return {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
        else:
            raise ValueError("Unknown prediction type for evaluation.")


# Clase concreta para estimadores de Scikit-learn
class ScikitLearnEstimator(Estimator):
    """
    Clase que implementa la interfaz Estimator para modelos de Scikit-learn.
    Puede encapsular cualquier modelo de Scikit-learn.
    """
    # Estos prints pueden volverse algo más reportable.
    def __init__(self, model: BaseEstimator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.estimator_name = f"ScikitLearn - {model.__class__.__name__}"  # Nombre descriptivo
        # mlflow.sklearn.autolog(True)  # Removed to keep MLflow run management outside the estimator

    def train(self, X, y):
        """
        Entrena el modelo de Scikit-learn con los datos de entrenamiento.
        """
        print(f"Entrenando modelo de Scikit-learn: {self.estimator_name}") # Añadido para claridad
        self.model.fit(X, y)

    def predict(self, X):
        """
        Realiza predicciones con el modelo de Scikit-learn entrenado.
        """
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """
        Evalúa el modelo de Scikit-learn
        """
        return super().evaluate(y_true, y_pred)


# # Clase concreta para estimadores de Deep Learning (Keras)
# class DeepLearningEstimator(Estimator):
#     """
#     Clase que implementa la interfaz Estimator para modelos de Deep Learning (Keras).
#     Puede encapsular cualquier modelo de Keras.
#     """
#     def __init__(self, model: keras.Model, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.model = model
#         self.estimator_name = f"DeepLearning - {model.__class__.__name__}" # Nombre descriptivo
#         self.history = None  # Para almacenar el historial de entrenamiento

#     def train(self, X, y, epochs=10, batch_size=32, validation_data=None):
#         """
#         Entrena el modelo de Keras con los datos de entrenamiento.
#         """
#         print(f"Entrenando modelo de Deep Learning: {self.estimator_name}") # Añadido para claridad
#         # Agregar manejo de tipos para asegurar que X e y sean ndarray
#         if not isinstance(X, np.ndarray):
#             X = np.array(X)
#         if not isinstance(y, np.ndarray):
#             y = np.array(y)
            
#         self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=0) # suprimido verbose
#     def predict(self, X):
#         """
#         Realiza predicciones con el modelo de Keras entrenado.
#         """
#         # Agregar manejo de tipos para asegurar que X sea ndarray
#         if not isinstance(X, np.ndarray):
#             X = np.array(X)
#         predictions = self.model.predict(X)
#         # Keras devuelve probabilidades para clasificación, necesitamos convertirlas a clases
#         if predictions.ndim > 1 and predictions.shape[1] > 1:  # Más de una clase
#             return np.argmax(predictions, axis=1)  # Devuelve la clase con la probabilidad más alta
#         elif predictions.shape[1] == 1:
#             return (predictions > 0.5).astype(int)
#         else:
#           return predictions

#     def evaluate(self, y_true, y_pred):
#         """
#         Evalúa el modelo de Keras.
#         """
#         return super().evaluate(y_true, y_pred)
    
#     def get_history(self):
#         """
#         Devuelve el historial de entrenamiento del modelo.
#         """
#         return self.history
