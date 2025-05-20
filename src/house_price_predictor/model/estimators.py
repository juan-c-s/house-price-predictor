import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, max_error,root_mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Clase abstracta Estimator (Strategy)
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

    def evaluate(self, y_true, y_pred):
        """
        Evalúa el rendimiento del modelo comparando las predicciones con los valores reales.
        Calcula varias métricas de evaluación relevantes.
        """
        print(f"Evaluando el modelo: {self.estimator_name}") # Añadido para claridad
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Manejar el caso de clasificación multiclase
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
        elif y_true.dtype in ['int64', 'int32', 'int8', 'bool']:
            # Manejar el caso de clasificación binaria
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        else:
            # Manejar el caso de regresión
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true,y_pred)
            max_err = max_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            return {
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
                "max_error": max_err,
                "mape": mape
            }


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


# Clase concreta para estimadores de TensorFlow
class TensorFlowEstimator(Estimator):
    """
    Clase que implementa la interfaz Estimator para modelos de TensorFlow.
    """
    def __init__(self, model: tf.keras.Model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.estimator_name = f"TensorFlow - {model.__class__.__name__}"
        self.history = None

    def train(self, X, y, epochs=10, batch_size=32, validation_data=None):
        """
        Entrena el modelo de TensorFlow.
        """
        print(f"Entrenando modelo de TensorFlow: {self.estimator_name}")
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=0)

    def predict(self, X):
        """
        Realiza predicciones con el modelo de TensorFlow.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        predictions = self.model.predict(X)
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        elif predictions.shape[1] == 1:
            return (predictions > 0.5).astype(int)
        else:
            return predictions

    def evaluate(self, y_true, y_pred):
        """
        Evalúa el modelo de TensorFlow.
        """
        return super().evaluate(y_true, y_pred)

    def get_history(self):
        return self.history


# Clase concreta para estimadores de Keras
class KerasEstimator(Estimator):
    """
    Clase que implementa la interfaz Estimator para modelos de Keras.
    """
    def __init__(self, model: keras.Model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.estimator_name = f"Keras - {model.__class__.__name__}"
        self.history = None

    def train(self, X, y, epochs=10, batch_size=32, validation_data=None):
        """
        Entrena el modelo de Keras.
        """
        print(f"Entrenando modelo de Keras: {self.estimator_name}")
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=0)

    def predict(self, X):
        """
        Realiza predicciones con el modelo de Keras.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        predictions = self.model.predict(X)
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        elif predictions.shape[1] == 1:
            return (predictions > 0.5).astype(int)
        else:
            return predictions

    def evaluate(self, y_true, y_pred):
        """
        Evalúa el modelo de Keras.
        """
        return super().evaluate(y_true, y_pred)

    def get_history(self):
        """
        Devuelve el historial de entrenamiento
        """
        return self.history



# Clase concreta para estimadores de PyTorch
class PyTorchEstimator(Estimator):
    """
    Clase que implementa la interfaz Estimator para modelos de PyTorch.
    """
    def __init__(self, model: nn.Module, criterion, optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.estimator_name = f"PyTorch - {model.__class__.__name__}"
        self.history = {'loss': [], 'val_loss': []}  # Para almacenar el historial de entrenamiento

    def train(self, X, y, epochs=10, batch_size=32, validation_data=None):
        """
        Entrena el modelo de PyTorch.
        """
        print(f"Entrenando modelo de PyTorch: {self.estimator_name}")
        # Convertir datos a tensores de PyTorch
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            if y.ndim == 1:
                y = y.unsqueeze(1)  # Asegurar que y tenga la forma (n_samples, 1) para regresión

        # Crear DataLoaders
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if validation_data:
            X_val, y_val = validation_data
            if not isinstance(X_val, torch.Tensor):
                X_val = torch.tensor(X_val, dtype=torch.float32)
            if not isinstance(y_val, torch.Tensor):
                y_val = torch.tensor(y_val, dtype=torch.float32)
                if y_val.ndim == 1:
                    y_val = y_val.unsqueeze(1)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.train()  # Establecer el modelo en modo de entrenamiento

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()  # Vaciar los gradientes
                outputs = self.model(inputs)  # Propagación hacia adelante
                loss = self.criterion(outputs, labels)  # Calcular la pérdida
                loss.backward()  # Propagación hacia atrás
                self.optimizer.step()  # Optimizar
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            self.history['loss'].append(epoch_loss)

            if validation_data:
                val_loss = 0.0
                self.model.eval()  # Establecer el modelo en modo de evaluación
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                epoch_val_loss = val_loss / len(val_loader)
                self.history['val_loss'].append(epoch_val_loss)
                self.model.train() # poner de nuevo en modo train
                print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')
        

    def predict(self, X):
        """
        Realiza predicciones con el modelo de PyTorch.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()  # Establecer el modelo en modo de evaluación para la inferencia
        with torch.no_grad():  # Desactivar el cálculo de gradientes
            predictions = self.model(X)
        
        # Convertir la salida a un array de NumPy
        predictions = predictions.numpy()
        
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        elif predictions.shape[1] == 1:
             return (predictions > 0.5).astype(int)
        else:
            return predictions

    def evaluate(self, y_true, y_pred):
        """
        Evalúa el modelo de PyTorch.
        """
        return super().evaluate(y_true, y_pred)

    def get_history(self):
        """
        Devuelve el historial de entrenamiento del modelo.
        """
        return self.history

