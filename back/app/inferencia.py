"""
inferencia.py — Funciones de inferencia para cada modelo.
Maneja: conversión a tensor, dispositivo (CPU/CUDA), sigmoid, y formatos de entrada.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config as C


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convierte array numpy a tensor y lo mueve al dispositivo."""
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def predecir_m1(modelo: nn.Module, imagenes: list[np.ndarray]) -> list[float]:
    """
    M1: MobileNetV3-Small para detección de ojos cerrados.
    
    Args:
        modelo: M1 cargado (state_dict)
        imagenes: lista de 2 arrays numpy (ojo_der, ojo_izq), cada uno (3, 96, 96) float32
    
    Returns:
        lista de 2 probabilidades [p_ojo_der, p_ojo_izq] en [0, 1]
    """
    device = next(modelo.parameters()).device
    
    # Stack de imágenes: (2, 3, 96, 96)
    batch = np.stack(imagenes, axis=0)
    tensor = _to_tensor(batch, device)
    
    # Forward
    logits = modelo(tensor)  # (2, 1)
    probs = torch.sigmoid(logits).squeeze(-1)  # (2,)
    
    return probs.cpu().tolist()


@torch.no_grad()
def predecir_m2(modelo: nn.Module, imagen: np.ndarray) -> float:
    """
    M2: MobileNetV3-Small para detección de bostezo.
    
    Args:
        modelo: M2 cargado (state_dict)
        imagen: array numpy (3, 96, 96) float32
    
    Returns:
        probabilidad de bostezo en [0, 1]
    """
    device = next(modelo.parameters()).device
    
    # Agregar batch dimension: (1, 3, 96, 96)
    tensor = _to_tensor(imagen[np.newaxis, ...], device)
    
    # Forward
    logits = modelo(tensor)  # (1, 1)
    prob = torch.sigmoid(logits).squeeze()  # scalar
    
    return prob.cpu().item()


@torch.no_grad()
def predecir_m3(modelo: nn.Module, tensor_senales: np.ndarray) -> float:
    """
    M3: CnnLstm(senales=True) para somnolencia por video.
    
    Args:
        modelo: M3 cargado (state_dict con buffers media/desv)
        tensor_senales: array numpy (1, 10, 6) float32 con NaN donde faltan frames
    
    Returns:
        probabilidad de somnolencia en [0, 1]
    """
    device = next(modelo.parameters()).device
    
    # Convertir a tensor (mantener NaN)
    tensor = torch.from_numpy(tensor_senales).to(device)
    
    # Forward
    logit = modelo(tensor)  # (1, 1)
    prob = torch.sigmoid(logit).squeeze()  # scalar
    
    return prob.cpu().item()


@torch.no_grad()
def predecir_m4(modelo, df_features: pd.DataFrame) -> float:
    """
    M4: Pipeline SVM o RandomForest para somnolencia por descriptores.
    
    Args:
        modelo: M4 cargado (Pipeline o RandomForest de scikit-learn)
        df_features: DataFrame con 4 columnas: [ear_medio, perclos, parpadeos, mar_max]
    
    Returns:
        probabilidad de somnolencia en [0, 1]
    """

    probas = modelo.predict_proba(df_features)  # (1, 2)
    prob = probas[0, 1]  # P(somnoliento)
    
    return float(prob)