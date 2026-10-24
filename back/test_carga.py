"""Script para verificar que los modelos cargan correctamente."""
import sys
from pathlib import Path

# Subir un nivel para que la carpeta 'back/' esté en sys.path
# Así Python puede ver 'app' como un paquete
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

from app.models_loader import cargar_modelos
import torch

def main():
    print("Iniciando carga de modelos...\n")
    modelos = cargar_modelos()
    
    print("\n" + "="*60)
    print("PRUEBAS DE INFERENCIA")
    print("="*60)
    
    # Prueba M1 (ojo)
    print("\n🔬 M1 - Inferencia de prueba:")
    dummy_m1 = torch.randn(1, 3, 96, 96).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out_m1 = modelos.m1(dummy_m1)
        prob_m1 = torch.sigmoid(out_m1).item()
    print(f"   Logit: {out_m1.item():.4f}")
    print(f"   Probabilidad: {prob_m1:.4f}")
    
    # Prueba M2 (boca)
    print("\n🔬 M2 - Inferencia de prueba:")
    dummy_m2 = torch.randn(1, 3, 96, 96).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out_m2 = modelos.m2(dummy_m2)
        prob_m2 = torch.sigmoid(out_m2).item()
    print(f"   Logit: {out_m2.item():.4f}")
    print(f"   Probabilidad: {prob_m2:.4f}")
    
    # Prueba M3 (LSTM señales)
    print("\n🔬 M3 - Inferencia de prueba:")
    # Crear tensor con 7 frames reales y 3 de NaN
    dummy_m3 = torch.randn(1, 10, 6).to("cuda" if torch.cuda.is_available() else "cpu")
    dummy_m3[0, 7:, :] = float('nan')  # Últimos 3 frames son NaN
    with torch.no_grad():
        out_m3 = modelos.m3(dummy_m3)
        prob_m3 = torch.sigmoid(out_m3).item()
    print(f"   Logit: {out_m3.item():.4f}")
    print(f"   Probabilidad: {prob_m3:.4f}")
    
    # Prueba M4 (scikit-learn)
    print("\n🔬 M4 - Inferencia de prueba:")
    import numpy as np
    dummy_m4 = np.random.rand(1, 4)  # [ear_medio, perclos, parpadeos, mar_max]
    prob_m4 = modelos.m4.predict_proba(dummy_m4)[0, 1]
    print(f"   Probabilidad: {prob_m4:.4f}")
    
    print("\n" + "="*60)
    print("✅ TODAS LAS PRUEBAS PASARON")
    print("="*60)

if __name__ == "__main__":
    main()