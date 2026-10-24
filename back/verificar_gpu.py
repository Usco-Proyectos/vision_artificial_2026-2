import torch
from app.models_loader import cargar_modelos

print("="*50)
print("VERIFICACIÓN DE DISPOSITIVO (DEVICE)")
print("="*50)

# 1. ¿PyTorch ve la GPU?
print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
print(f"Nombre de la GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Ninguna'}")

# 2. Cargar modelos y verificar dónde están
print("\nCargando modelos (esto puede tardar unos segundos)...")
modelos = cargar_modelos()

print(f"\n📍 M1 (Ojos) está en: {next(modelos.m1.parameters()).device}")
print(f"📍 M2 (Boca) está en: {next(modelos.m2.parameters()).device}")
print(f"📍 M3 (LSTM) está en: {next(modelos.m3.parameters()).device}")

print("\n" + "="*50)
print("Si todos dicen 'cuda:0', ¡tus modelos están en la GPU! ✅")
print("Si dicen 'cpu', hay un problema en la carga. ❌")