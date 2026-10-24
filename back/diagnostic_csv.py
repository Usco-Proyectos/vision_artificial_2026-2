# diagnostic_temporal.py
import pandas as pd

df = pd.read_csv("data/prueba.csv")
df = df[df['somnolencia'].notna()].copy()

print("=" * 60)
print("ESTRUCTURA TEMPORAL DEL CSV")
print("=" * 60)

print(f"\nTotal de filas con somnolencia: {len(df)}")

# Agrupar por video
if 'video' in df.columns:
    videos = df.groupby('video')
    print(f"\nVideos únicos: {videos.ngroups}")
    print("\nFrames por video:")
    frames_per_video = videos.size()
    print(frames_per_video.describe())
    print("\nPrimeros 10 videos:")
    print(frames_per_video.head(10))
    
    # Verificar si hay columna 'fotograma'
    if 'fotograma' in df.columns:
        print("\n" + "=" * 60)
        print("VERIFICACIÓN DE ORDEN TEMPORAL")
        print("=" * 60)
        for video_name, group in list(videos)[:3]:
            print(f"\nVideo: {video_name}")
            print(f"  Frames: {len(group)}")
            print(f"  Rango fotogramas: {group['fotograma'].min()} - {group['fotograma'].max()}")
            print(f"  ¿Están ordenados? {group['fotograma'].is_monotonic_increasing}")
else:
    print("\n❌ No hay columna 'video'")

# Verificar columna 'sujeto'
if 'sujeto' in df.columns:
    sujetos = df.groupby('sujeto')
    print(f"\n\nSujetos únicos: {sujetos.ngroups}")
    print("Frames por sujeto:")
    print(sujetos.size().describe())