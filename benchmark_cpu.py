"""
benchmark_cpu.py
================
Benchmark contínuo — CPU only — yolo26x
Unitree G1 EDU (Jetson Orin NX / JetPack 5.x)


Uso:
   python /home/unitree/Documents/Camera/YOLO_CPU_ONLY.py
"""


import time
import json
import statistics
import platform
from pathlib import Path
from datetime import datetime


import numpy as np
import cv2
import torch
import pyrealsense2 as rs
from ultralytics import YOLO


# ─────────────────────────────────────────────
# ✏️  CONFIGURAÇÃO — edite aqui
# ─────────────────────────────────────────────


MODELO     = "yolo26x.pt"
DEVICE     = "cpu"
N_AMOSTRAS = 150          # ← defina o número de frames a medir
N_WARMUP   = 5
OUTPUT_DIR = "/home/unitree/Documents"


# ─────────────────────────────────────────────
# CÂMERA
# ─────────────────────────────────────────────


def iniciar_camera():
   pipeline = rs.pipeline()
   config   = rs.config()
   config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
   config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
   try:
       pipeline.start(config)
   except RuntimeError as e:
       if "errno=16" in str(e) or "busy" in str(e).lower():
           print("  ⚠️  Câmera ocupada — aguardando 3s e tentando novamente...")
           time.sleep(3)
           pipeline.start(config)
       else:
           raise
   align = rs.align(rs.stream.color)
   print("  📷 Câmera RealSense iniciada")
   return pipeline, align




def capturar_frame(pipeline, align):
   frames  = pipeline.wait_for_frames()
   aligned = align.process(frames)
   color_frame = aligned.get_color_frame()
   depth_frame = aligned.get_depth_frame()
   if not color_frame or not depth_frame:
       return None, None
   return np.asanyarray(color_frame.get_data()), depth_frame




def distancia_mediana(depth_frame, cx, cy, janela=10):
   x1 = max(0, cx - janela);  x2 = min(640, cx + janela)
   y1 = max(0, cy - janela);  y2 = min(480, cy + janela)
   arr = np.asanyarray(depth_frame.get_data()).astype(float)
   reg = arr[y1:y2, x1:x2]
   reg[reg == 0] = np.nan
   if np.all(np.isnan(reg)):
       return 0.0
   return float(np.nanmedian(reg)) * depth_frame.get_units()


# ─────────────────────────────────────────────
# ESTATÍSTICAS
# ─────────────────────────────────────────────


def calcular_estatisticas(tempos_ms):
   arr = sorted(tempos_ms)
   n   = len(arr)


   def pct(p):
       idx = (p / 100) * (n - 1)
       lo, hi = int(idx), min(int(idx) + 1, n - 1)
       return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)


   media = statistics.mean(arr)
   dp    = statistics.stdev(arr) if n > 1 else 0.0
   return {
       "n_amostras" : n,
       "media_ms"   : round(media, 2),
       "mediana_ms" : round(pct(50), 2),
       "min_ms"     : round(arr[0], 2),
       "max_ms"     : round(arr[-1], 2),
       "dp_ms"      : round(dp, 2),
       "cv_pct"     : round((dp / media * 100) if media else 0, 2),
       "p25_ms"     : round(pct(25), 2),
       "p75_ms"     : round(pct(75), 2),
       "p95_ms"     : round(pct(95), 2),
       "p99_ms"     : round(pct(99), 2),
       "fps_media"  : round(1000 / media, 1) if media else 0,
       "fps_p95"    : round(1000 / pct(95), 1),
       "fps_p99"    : round(1000 / pct(99), 1),
   }


# ─────────────────────────────────────────────
# RELATÓRIO
# ─────────────────────────────────────────────


LINHA  = "─" * 70
LINHA2 = "═" * 70


def imprimir_relatorio(stats):
   print(f"\n{LINHA2}")
   print(f"  {MODELO} @ {DEVICE.upper()}")
   print(LINHA2)
   print(f"  Amostras   : {stats['n_amostras']}")
   print(f"  Média      : {stats['media_ms']} ms  →  {stats['fps_media']} FPS")
   print(f"  Mediana    : {stats['mediana_ms']} ms")
   print(f"  Mín / Máx  : {stats['min_ms']} ms / {stats['max_ms']} ms")
   print(f"  DP / CV    : {stats['dp_ms']} ms / {stats['cv_pct']}%")
   print(f"  P25 / P75  : {stats['p25_ms']} ms / {stats['p75_ms']} ms")
   print(f"  P95 / P99  : {stats['p95_ms']} ms / {stats['p99_ms']} ms")
   print(f"  FPS P95    : {stats['fps_p95']}    FPS P99: {stats['fps_p99']}")
   print(LINHA2)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
   print(LINHA2)
   print(f"  BENCHMARK CPU CONTÍNUO — {MODELO}")
   print(f"  N_AMOSTRAS={N_AMOSTRAS}  |  N_WARMUP={N_WARMUP}")
   print(f"  Plataforma : {platform.platform()}")
   print(f"  Python     : {platform.python_version()}")
   print(f"  PyTorch    : {torch.__version__}")
   print(f"  CUDA       : {torch.cuda.is_available()} (não usado neste script)")
   print(LINHA2)


   pipeline, align = iniciar_camera()


   print(f"\n  📦 Carregando {MODELO} → CPU...", end=" ", flush=True)
   model = YOLO(MODELO)
   model.to(DEVICE)
   print("OK")


   # Warmup
   print(f"  🔥 Warmup ({N_WARMUP} frames)...", end=" ", flush=True)
   ok = 0
   while ok < N_WARMUP:
       img, _ = capturar_frame(pipeline, align)
       if img is None:
           continue
       model(img, verbose=False)
       ok += 1
   print("OK")


   historico = []   # guarda stats de todas as rodadas
   rodada    = 0


   print(f"\n  🚀 Iniciando benchmark contínuo — Ctrl+C para parar\n")


   try:
       tempos = []
       ultimo_img = ultimo_depth = None


       print(f"  ⏱️  Medindo {N_AMOSTRAS} frames: ", end="", flush=True)


       while len(tempos) < N_AMOSTRAS:
           img, depth_frame = capturar_frame(pipeline, align)
           if img is None:
               continue


           ultimo_img   = img.copy()
           ultimo_depth = depth_frame


           t0 = time.perf_counter()
           results = model(img, verbose=False)
           t1 = time.perf_counter()
           tempos.append((t1 - t0) * 1000)


           n = len(tempos)
           if N_AMOSTRAS >= 10 and n % (N_AMOSTRAS // 10) == 0:
               print(f"{n}..", end="", flush=True)


       print(" ✓")


       stats = calcular_estatisticas(tempos)
       stats["timestamp"] = datetime.now().isoformat()


       imprimir_relatorio(stats)


       # Salva imagem anotada
       if ultimo_img is not None:
           img_out = ultimo_img.copy()
           for result in results:
               for box in result.boxes:
                   conf = float(box.conf[0])
                   if conf < 0.5:
                       continue
                   classe = model.names[int(box.cls[0])]
                   x1, y1, x2, y2 = map(int, box.xyxy[0])
                   cx, cy = (x1+x2)//2, (y1+y2)//2
                   dist = distancia_mediana(ultimo_depth, cx, cy)
                   cor  = (0,255,0) if conf>=0.8 else (0,165,255) if conf>=0.6 else (0,0,255)
                   cv2.rectangle(img_out, (x1,y1), (x2,y2), cor, 2)
                   label = f"{classe} {conf:.0%} {dist:.1f}m"
                   (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                   cv2.rectangle(img_out, (x1,y1-th-8), (x1+tw,y1), cor, -1)
                   cv2.putText(img_out, label, (x1,y1-4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


           cv2.putText(img_out, f"CPU | FPS: {stats['fps_media']}", (10,25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
           cv2.putText(img_out, f"Inf: {stats['media_ms']:.0f}ms | {N_AMOSTRAS} frames", (10,50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)


           fname = Path(__file__).parent / "benchmark_cpu_resultado.jpg"
           cv2.imwrite(str(fname), img_out)
           print(f"  💾 Imagem salva: {fname}")


       # Salva JSON
       with open(f"{OUTPUT_DIR}/benchmark_cpu_resultado.json", "w") as f:
           json.dump({
               "modelo"    : MODELO,
               "device"    : DEVICE,
               "n_amostras": N_AMOSTRAS,
               "resultado" : stats
           }, f, indent=2)


       print(f"\n  ✅ Resultados salvos em {OUTPUT_DIR}/benchmark_cpu_resultado.json")


   except KeyboardInterrupt:
       print(f"\n\n  ⏹️  Benchmark interrompido.")


   finally:
       pipeline.stop()
       print("  📷 Câmera encerrada")




if __name__ == "__main__":
   main()

