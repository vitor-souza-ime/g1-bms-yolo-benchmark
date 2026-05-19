#!/usr/bin/env python3
"""
BMS_Monitor.py
=================
Monitor de bateria do Unitree G1 via ROS2 — otimizado para experimentos
de comparação de consumo (ex: YOLO GPU vs CPU).


Roda no PC com o workspace Unitree sourced:
 source /opt/ros/jazzy/setup.bash
 source ~/unitree_ros2/install/setup.bash


Uso:
 python3 BMS_Monitor_04.py --condition baseline --samples 600
 python3 BMS_Monitor_04.py --condition yolo_gpu --samples 600
 python3 BMS_Monitor_04.py --condition yolo_cpu --samples 600


Notas:
 - O BMS do G1 publica a ~10 Hz → 600 amostras ≈ 60 s de experimento
 - Corrente, potência e energia em VALOR ABSOLUTO (descarga sempre positiva)
 - Timestamps wall-clock ISO armazenados no CSV para merge posterior
"""


DEFAULT_SAMPLES = 10000


# ─────────────────────────────────────────────────────────────
# Ambiente ROS2 + Unitree
# ─────────────────────────────────────────────────────────────
import os
import sys


os.environ.setdefault(
   'AMENT_PREFIX_PATH',
   '/home/u7/unitree_ros2/install/unitree_hg:'
   '/home/u7/unitree_ros2/install/unitree_go:'
   '/home/u7/unitree_ros2/install/unitree_api:'
   '/opt/ros/jazzy'
)
os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
os.environ.setdefault('ROS_LOCALHOST_ONLY', '0')


sys.path.insert(0, '/opt/ros/jazzy/lib/python3.12/site-packages')
for pkg in ('unitree_hg', 'unitree_go', 'unitree_api'):
   sys.path.insert(
       0,
       f'/home/u7/unitree_ros2/install/{pkg}/local/lib/python3.12/dist-packages'
   )


# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import csv
import math
import time
import argparse
from pathlib import Path
from datetime import datetime


try:
   import rclpy
   from rclpy.node import Node
   from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                           DurabilityPolicy, HistoryPolicy)
   from unitree_hg.msg import BmsState
except ImportError as e:
   print(f'ERRO ROS2: {e}')
   print('Verifique:')
   print('  source /opt/ros/jazzy/setup.bash')
   print('  source ~/unitree_ros2/install/setup.bash')
   sys.exit(1)


# ─────────────────────────────────────────────────────────────
# QoS
# ─────────────────────────────────────────────────────────────
QOS_BMS = QoSProfile(
   reliability=ReliabilityPolicy.RELIABLE,
   durability=DurabilityPolicy.VOLATILE,
   history=HistoryPolicy.KEEP_LAST,
   depth=1,
)


# ─────────────────────────────────────────────────────────────
# Helpers estatísticos
# ─────────────────────────────────────────────────────────────


def _percentile(sv: list, p: float) -> float:
   n = len(sv)
   if n == 1:
       return sv[0]
   idx = p / 100 * (n - 1)
   lo = int(idx)
   hi = lo + 1 if lo + 1 < n else lo
   return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])




def _stats(values: list) -> dict:
   if not values:
       return dict(min=0.0, max=0.0, mean=0.0, std=0.0,
                   p5=0.0, p25=0.0, p50=0.0, p75=0.0, p95=0.0)
   sv  = sorted(values)
   mu  = sum(values) / len(values)
   std = math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))
   return dict(
       min=sv[0], max=sv[-1], mean=mu, std=std,
       p5 =_percentile(sv,  5),
       p25=_percentile(sv, 25),
       p50=_percentile(sv, 50),
       p75=_percentile(sv, 75),
       p95=_percentile(sv, 95),
   )




def _energy_j(power_list: list, elapsed_list: list) -> float:
   """Integração trapezoidal usando timestamps reais (em segundos).
   Todos os valores de potência já são absolutos (positivos).
   """
   if len(power_list) < 2:
       return 0.0
   total = 0.0
   for i in range(1, len(power_list)):
       dt = elapsed_list[i] - elapsed_list[i - 1]
       total += 0.5 * (power_list[i] + power_list[i - 1]) * dt
   return total




def _boxplot(s: dict, width: int = 38) -> str:
   mn, mx = s['min'], s['max']
   if mx == mn:
       return '─' * width


   def pos(v):
       return round((v - mn) / (mx - mn) * (width - 1))


   buf = [' '] * width
   for i in range(pos(s['p5']), pos(s['p95']) + 1):
       buf[i] = '─'
   for i in range(pos(s['p25']), pos(s['p75']) + 1):
       buf[i] = '█'
   buf[pos(s['p50'])] = '┼'
   buf[pos(s['p5'])]  = '├'
   buf[pos(s['p95'])] = '┤'
   return ''.join(buf)




def print_stats(log: dict, condition: str, csv_path: str):
   W = 72


   def rule(c='─'):
       print(c * W)


   def row(label, key, unit):
       vals = log.get(key, [])
       if not vals:
           return
       s  = _stats(vals)
       bx = _boxplot(s)
       print(f"  {label:<16} μ={s['mean']:>8.3f}  σ={s['std']:>7.3f}"
             f"  P50={s['p50']:>8.3f}  {unit}")
       print(f"  {'':16} min={s['min']:>8.3f}  P25={s['p25']:>8.3f}"
             f"  P75={s['p75']:>8.3f}  max={s['max']:>8.3f}")
       print(f"  {'':16} P5={s['p5']:.3f} [{bx}] P95={s['p95']:.3f}")
       rule()


   n        = len(log.get('elapsed_s', []))
   duration = log['elapsed_s'][-1] if log.get('elapsed_s') else 0.0
   taxa_hz  = n / duration if duration > 0 else 0.0


   print()
   rule('═')
   print(f"  📊  RESUMO BMS — condição: {condition}")
   rule('═')
   print(f"  Amostras       : {n}")
   print(f"  Duração real   : {duration:.1f} s")
   print(f"  Taxa efetiva   : {taxa_hz:.1f} Hz")
   print(f"  CSV            : {csv_path}")
   rule()


   row('Tensão',     'voltage_V',  'V')
   row('Corrente',   'current_A',  'A')
   row('Potência',   'power_W',    'W')
   row('SOC',        'soc',        '%')
   row('Temp bat 1', 'temp1',      '°C')
   row('Temp bat 2', 'temp2',      '°C')


   # Energia integrada (todos os valores já são absolutos)
   pw  = log.get('power_W', [])
   ts  = log.get('elapsed_s', [])
   if pw and ts:
       ej      = _energy_j(pw, ts)
       ewh     = ej / 3600.0
       mean_p  = _stats(pw)['mean']
       print(f"  Potência média : {mean_p:>10.2f} W")
       print(f"  Energia (trap) : {ej:>10.1f} J  /  {ewh:.4f} Wh")


   if log.get('soc'):
       d     = log['soc'][0] - log['soc'][-1]
       arrow = '▼' if d >= 0 else '▲'
       print(f"  ΔSOC           : {log['soc'][0]:.0f}% → {log['soc'][-1]:.0f}%"
             f"  ({arrow} {abs(d):.1f} pp)")


   rule('═')
   print()




# ─────────────────────────────────────────────────────────────
# Node BMS
# ─────────────────────────────────────────────────────────────


class G1BMSMonitor(Node):


   def __init__(self, condition: str, max_samples: int, csv_path: str):
       super().__init__('g1_bms_monitor')


       self.condition   = condition
       self.max_samples = max_samples
       self.csv_path    = csv_path
       self.samples     = 0


       # Tempo de referência wall-clock (para elapsed consistente)
       self._t0_wall = time.monotonic()


       self.log = {
           'elapsed_s': [],
           'voltage_V': [],
           'current_A': [],
           'power_W':   [],
           'soc':       [],
           'temp1':     [],
           'temp2':     [],
       }


       self._csv  = open(csv_path, 'w', newline='')
       self._writer = csv.writer(self._csv)
       self._writer.writerow([
           'timestamp_iso',   # wall-clock ISO — para merge com Jetson CSV
           'elapsed_s',       # segundos desde início da coleta
           'condition',
           'voltage_V',
           'current_A',       # valor absoluto (A)
           'power_W',         # valor absoluto (W)
           'soc_pct',
           'soh_pct',
           'temp1_C',
           'temp2_C',
       ])


       self.create_subscription(
           BmsState, '/lf/bmsstate', self._cb, QOS_BMS)


       self.get_logger().info(
           f'BMS Monitor | condição={condition} | max_samples={max_samples}')


       print(f'\n{"#":>5} {"t(s)":>9} {"V":>7} {"I(A)":>9} '
             f'{"P(W)":>9} {"SOC%":>5} {"T1":>5} {"T2":>5}')
       print('─' * 62)


   def _cb(self, msg: BmsState):
       now_wall = time.monotonic()
       elapsed  = now_wall - self._t0_wall
       ts_iso   = datetime.now().isoformat()


       # Tensão: primeiro elemento do array, em mV → V
       try:
           voltage_v = msg.bmsvoltage[0] / 1000.0
       except Exception:
           voltage_v = 0.0


       # Corrente: mA → A, valor absoluto (descarga sempre positiva)
       try:
           current_a = abs(msg.current / 1000.0)
       except Exception:
           current_a = 0.0


       # Potência: valor absoluto
       power_w = voltage_v * current_a


       soc   = getattr(msg, 'soc', 0)
       soh   = getattr(msg, 'soh', 0)
       temps = [t for t in msg.temperature if t != 0]
       temp1 = temps[0] if len(temps) > 0 else 0.0
       temp2 = temps[1] if len(temps) > 1 else 0.0


       # CSV
       self._writer.writerow([
           ts_iso, f'{elapsed:.4f}', self.condition,
           f'{voltage_v:.4f}', f'{current_a:.4f}', f'{power_w:.4f}',
           soc, soh, temp1, temp2,
       ])
       self._csv.flush()


       # Acumula
       self.log['elapsed_s'].append(elapsed)
       self.log['voltage_V'].append(voltage_v)
       self.log['current_A'].append(current_a)
       self.log['power_W'].append(power_w)
       self.log['soc'].append(float(soc))
       self.log['temp1'].append(float(temp1))
       self.log['temp2'].append(float(temp2))


       self.samples += 1


       print(f'{self.samples:5d} {elapsed:9.2f} {voltage_v:7.2f} '
             f'{current_a:9.3f} {power_w:9.2f} {soc:5d} '
             f'{temp1:5.1f} {temp2:5.1f}')


       if self.samples >= self.max_samples:
           print(f'\n{"═"*62}')
           print(f'  {self.max_samples} amostras coletadas. Encerrando...')
           print(f'{"═"*62}')
           rclpy.shutdown()


   def destroy_node(self):
       try:
           self._csv.close()
       except Exception:
           pass
       if self.log['voltage_V']:
           print_stats(self.log, self.condition, self.csv_path)
       super().destroy_node()




# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main():
   parser = argparse.ArgumentParser(
       description='Monitor BMS — Unitree G1 Edu (experimentos YOLO GPU/CPU)'
   )
   parser.add_argument(
       '--condition', default='unknown',
       help='Condição experimental: baseline | yolo_gpu | yolo_cpu'
   )
   parser.add_argument(
       '--samples', type=int, default=DEFAULT_SAMPLES,
       help='Número de amostras BMS (~10 Hz → 600 ≈ 60 s). Default: 600'
   )
   args = parser.parse_args()


   output_dir = Path.home() / 'Documents' / 'bms_data'
   output_dir.mkdir(parents=True, exist_ok=True)


   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   csv_path  = output_dir / f'bms_{args.condition}_{timestamp}.csv'


   print('═' * 62)
   print('  🔋 BMS Monitor — Unitree G1 Edu')
   print(f'  Condição  : {args.condition}')
   print(f'  Amostras  : {args.samples}  (~{args.samples/10:.0f} s a 10 Hz)')
   print(f'  CSV       : {csv_path}')
   print('═' * 62)


   rclpy.init()
   node = G1BMSMonitor(args.condition, args.samples, str(csv_path))


   try:
       rclpy.spin(node)
   except KeyboardInterrupt:
       print('\nEncerrado pelo usuário.')
   finally:
       node.destroy_node()
       try:
           rclpy.shutdown()
       except Exception:
           pass
       print('═' * 62)
       print('  Concluído.')
       print('═' * 62)




if __name__ == '__main__':
   main()

