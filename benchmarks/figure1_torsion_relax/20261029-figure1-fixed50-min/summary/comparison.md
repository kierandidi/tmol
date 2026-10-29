### torsion: median over systems (s/structure; speedup = PyRosetta CPU 1 thread / tmol, same mode and step budget)

| series | n | 50-step (s) | 200-step (s) | 50/200 | PyRos speedup, 50-step | PyRos speedup, 200-step |
|---|---:|---:|---:|---:|---:|---:|
| PyRosetta CPU B1 | 27 | 26.22 | 110.30 | 0.26 | 1.0 | 1.0 |
| tmol CPU B1 (32 thr) | 27 | 17.26 | 36.15 | 0.50 | 1.5 | 3.2 |
| tmol GPU B1 | 27 | 8.07 | 9.26 | 0.88 | 3.4 | 12.0 |
| tmol GPU B10 | 27 | 2.04 | 2.57 | 0.86 | 12.1 | 44.9 |
| tmol GPU B100 | 27 | 1.43 | 1.91 | 0.82 | 17.0 | 59.7 |
| tmol GPU B1000 | 20 | 0.79 | 1.98 | 0.79 | 20.0 | 58.1 |

### cartesian: median over systems (s/structure; speedup = PyRosetta CPU 1 thread / tmol, same mode and step budget)

| series | n | 50-step (s) | 200-step (s) | 50/200 | PyRos speedup, 50-step | PyRos speedup, 200-step |
|---|---:|---:|---:|---:|---:|---:|
| PyRosetta CPU B1 | 27 | 27.79 | 404.55 | 0.07 | 1.0 | 1.0 |
| tmol CPU B1 (32 thr) | 27 | 13.58 | 33.14 | 0.42 | 2.1 | 10.7 |
| tmol GPU B1 | 27 | 4.78 | 6.17 | 0.78 | 5.9 | 58.2 |
| tmol GPU B10 | 27 | 1.69 | 2.13 | 0.84 | 15.1 | 175.2 |
| tmol GPU B100 | 27 | 1.36 | 1.74 | 0.84 | 20.5 | 234.4 |
| tmol GPU B1000 | 21 | 0.85 | 1.06 | 0.82 | 22.4 | 231.1 |

### torsion / Cartesian time ratio at 50 steps (median over systems)

| series | torsion/Cartesian |
|---|---:|
| PyRosetta CPU B1 | 0.92 |
| tmol CPU B1 (32 thr) | 1.30 |
| tmol GPU B1 | 1.71 |
| tmol GPU B10 | 1.20 |
| tmol GPU B100 | 1.06 |
| tmol GPU B1000 | 1.05 |
