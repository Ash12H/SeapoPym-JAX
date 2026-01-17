#!/usr/bin/env python
"""Benchmark du modèle SEAPODYM-LMTL original (C++/Fortran).

Compare les performances du modèle original avec notre implémentation Python.
"""

import subprocess
import time
from pathlib import Path

# Configuration
SEAPODYM_BIN = "/Users/adm-lehodey/Documents/Workspace/Projects/seapodym-project/seapodym-lmtl/build/bin/seapodym-lmtl"
CONFIG_XML = "/Users/adm-lehodey/Documents/Workspace/Projects/seapopym-message/data/article/data/LMTL_Pacific_Run/pacific_run.xml"
GROUP = "D1N1"  # Functional group to run
N_PROCS = 1

# Vérifier que les fichiers existent
if not Path(SEAPODYM_BIN).exists():
    print(f"❌ Binaire non trouvé: {SEAPODYM_BIN}")
    exit(1)

if not Path(CONFIG_XML).exists():
    print(f"❌ Config XML non trouvée: {CONFIG_XML}")
    exit(1)

print("=" * 70)
print("BENCHMARK SEAPODYM-LMTL ORIGINAL")
print("=" * 70)
print(f"Binaire: {SEAPODYM_BIN}")
print(f"Config:  {CONFIG_XML}")
print(f"Groupe:  {GROUP}")
print(f"Procs:   {N_PROCS}")
print("=" * 70)


def run_seapodym(mode: str, description: str) -> float:
    """Exécute SEAPODYM-LMTL et retourne le temps d'exécution."""
    cmd = [
        "mpirun",
        "-mca",
        "btl",
        "^openib",
        "-np",
        str(N_PROCS),
        SEAPODYM_BIN,
        mode,  # -P ou -B
        "-G",
        GROUP,
        "-V",
        "error",
        CONFIG_XML,
    ]

    print(f"\n--- {description} ---")
    print(f"Commande: {' '.join(cmd)}")
    print("Exécution en cours...")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"❌ Erreur (code {result.returncode})")
        if result.stderr:
            print(f"Stderr: {result.stderr[:500]}")
        return -1

    print(f"✅ Terminé en {elapsed:.2f}s")
    return elapsed


# Exécution des benchmarks
timings = {}

# 1. Production (-P)
t_production = run_seapodym("-P", "Calcul Production")
if t_production > 0:
    timings["production"] = t_production

# 2. Biomasse (-B)
t_biomass = run_seapodym("-B", "Calcul Biomasse")
if t_biomass > 0:
    timings["biomass"] = t_biomass

# Résumé
print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)

if timings:
    total = sum(timings.values())
    print(f"{'Étape':<20} {'Temps (s)':<15}")
    print("-" * 35)
    for name, t in timings.items():
        print(f"{name:<20} {t:<15.2f}")
    print("-" * 35)
    print(f"{'TOTAL':<20} {total:<15.2f}")

    # Sauvegarder les résultats
    results_file = Path(CONFIG_XML).parent / "benchmark_seapodym_original.txt"
    with open(results_file, "w") as f:
        f.write("SEAPODYM-LMTL Original Benchmark\n")
        f.write("=" * 40 + "\n")
        for name, t in timings.items():
            f.write(f"{name}: {t:.2f}s\n")
        f.write(f"TOTAL: {total:.2f}s\n")
    print(f"\n✅ Résultats sauvegardés: {results_file}")
else:
    print("❌ Aucun benchmark réussi")

print("=" * 70)
