#!/usr/bin/env python
"""Benchmark du modèle SEAPODYM-LMTL original (C++/Fortran).

Compare les performances du modèle original avec notre implémentation Python.
"""

import subprocess
import time
from pathlib import Path

import pandas as pd

# Configuration
# Configuration
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "data"

SEAPODYM_BIN = "/Users/adm-lehodey/Documents/Workspace/Projects/seapodym-project/seapodym-lmtl/build/bin/seapodym-lmtl"
CONFIG_XML = DATA_DIR / "LMTL_Pacific_Run" / "pacific_run.xml"
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
        str(CONFIG_XML),
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

# Résumé et génération du Summary
SUMMARY_DIR = (
    Path(__file__).parent.parent / "summary"
    if "__file__" in globals()
    else Path.cwd().parent / "summary"
)
SUMMARY_DIR.mkdir(exist_ok=True)

FIGURE_PREFIX = "fig_05d_benchmark_seapodym"
summary_filename = f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
summary_path = SUMMARY_DIR / summary_filename

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

    # Génération du summary détaillé
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NOTEBOOK 05D: BENCHMARK SEAPODYM-LMTL ORIGINAL (C++/FORTRAN)\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATE: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        f.write("OBJECTIF:\n")
        f.write("-" * 80 + "\n")
        f.write("Mesurer le temps d'exécution du modèle SEAPODYM-LMTL original (C++/Fortran)\n")
        f.write("pour servir de référence de performance lors de la comparaison avec\n")
        f.write("l'implémentation Python (SeapoPym DAG).\n\n")

        f.write("CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Binaire SEAPODYM          : {SEAPODYM_BIN}\n")
        f.write(f"Fichier configuration XML : {CONFIG_XML}\n")
        f.write(f"Groupe fonctionnel        : {GROUP}\n")
        f.write(f"Nombre de processus MPI   : {N_PROCS}\n\n")

        f.write("DESCRIPTION DU MODÈLE ORIGINAL:\n")
        f.write("-" * 80 + "\n")
        f.write("SEAPODYM-LMTL est le modèle de référence développé en C++/Fortran par SPC.\n")
        f.write("Il calcule la dynamique des niveaux trophiques bas et moyens (LMTL)\n")
        f.write("à partir de forçages environnementaux (NPP, température, courants).\n\n")

        f.write("ÉTAPES D'EXÉCUTION:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Calcul Production (-P) : Calcule la production primaire transformée\n")
        f.write("   en biomasse micronectonique via les cohortes d'âge.\n")
        f.write("2. Calcul Biomasse (-B)   : Calcule la biomasse finale avec transport\n")
        f.write("   (advection + diffusion) et mortalité.\n\n")

        f.write("TEMPS D'EXÉCUTION:\n")
        f.write("-" * 80 + "\n")
        for name, t in timings.items():
            f.write(f"{name:<20} : {t:.2f} s ({t / 60:.2f} min)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'TOTAL':<20} : {total:.2f} s ({total / 60:.2f} min)\n\n")

        f.write("UTILISATION:\n")
        f.write("-" * 80 + "\n")
        f.write("Ces temps servent de référence pour évaluer les performances de SeapoPym.\n")
        f.write("Un facteur d'accélération ou de ralentissement peut être calculé en\n")
        f.write("comparant avec les temps du notebook 05b (simulation SeapoPym).\n\n")

        f.write("FICHIERS GÉNÉRÉS:\n")
        f.write("-" * 80 + "\n")
        f.write("- Sortie SEAPODYM (Zarr référence) dans data/LMTL_Pacific_Run/\n")
        f.write(f"- {summary_filename} (ce fichier)\n\n")

        f.write("=" * 80 + "\n")

    print(f"\n✅ Résumé sauvegardé: {summary_path}")
else:
    print("❌ Aucun benchmark réussi")

print("=" * 70)
