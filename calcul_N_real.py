#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from distillation_multicomposants import Compound, ThermodynamicPackage, ShortcutDistillation

# Créer le système BTX
compounds = [Compound(n) for n in ['benzene', 'toluene', 'o-xylene']]
thermo = ThermodynamicPackage(compounds)

# Paramètres
F = 100.0
z_F = np.array([0.333, 0.333, 0.334])
P = 101325

# Dimensionnement
shortcut = ShortcutDistillation(thermo, F, z_F, P)
results = shortcut.complete_shortcut_design(
    recovery_LK_D=0.95,
    recovery_HK_B=0.95,
    R_factor=1.3,
    q=1.0,
    efficiency=0.70
)

# Afficher les résultats
print("\n" + "="*60)
print("NOMBRE RÉEL D'ÉTAGES (PLATEAUX)")
print("="*60)
print(f"\nN_min (Fenske):           {results['N_min']:.2f} plateaux")
print(f"N théorique (Gilliland):  {results['N_theoretical']:.2f} plateaux")
print(f"Efficacité:               {results['efficiency']*100:.1f}%")
print(f"\nN_réel = N_théorique / efficacité")
print(f"N_réel = {results['N_theoretical']:.2f} / {results['efficiency']}")
print(f"\n>>> N_réel = {results['N_real']} plateaux\n")
print("="*60)
