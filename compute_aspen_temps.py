#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from distillation_multicomposants import Compound, ThermodynamicPackage, ShortcutDistillation

# System
compound_names = ['benzene', 'toluene', 'o-xylene']
compounds = [Compound(n) for n in compound_names]
thermo = ThermodynamicPackage(compounds)

F = 100.0
z_F = np.array([0.333, 0.333, 0.334])
P = 101325

# Run shortcut design
shortcut = ShortcutDistillation(thermo, F, z_F, P)
results = shortcut.complete_shortcut_design(
    recovery_LK_D=0.95,
    recovery_HK_B=0.95,
    R_factor=1.3,
    q=1.0,
    efficiency=0.70
)

# Get compositions
x_D = results['x_D']
x_B = results['x_B']

# Bubble temperatures at column pressure
T_top_K, y_top = thermo.bubble_temperature(P, x_D)
T_bottom_K, y_bottom = thermo.bubble_temperature(P, x_B)

# Feed bubble temp (saturated liquid q=1)
T_feed_K, y_feed = thermo.bubble_temperature(P, z_F)

print('\n=== Températures recommandées pour Aspen (approximatives) ===')
print(f"Pression colonne (P): {P/101325:.3f} atm  ({P:.0f} Pa)")
print(f"Composition distillat x_D: {x_D}")
print(f"Composition résidu   x_B: {x_B}\n")
print(f"Température tête (bulbe @ P, distillat): {T_top_K - 273.15:.2f} °C")
print(f"Température fond (bulbe @ P, résidu):  {T_bottom_K - 273.15:.2f} °C")
print(f"Température alimentation (bulbe @ P):  {T_feed_K - 273.15:.2f} °C\n")

print('Recommandations pour Aspen Plus:')
print(' - Utiliser la même pression de colonne (1 atm) dans les blocs (Condenser/Reboiler).')
print(' - Renseigner la température de condenseur = Température tête (°C)')
print(' - Renseigner la température de rebouillage = Température fond (°C)')
print(' - Pour la température interne par étage, laissez Aspen calculer l\'équilibre par étage (MESH).')
print(" - Si besoin d'initialiser, fournir un profil linéaire entre T_tete et T_fond ou utiliser la 'Feed' temperature comme point de départ.")
print(' - Choisir un modèle thermodynamique approprié (ex: NRTL ou UNIQUAC pour mélanges non-idéaux; Peng-Robinson pour vapeur/haute P).')
print('\nNote: Ces températures sont des estimations basées sur l\'équilibre bulle à la pression indiquée; Aspen résoudra les températures internes avec son solveur MESH.')

# Print single-line values for quick copy/paste
print('\n# Copy/paste values:')
print(f'T_top_C={T_top_K - 273.15:.2f}')
print(f'T_bottom_C={T_bottom_K - 273.15:.2f}')
print(f'T_feed_C={T_feed_K - 273.15:.2f}')

# End
