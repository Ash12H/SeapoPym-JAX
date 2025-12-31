# Introduction

## Contexte : Modéliser les Niveaux Trophiques Intermédiaires (LMTL)

La gestion durable des ressources marines et l'étude de l'impact climatique nécessitent une compréhension fine des niveaux trophiques intermédiaires (LMTL - Low and Mid-Trophic Levels), maillon crucial transférant l'énergie de la production primaire vers les grands prédateurs (thons, mammifères marins). Ces niveaux englobent le mésozooplancton (0.2–2 mm) et le micronecton (2–20 cm), incluant poissons mésopélagiques et céphalopodes [Sieburth et al., 1978; Brodeur et al., 2004].

Le modèle **SEAPODYM-LMTL** [Lehodey et al., 2008; 2015] s'est imposé comme une référence pour simuler la dynamique spatio-temporelle de ces groupes fonctionnels à l'échelle globale. Historiquement implémenté en C++, ce modèle couple étroitement les équations de transport physique (Advection-Diffusion sur grille) avec les réactions biologiques (croissance, prédation).

## Limites de l'Approche Actuelle et Travaux Précédents

Bien que performante pour la production opérationnelle (ex: Copernicus Marine Service), l'architecture monolithique du C++ freine l'innovation scientifique. La complexité d'ajouter une nouvelle interaction biologique ou de modifier le schéma numérique est prohibitive.

Une première tentative de modernisation, **SeapoPym v0.3** [Lehodey Jr. et al., en prép.], a proposé une réécriture en Python utilisant l'écosystème scientifique moderne (xarray, numba). Cette version a introduit un cadre d'estimation de paramètres stochastique (Algorithmes Génétiques), mais s'est limitée à une formulation **0D** (sans transport spatial) pour contourner la difficulté de coupler dynamiquement le transport et la biologie dans un code impératif.

## Proposition : Une Architecture DAG Unifiée

Cet article présente la nouvelle génération du moteur de modélisation (v1.0), qui résout le dilemme "0D vs 3D" en adoptant une architecture fondée sur les **Graphes Dirigés Acycliques (DAG)**.

Contrairement aux approches précédentes, nous ne traitons plus le transport comme une "grille" et la biologie comme une "boucle", mais unifions les deux concepts sous forme de flux de données. Cette approche permet :
1.  **Réintégration du Transport** : L'advection et la diffusion deviennent de simples opérateurs fonctionnels au sein du graphe, traites à égalité avec la mortalité ou la croissance.
2.  **Rigueur Mathématique** : L'architecture impose une résolution explicite et acyclique, garantissant la causalité et la conservation de la masse.
3.  **Modularité Totale** : Le modélisateur peut assembler des "briques" biologiques (définies par les équations de dynamique de population standard) sans se soucier de l'orchestration informatique.

Nous validons cette approche à travers une série de benchmarks démontrant que l'overhead du graphe est négligeable face aux gains de flexibilité et de précision scientifique.
