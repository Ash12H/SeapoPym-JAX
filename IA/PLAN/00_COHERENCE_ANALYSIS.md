# Analyse de Cohérence - Design SeapoPym-JAX

**Date** : 2026-01-25
**Documents analysés** : Axes 1 à 5 (`IA/DESIGN/`)

---

## Synthèse Exécutive

L'architecture proposée est globalement **cohérente et bien structurée**. Les 5 axes couvrent les aspects essentiels d'une migration JAX. Cependant, l'analyse révèle **3 tensions critiques** et **7 zones d'ombre** qui nécessitent clarification avant implémentation.

---

## 1. Tensions Critiques (Bloquantes)

### T1. Chunking vs Auto-Diff : Deux modes incompatibles

**Axes concernés** : 3 (Engine) ↔ 5 (Auto-Diff)

**Contradiction** :

- L'Axe 3 définit un _Chunked Runner_ qui écrit sur disque entre les chunks temporels
- L'Axe 5 explicite que cela "brise la chaîne de dérivation" et impose un **seul appel JAX** pour la calibration

**Impact** : Ce sont deux architectures d'exécution fondamentalement différentes.

**Questions non résolues** :

1. Comment l'utilisateur choisit-il entre "simulation pure" et "calibration" ?
2. Y a-t-il une API unifiée ou deux runners distincts ?
3. Comment le système détermine-t-il automatiquement quelle stratégie appliquer ?
4. Le mode calibration nécessite-t-il une réécriture du graphe ou juste un paramètre ?

**Recommandation** : Définir explicitement une abstraction `ExecutionMode` avec deux implémentations (`StreamingRunner`, `GradientRunner`) partageant la même `step_fn`.

---

### T2. Dual Backend : `vmap` inexistant en NumPy

**Axes concernés** : 1 (Blueprint) ↔ 3 (Engine)

**Contradiction** :

- L'Axe 1 stipule que les noyaux biologiques sont écrits en **0D (pointwise)** et le Compilateur utilise `jax.vmap` pour les étendre aux dimensions
- L'Axe 3 propose un mode NumPy avec une "boucle `for` Python" en remplacement de `scan`

**Problème** : NumPy n'a pas d'équivalent à `vmap`. Comment la même fonction 0D peut-elle fonctionner sur les deux backends ?

**Options implicites non documentées** :

1. **Fonctions écrites vectorisées nativement** (contradiction avec "0D pointwise")
2. **Double implémentation par fonction** (explosion de la maintenance)
3. **Couche d'abstraction vmap-like** (performance NumPy catastrophique)

**Recommandation** : Clarifier si les fonctions sont vraiment 0D ou si elles doivent être écrites avec broadcasting NumPy natif (ce qui les rend automatiquement compatibles JAX).

---

### T3. Layout Canonique : Notation incohérente

**Axes concernés** : 1 ↔ 2

**Incohérence factuelle** :

- Axe 1 (ligne 81) : `(T, C, Z, Y, X)`
- Axe 2 (ligne 27) : `(Time, Z, Y, X, Cohort)`
- Axe 2 (ligne 44) : `(Time, Cohort, Depth, Lat, Lon)`

**Impact** : L'ordre des dimensions affecte directement :

- La contiguïté mémoire (C-order vs F-order)
- La performance des opérations JAX (coalescence mémoire GPU)
- L'écriture des kernels de transport spatial

**Recommandation** : Fixer **un seul ordre canonique** dans un document de référence unique. Suggestion basée sur les patterns JAX standards : `(Batch, Time, Channel, Spatial...)` → `(Ensemble, Time, Cohort, Z, Y, X)`.

---

## 2. Zones d'Ombre (Clarifications nécessaires)

### Z1. Responsabilité des masques (Axe 2)

Le prétraitement remplace les NaN par 0.0 et l'utilisateur fournit un masque.

**Questions ouvertes** :

- Qui applique le masque ? Chaque fonction du registre ? Le moteur après chaque step ?
- Comment éviter que les 0.0 polluent les calculs de moyenne/réduction ?
- Les gradients sont-ils masqués automatiquement ?

---

### Z2. Registre bi-backend (Axe 1)

Le décorateur `@functional` enregistre une fonction, mais le dual backend nécessite potentiellement deux implémentations.

**Questions ouvertes** :

- Enregistre-t-on une seule fonction polymorphe ou deux variantes explicites ?
- Comment le moteur résout-il `"biol:growth"` → version JAX vs NumPy ?
- Syntaxe de déclaration pour une fonction avec deux implémentations ?

---

### Z3. Noyaux Grid vs Pointwise (Axes 1, 4)

L'Axe 1 mentionne les noyaux "Grid" (transport) nécessitant les voisins spatiaux.
L'Axe 4 interdit le sharding spatial en V1.

**Questions ouvertes** :

- Comment écrire un noyau de transport en "0D + vmap" si le transport nécessite les voisins ?
- Comment le système distingue-t-il un kernel pointwise d'un kernel stencil ?
- Le décorateur `@functional(core_dims=["spatial"])` suffit-il ? Comment est-il exploité ?

---

### Z4. Dimensions dynamiques (Axe 2)

L'Axe 2 pose la question du streaming temporel infini mais n'y répond pas.

**Question ouverte** : Si le temps est "infini", comment `jax.jit` (qui fige les shapes) peut-il fonctionner ?

**Réponse implicite probable** : Le Chunked Runner résout ce problème (shape fixe par chunk). Mais cela mérite d'être explicité.

---

### Z5. Granularité du Checkpointing (Axe 5)

`jax.remat` est mentionné avec "checkpoints tous les N pas".

**Questions ouvertes** :

- Quelle valeur de N ? Configurable ?
- Lien avec la taille des chunks de l'Axe 3 ?
- Heuristique automatique basée sur la VRAM disponible ?

---

### Z6. I/O Asynchrone et GIL (Axe 4)

L'Axe 4 mentionne `threading` pour paralléliser I/O et calcul.

**Clarification nécessaire** :

- Le GIL Python bloque-t-il réellement l'écriture pendant le calcul JAX ?
- Réponse probable : Non, car JAX libère le GIL pendant l'exécution XLA. Mais cela mérite documentation.

---

### Z7. Validation du Registre (Axe 1)

La validation Xarray/Pint est mentionnée, mais pas la validation du registre lui-même.

**Questions ouvertes** :

- Que se passe-t-il si `"biol:growth"` n'existe pas dans le registre ?
- Validation des signatures (nombre d'arguments, types) ?
- À quel moment cette validation intervient-elle (parsing YAML, compilation, exécution) ?

---

## 3. Points de Cohérence Validés

| Lien Inter-Axes | Description                                                           | Statut   |
| --------------- | --------------------------------------------------------------------- | -------- |
| 1 → 2           | Format Split YAML → Compilateur lit les métadonnées                   | Cohérent |
| 2 → 3           | Transpose Xarray → Arrays C-contiguous pour JAX                       | Cohérent |
| 3 → 4           | Chunked Runner → I/O Asynchrone via threading                         | Cohérent |
| 4 → 5           | Batch Parallelism (`vmap`) → Compatible avec `grad`                   | Cohérent |
| 1 → 5           | Flag `trainable: true` dans config → Paramètres pour `value_and_grad` | Cohérent |

---

## 4. Matrice de Dépendances

```
Axe 1 (Blueprint)
    │
    ├──→ Axe 2 (Compiler) : fournit le graphe déclaratif
    │        │
    │        └──→ Axe 3 (Engine) : produit les arrays prêts pour JAX
    │                 │
    │                 ├──→ Axe 4 (Parallelism) : le Chunked Runner permet vmap/sharding
    │                 │
    │                 └──→ Axe 5 (AutoDiff) : tension avec le mode streaming
    │
    └──→ Axe 5 (AutoDiff) : flag trainable déclaré dans config
```

---

## 5. Recommandations Prioritaires

### Priorité 1 : Résoudre T1 (Chunking vs Gradient)

Documenter explicitement les deux modes d'exécution :

```yaml
# run.yaml
execution:
  mode: "streaming" # ou "gradient"
  chunk_size: 365 # ignoré si mode=gradient
  checkpoint_interval: 50 # utilisé si mode=gradient
```

### Priorité 2 : Résoudre T3 (Layout Canonique)

Créer un fichier `CONVENTIONS.md` avec l'ordre canonique officiel, par exemple :

```
CANONICAL_DIMS = ("ensemble", "time", "cohort", "depth", "lat", "lon")
```

### Priorité 3 : Clarifier T2 (Dual Backend)

Documenter la stratégie choisie :

- Option A : Les fonctions sont écrites en NumPy vectorisé natif (compatible JAX automatiquement)
- Option B : Le registre accepte deux implémentations par clé

---

## 6. Conclusion

Le design est **mûr pour passer en phase d'implémentation** à condition de :

1. **Trancher** les 3 tensions critiques (T1, T2, T3)
2. **Documenter** les réponses aux 7 zones d'ombre

Une fois ces clarifications apportées, l'architecture est solide et les interfaces entre axes sont bien définies.

# Réponse utilisateur

## T1

Est-ce qu'on peut désactiver ce chunked runner ? Ou considérer une seul chunk (la totalité de la simulation par exemple) ?
Je pense que la version RUN et la version OPTIMIZATION peuvent inclure des chaines d'exécution différentes, ou des exigences particulières. L'idée est de conserver quelque chose de cohérent globalement et d'automatiser soit la vérification (vérifier et expliquer à l'utilisateur ce qui est impossible dans sa configuration), soit la correction/mise en place (c'est à dire partir de la configuration de l'utilisateur pour construire la chaine d'exécution).

Ce sont donc 2 systèmes assez différents mais qui partagent une déclaration et des fonctions similaires. Avoir quelque chose de modulaire permettrait probablement de débrancher les éléments qu'on ne souhaite pas (écriture, dérivée etc...).
Le mode calibration va permettre de trouver quel valeur de paramètre utiliser, mais ne réécrit pas le graphe.

Ta recommandation me semble aller dans le bon sens.

## T2

Oui je pensais en réalité utiliser Xarray (avec numpy comme backend). L'idée est de permettre à un utilisateur non expérimenté d'ajouter une fonction avec xarray et de voir si ça fonctionne, sans avoir à se préoccuper des besoin de jax.
La performance de numpy n'est donc pas un problème car c'est pour du test d'hypothèse scientifique.
Les fonctions ne sont pas tout a fait 0D, par exemple on pourrait avoir une fonction qui récupère un state et déplace les éléments selon l'axe cohorte (revient à faire vieillir) et dans ce cas on a au moins une "core" dimension meme si ça reste interne a la fonction.

## T3

Oui absolument. C'est pourquoi je propose de définir dans les fonctions ce qui est attendu en ordre de dimension en entrée et ce qu'on fourni en sortie. ça permet au compilateur de blueprint de vérifier que tout est en ordre ET/OU de modifier (transposer) les données de l'utilisateur avant exécution.
De plus, je souhaite définir une liste de noms de dimensions (T,Z,Y,X et pourquoi pas C pour cohorte) qui soit standard. Ainsi on pourra facilement s'y référer et etre certain que tout le monde adopte le meme standard lors de l'utilisation du modèle.
On pourrait même permettre un renommage à la volée si l'utilisateur fourni une liste de correspondance.

## Z1

Il me semble que l'utilisation de valeurs NAN est non recommandée pour JAX. Sachant qu'on utilise souvent un mask en océanographie je proposait d'utiliser cette méthode. Le traitement se fait en amont sur les données pour ne pas avoir à refaire les calcul dans chaque itération.
Tous les calculs qui risquent d'être pollués par ces valeurs 0 doivent donc impérativement utiliser le mask.

## Z2

La majorité des fonctions seront écritent en JAX. L'utilisation de fonction python seront souvent limités à du test d'hypothèse.
Donc on reste sur une formution explicite (pas de polymorphisme). Il faut d'ailleurs expliciter la version dans le décorateur.

## Z3

Quoi qu'il arrive il y aura des fonctions qui nécessite d'obtenir les voisins. Je pensais utiliser les core dims pour le spécifier. On peut imaginer quelque chose comme "@functional(core_dims=['X','Y'])". Si ta question porte sur le parallélisme, dans un premier temps on considèrera impossible le parallélisme sur les chunks "core". C'est plus simple ainsi. Il y a déjà bcp de choses à faire avec le reste.

## Z4

Par temps "inifini" je présume que tu parles d'une version temps réel ou quelque chose comme ça ? Pour le moment il n'en est âs question. On reste sur du offline.

## Z5

Valeur de N configurable si besoin. Par défaut toute la dimension time, sinon on peut imagine une heuristique plus tard pour calculer la bonne valeur en fonction de la RAM. Je pense que c'est quelque chose qui n'est pas bloquant dans un premier temps.

## Z6

Tu dois pouvoir vérifier cela (context7 ? recherche internet ?) mais il me semble que JAX libère le GIL.

## Z7

Si une fonciton n'existe pas on retourne une erreur.
Oui on valide effectivement les signatures des fonctions.
Tout cela intervient lors de la création du graphe.
J'imagine : 1. on reçoit la définition (YAML, JSON, python dict ...) du blueprint; 2. on construit le graphe (ce que j'appelle compilation) et donc on vérifie que l'ensemble est fonctionnel (nom de fonction, dimension des données et IN/OUT, unité, nommage, etc...); 3. On valide (retourne le graphe) ou informe si invalide; 4. On reçoit les données (param, forçages et états initiaux, meta-paramètres, etc...); 5. On valide les données par rapport au graphe (unité, dims, NAN ? présence de tous les objets, etc...); 6. On retourne l'objet prêt pour l'exécution.
