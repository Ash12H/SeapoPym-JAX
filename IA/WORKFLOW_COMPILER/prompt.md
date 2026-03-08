Aujourd'hui le pipeline Blueprint → Config → CompiledModel mélange validation et transformation dans `compile_model()`. Les types de la Config ne sont pas stricts (`dict[str, Any]` pour forcings et initial_state), et la validation croisée Blueprint × Config se fait en même temps que la compilation.

On veut séparer clairement 3 phases :

1. **Config** : typage strict (forcings et initial_state en `dict[str, xr.DataArray]`), validation Pydantic à la construction.
2. **Validation croisée** Blueprint × Config : cohérence des variables, dimensions, unités, couverture temporelle. Peut échouer avec des messages clairs.
3. **Compilation** : transformation mécanique (grille temporelle, shapes, transposition, ForcingStore, JAX arrays). Ne devrait jamais échouer si la validation est passée.

Contraintes :
- Blueprint et Config sont considérés stables dans leur design. On peut modifier le code (typage, validators) mais pas changer leur rôle.
- Le CompiledModel reste un conteneur passif — pas de gestion mémoire.
- L'interface avec le Runner doit être préservée (CompiledModel est consommé par le Runner tel quel).
