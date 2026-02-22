- Ajouter la dimensions "F" (groupe fonctionnel) aux modèles LMTL pour tous les paramètres et pas uniquement pour les day/night_layer.
- Est-ce que le backend python (i.e. numpy/xarray) est toujours utile ? On pourait le retire pour simplifier le modèle.
- Créer un SimpleRunner (ou `model.run()`) pour le compute pur — les exemples d'optimisation (03-08)
  dupliquent tous le même boilerplate (get_all + build_step_fn + lax.scan manuel). Le StreamingRunner
  est conçu pour la production (chunking + I/O), pas pour l'optimisation (vmap, grad). Il manque un
  runner léger qui fait un seul lax.scan sans chunks ni I/O, compatible jax.vmap et jax.grad.
  Benchmark de référence : l'overhead par chunk est ~0.07s CPU, ~0.17s GPU (transfert CPU→GPU inclus).