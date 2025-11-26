# Fonctions pour SeapoPym

## Seuil de température

On applique un seuil de température en fonction d'un T_0. Ainsi la température ne peut jamais être inférieure à T_0.

## Temperature Loi de Gillooly

Applique une transformation sur la température moyenne rencontrée selon la loi de Gillooly : `T = T / (1 + T/273)`
Note : Si le calcul peut être fait après celui de la température moyenne rencontrée durant un timestep, ça devrait réduire le coût de calcul. Si ce n'est pas équivalent on le fait bien au début.

## Day length

Calcul la durée du jour en fonction de la latitude et de la date.
Retourne un float compris entre 0 et 1 qui est le ratio durée du jour divisé par la durée d'un jour.

## Température moyenne

Calcul la température moyenne rencontrée durant un timestep en fonction de la profondeur et cycle jour/nuit.
Un groupe fonctionnel qui effectue une migration vertical diurne passe d'une couche à l'autre chaque jour.
En fonction de la durée du jour on doit obtenir :
`T_mean = T.sel(depth=group.param.position_day) * day_length + T.sel(depth=group.param.position_night) * (1-day_length)`

## Age minimal de recrutement

Calcul l'âge minimal de recrutement en fonction de la température moyenne rencontrée durant un timestep. On utilise la fonction `tau_r = tau_r_0 * exp(-gamma_tau_r * T_mean)`.

## Transfert d'énergie

Applique le coefficient de transfert d'énergie pour obtenir la production d'âge 0.
`p(t=n, age=0) = E * primary_productivity(t=n)`
Le coefficient E est constant pour un groupe fonctionnel.
C'est une tendance pour l'état "production".

## Vieillissement de la production

On transfert l'intégralité de la production d'âge tau vers l'âge tau+delta_tau.
Note : En général on choisi delta_tau = delta_t.
c'est une tendance pour l'état "production" également.

## Recrutement

On transfert l'intégralité de la production d'âge 0 vers vers la biomasse (celle-ci n'a pas d'âge).
On sort donc deux tendances ici : "production" et "biomasse".
La tendance de la "production" va retirer tous les âge >= à tau_r.
La tendance de la "biomasse" va sommer tous les âge >= à tau_r sur le pas de temps delta_t.
Attention il y a une gestion stricte des pas de temps et des pas d'âge.
Je te propose de te référer à la description mathématique du modèle dans mon annexe.
TODO(Jules): Fournir le lien vers l'annexe.

## Calcul du taux de mortalité

Le taux de mortalité dépend de la température moyenne rencontrée durant un timestep. Il est appliqué uniquement à la biomasse donc il ne dépend pas de l'âge.
La formule est : `mortality_rate = mortality_rate_0 * exp(gamma_mortality_rate * T_mean)`

## Mortalité

On applique le taux de mortalité à la biomasse pour obtenir la tendance.
