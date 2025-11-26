# Annexe A – Formulation du modèle sans transport

Cette annexe présente la formulation mathématique du modèle de biomasse sans transport, dérivé de la structure du modèle **SEAPODYM-LMTL**. L’objectif est de décrire la dynamique d’une population de zooplancton en distinguant la production non recrutée et de la biomasse, sans prise en compte des processus d’advection ou de diffusion.

La dynamique de la biomasse totale $B$ est décrite par :

$$
\frac{\partial B}{\partial t} = R -\lambda B
\tag{Eq. 1}
$$

-   Où $R$ est le recrutement exprimé en $[\rm kg.m^{-2}.day^{-1}]$ ;
-   $B$ est la biomasse exprimée en $[\rm kg.m^{-2}]$ ;
-   $\lambda$ est le taux de mortalité exprimé en $[\rm day^{-1}]$.

La mortalité est paramétrée en fonction de la température selon une loi exponentielle :

$$
\lambda(t) = \lambda_0 \exp\big(\gamma_\lambda (T(t) - T_{\mathrm{ref}})\big)
\tag{Eq. 2}
$$

-   où $\lambda_0$ est le taux de mortalité à la température de référence $T_{\mathrm{ref}}$ exprimé en $[\rm day^{-1}]$ ;
-   $\gamma_\lambda$ est le coefficient de sensibilité thermique exprimé en $[\rm ^\circ C^{-1}]$.

Notons que dans SEAPODYM-LMTL la température est obtenue avec la formule :

$$
\begin{cases}
T(t)=\max(T(t), T_{\mathrm{ref}})
\\
T_{\mathrm{ref}}=0\rm ^\circ C
\end{cases}
\tag{Eq. 3}
$$

L’équation qui définit la production de zooplancton $p$ en fonction du temps $t$ et de l’âge $\tau$ , exprimée en $[\rm kg.m^{-2}.day^{-1}]$, est une équation en âge de McKendrick-Von Foerster [reference]. La source correspond au transfert de la production primaire nette de phytoplancton à la production de zooplancton d’âge $\tau=0$, exprimé en $\rm [day]$. Tandis que le puit correspond au transfert de la production de zooplancton à la biomasse selon un taux de recrutement $\mu$ par âge, exprimé en $\rm [day^{-1}]$.

$$

\frac{\partial p}{\partial t} + \frac{\partial p}{\partial \tau} = -\mu(t, \tau) \times p  \tag{Eq. 4.a}
$$

avec la condition de bord :

$$
p(t, \tau=0) = E \times NPP(t) \tag{Eq. 4.b}
$$

Où :

-   $NPP$ est la production primaire nette de phytoplancton exprimé en $[\rm kg.m^-2.day^{-1}]$ ;
-   $E$ est le coefficient d’efficacité de transfert entre la poduction primaire et la production de zooplancton (sans unité).

On définit $\mu$ à l’aide de la fonction indicatrice de type porte $\mathrm{\Pi}(\tau)$ :

$$
\begin{cases}

\mu(t, \tau)&=\alpha\,\mathrm{\Pi}(t, \tau)

\\

\mathrm{\Pi}(t, \tau) &=
\begin{cases}
1, \quad \tau \in \mathrm{A}(t)
\\
0, \quad sinon
\end{cases}

\end{cases}
$$

Avec :

$$
\begin{cases}

A(t) &= [\tau_r(t), \tau_{r_0}]

\\

\tau_r(t) &= \tau_{r_0} \exp{(-\gamma_{\tau_r}(T(t)-T_{\mathrm{ref}})})

\end{cases}
$$

Où :

-   $\tau_r(t)$ est l’âge minimal de recrutement à la température $T$ au temps $t$, exprimé en $[\rm day]$ ;
-   $\tau_{r_0}$ est l’âge maximal de recrutement, exprimé en $[\rm day]$, à la température de référence $T_{\mathrm{ref}}$ ;
-   $\gamma_{\tau_r}$ est le coefficient de sensibilité thermique de l’âge de recrutement, exprimé en ${\rm ^\circ C^{-1}}$ ;
-   $\alpha$ le taux de recrutement constant, exprimé en $[\rm day^{-1}]$.
-   $A(t)$ est la fenêtre d’âge de recrutement dépendant de la température.

Pour une fonction $\mu(t,\tau)$ intégrable, la solution analytique le long des caractéristiques s’écrit, pour $\tau \le t$ :

$$
p(t,\tau) = E \, \mathrm{NPP}(t−\tau)\,\exp(−\int_0^{\tau}\mu(t-\tau+s, s){\rm d}s)
\tag{Eq. 5}
$$

L’exponentielle est interprétée comme la proportion de la production qui est conservée lors du vieillissement par rapport à la première génération d’âge $\tau=0$.

Il est possible d’exprimer :

$$
\forall \tau \in \mathrm{A}(t), \quad \lim_{\alpha \to +\infin} p(t, \tau) = 0
$$

Ce cas limite correspond à une absorption totale : la production atteignant l’âge de recrutement est intégralement transférée à la biomasse, ce qui revient à la condition absorbante suivante $p(t,\tau_r)=0$.

Dans la version originale de l’équation de McKendrick-Von Foerster, $\mu$ est une fonction de mortalité. Dans notre cas, la mortalité de la production n’est rien de plus qu’un tranfert vers la biomasse. $B$ et $p$ sont similaires dans leur définition et pourraient être considérés comme des populations différentes (adulte et juvénile) avec leur dynamique propre. La vrai différence apparait dans la capacité à représenter le vieillissement de la population. Le choix a été fait ici de représenter une biomasse sans âge avec une mortalité liée à la température et une production avec âge sans mortalité (inclue implicitement dans le coefficient $E$, et donc constante quelque soit l’environnement de développement).

Par extension, la mortalité de la biomasse $B$ peut être interprétée comme un flux vers les niveaux supérieurs de la chaîne alimentaire.

# Résolution numérique

Par la suite nous utiliserons la convention $X_{a}^{n+1}$ pour signifier la valeur de $X$ à l’âge $\tau=a \times \Delta \tau$ et au temps $t = (n+1) \times \Delta t$. On simplifira l’analyse en utilisant un pas de temps égale au pas d’âge $\Delta t = \Delta \tau = \Delta$.

En utilisant la méthode d’Euler implicite nous obtenons cette solution pour $B$ :

$$
B^{n+1} = \frac{B^n + \Delta \, R(\tau_r(T^{n+1}))}{1 + \Delta \, \lambda(T^{n+1})}
\tag{Eq. 6}
$$

La définition de $p$ obtenue dans $\rm Eq. 5$ impose de connaitre l’historique des températures rencontrées par une cohorte à tout âge de sa vie, ce que nous ne faisons pas dans le contexte Eulerien. C’est pour cela que l’on préfèrera calculer $p$ en fonction du pas de temps précédent et de sa survie ($S$) durant ce même pas de temps :

$$
\begin{cases}
\begin{align*}
p_{a}^{n+1} &= p_{a-1}^{n} \, S_{a}^{n+ 1}
\\
S^n_a &= e^{-\mu^n_a \Delta}
\end{align*}
\end{cases}
$$

Ainsi nous pouvons définir le transfert de la production au temps $j$ pour l’âge $i$ comme étant :

$$
\begin{align*}

R_{a}^{n+1} &= p_{a-1}^{n} - p_{a}^{n+1}

\\

&= p_{a-1}^{n} (1 - S_{a}^{n+1})

\end{align*}
$$

Où $R_{a}^{n}$ est la production qui a été perdue pour une classe d’âge donnée, exprimée en $\rm [kg.m^{-2}.day^{-1}]$.

Puisque $\Delta t = \Delta \tau$ on peut définir :

$$
\begin{align*}

R^n &=\frac{1}{\Delta t} \sum_{a=1}^{\tau_{r_0}}{R_{a}^{n}\Delta\tau}

= \sum_{a=0}^{\tau_{r_0}}{R_{a}^{n}}

\end{align*}
$$

Où $R^{n}$ est le recrutement (i.e. la biomasse juvénile perdue par unité de temps) qui vient alimenter la biomasse $B$.
