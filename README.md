# MCTS pour l'optimisation de l'ordre de jointures (bushy join trees)

## Structure
- `join_graph.py`        : génère un schéma synthétique (N tables, cardinalités, sélectivités, graphe connexe)
- `cost_model.py`        : modèle de coût (hash-join: build+probe+output), avec `join_pair_cost`
- `bushy_algorithms.py`  : DP exact O(3^n) (Selinger-style), Greedy agglomératif, MCTS, baselines à budget équitable (random search / greedy randomisé), reconstruction d'arbre de plan  <-- LES ALGOS PRINCIPAUX
- `algorithms.py`        : version left-deep (chaîne simple) -- gardée pour la discussion (voir plus bas)
- `stats_utils.py`       : moyenne géométrique, intervalle de confiance bootstrap, test de permutation apparié (sans dépendance à scipy)
- `experiments.py`       : lance les 7 expériences (A-G, voir ci-dessous)
- `plots.py`             : génère les 9 figures + le résumé statistique à partir des CSV
- `outputs/`             : résultats (CSV, TXT) et figures (PNG)

## Pour relancer
```
python experiments.py   # régénère les CSV/TXT (~9-13 min avec les nouvelles expériences)
python plots.py          # régénère les figures
```

## Expériences (A-G)
- **(A) petite échelle** (n=4..14, `N_REPEATS=10`) : DP exact vs Greedy vs MCTS -- fig1, fig2, fig7.
- **(B) grande échelle** (n=16..40, `n_repeats=4`) : Greedy vs MCTS sans DP -- fig3, fig4.
- **(C) convergence multi-tailles** (n=16, 30, 40) : courbe de convergence MCTS normalisée par Greedy, pour voir si un espace de recherche plus grand demande plus d'itérations -- fig6.
- **(D) baselines à budget équitable** (n=10, 14, 20, 25) : compare Greedy / recherche aléatoire pure / greedy-randomisé-avec-redémarrages / MCTS, tous avec le même budget de rollouts (`MCTS_ITERS=1000`), pour isoler si l'avantage de MCTS vient de la recherche arborescente (UCB) ou juste d'un budget de calcul supérieur à Greedy -- fig8.
- **(E) ablation sur `p_greedy`** (n=20, 5 valeurs de `p_greedy` × 5 répétitions) : mesure l'effet du biais glouton dans le rollout -- fig9.
- **(F) étude de cas qualitative** : sur l'instance small-scale avec le pire ratio Greedy/optimal (n≤10), affiche les arbres de jointure réels choisis par DP/Greedy/MCTS -- `outputs/case_study.txt`.
- **(G) résumé statistique** : moyenne géométrique + IC bootstrap à 95% + test de permutation apparié (H0: pas de différence systématique Greedy/MCTS) -- `outputs/statistical_summary.txt`.

## Pourquoi bushy trees et pas left-deep ?
Version 1 du projet utilisait des plans "left-deep" (une seule chaîne qui grossit
table par table) avec un coût = somme des tailles intermédiaires. Résultat : Greedy
et MCTS étaient QUASI IDENTIQUES au DP optimal (ratio ~1.000) -- pas de différence
à montrer. En creusant, deux raisons :
1. Le terme final (taille de la jointure complète) domine ~99.8% du coût total ET
   est identique pour tout algorithme (il ne dépend pas du plan) -> il "noie"
   toute vraie différence.
2. Même en corrigeant la métrique, ce cas précis (coût = somme des tailles,
   plans left-deep) est un problème POLYNOMIAL connu (algorithme IKKBZ,
   Krishnamurthy/Boral/Zaniolo 1986) -- Greedy y est donc déjà quasi-optimal
   par construction, sans lien avec la qualité de MCTS.

La vraie version utilisée par les optimiseurs réels (System R / Selinger) autorise
des arbres BUSHY (fusionner deux résultats intermédiaires quelconques, pas
seulement étendre une chaîne) -- c'est NP-difficile, et c'est là que Greedy peut
vraiment se planter (pas de vision d'ensemble) alors que MCTS explore l'arbre
de fusions possibles.

## Résultats obtenus (voir outputs/*.png, *.csv, *.txt)
- **fig1/fig7** : sur les instances où le DP exact est calculable (n=4 à 14,
  60 instances), Greedy est parfois jusqu'à ~1 600 000x pire que l'optimum ;
  sa moyenne géométrique est **166.9x pire que l'optimum** (IC bootstrap 95% :
  [58.9x, 469.2x]), alors que MCTS reste quasi-systématiquement proche de
  l'optimum -- moyenne géométrique **1.08x** (IC 95% : [1.02x, 1.16x]). Le test
  de permutation apparié (voir `outputs/statistical_summary.txt`) donne
  p < 0.00001 : cet écart n'est pas un artefact du hasard sur ces instances.
- **fig2/fig5** : le DP exact explose bien en O(3^n) -- de 0.0001s (n=4) à ~10-14s
  (n=14), et jusqu'à 77s à n=16 (mesuré séparément) -- déjà impraticable au-delà.
- **fig3/fig4** : à grande échelle (16 à 40 tables, DP infaisable), MCTS bat
  Greedy de façon spectaculaire (souvent >1000x moins cher, ratio MCTS/Greedy
  parfois < 0.0001) tout en restant rapide (moins d'une seconde à ~30s, budget
  contrôlable).
- **fig6** : MCTS améliore sa solution au fil des itérations, sur 3 tailles
  (n=16, 30, 40) normalisées par le coût Greedy -- les instances plus grandes
  ont besoin de sensiblement plus d'itérations avant de dépasser Greedy.
- **fig8 (baselines à budget équitable)** : voir "Limites et résultats
  nuancés" ci-dessous -- ce n'est PAS un simple "MCTS gagne toujours".
- **fig9 (ablation p_greedy)** : voir "Limites et résultats nuancés" ci-dessous.
- **case_study.txt** : exemple concret (n=10, ratio Greedy/optimal = 10400x)
  montrant les arbres de jointure réels choisis par DP/Greedy/MCTS, avec les
  cardinalités et sélectivités de l'instance.

## Limites et résultats nuancés (important pour la rigueur du rapport)
- **MCTS ne gagne pas toujours à budget égal.** L'expérience (D) compare
  Greedy, une recherche aléatoire pure, un "greedy randomisé avec
  redémarrages" et MCTS, tous avec le même budget de rollouts (1000). Jusqu'à
  n=20, MCTS retrouve quasi-systématiquement le meilleur coût trouvé parmi les
  quatre méthodes. **Mais à n=25, MCTS perd face au greedy-randomisé-avec-
  redémarrages dans 3 des 4 répétitions** (jusqu'à 78x pire que le meilleur
  trouvé) -- à budget fixe, le coût de gestion de l'arbre MCTS (exploration en
  largeur au niveau racine) semble ne plus être amorti quand le facteur de
  branchement grandit, alors qu'une recherche par redémarrages complets reste
  efficace. Ce résultat mérite d'être creusé (est-ce que augmenter le budget à
  n=25 referme l'écart, ou est-ce structurel ?) plutôt que d'être caché.
- **Le rollout guidé (`p_greedy`) a un effet plus modeste que prévu.**
  L'affirmation initiale ("un rollout 100% aléatoire s'effondre à grande
  échelle") n'était qu'une intuition non testée. Mesurée à n=20 (voir fig9),
  même `p_greedy=0` (rollout entièrement aléatoire) bat encore Greedy d'un
  facteur ~10^5 -- il y a bien une tendance (les valeurs de `p_greedy` plus
  élevées font légèrement mieux), mais ce n'est pas un effondrement. Il est
  possible que l'effondrement attendu n'apparaisse qu'à plus grande échelle
  (n>30) ou avec un budget d'itérations plus serré -- non testé ici.
- Ces deux points montrent que l'avantage de MCTS n'est ni uniforme ni acquis
  "par construction" -- il dépend du rapport entre le budget de calcul et la
  taille de l'espace de recherche, ce qui est en soi un résultat plus
  intéressant qu'un simple "MCTS > Greedy partout".

## Détails techniques importants
- Le coût exclut volontairement le terme de matérialisation du résultat final
  (fixe, identique pour tout plan valide) -- voir le commentaire dans
  `cost_model.join_pair_cost` pour la justification complète.
- MCTS utilise UCB1, un rollout biaisé (75% greedy / 25% aléatoire par défaut),
  et un warm-start avec la solution Greedy comme incumbent initial (désactivé
  dans l'expérience d'ablation (E) pour isoler l'effet du rollout).
- `N_REPEATS` instances aléatoires par taille pour avoir des statistiques
  (médiane + min/max sur les figures, pas moyenne/écart-type classique car les
  ratios sont très asymétriques -- une moyenne géométrique + IC bootstrap est
  utilisée pour le résumé global, et un test de permutation apparié pour la
  significativité -- voir `stats_utils.py`, sans dépendance à scipy).
- Les baselines à budget équitable (D) reçoivent exactement le même nombre de
  rollouts complets que MCTS a d'itérations -- une approximation raisonnable
  du budget de calcul (MCTS fait aussi un rollout par itération, plus une
  gestion d'arbre légère), mais pas une égalité stricte en temps CPU.
