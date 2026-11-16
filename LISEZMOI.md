# Pré-tri de photos, 100 % local

## Installation (Linux, une seule fois)

```bash
sudo apt install libimage-exiftool-perl python3-venv
mkdir ~/tri-photos && cd ~/tri-photos        # y copier tri_photos.py, tri_videos.py, tri_commun.py et tri_ia.py
python3 -m venv .venv && . .venv/bin/activate
# torch ET torchvision depuis l'index CPU, ensemble : sinon open_clip installe un
# torchvision incompatible (erreur « operator torchvision::nms does not exist »)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # version CPU, ~200 Mo
pip install open_clip_torch pillow pillow-heif opencv-python-headless numpy scipy reverse_geocoder
```

Au premier lancement, le modèle CLIP (environ 600 Mo) est téléchargé une seule fois. Tout tourne ensuite hors ligne.

## Utilisation

`decouvrir` et `analyse` construisent un **plan** sans toucher aux photos. `apply` l'exécute, `undo` l'annule.
Tout ce que le script produit (plan, rapport, cache, vignettes, journaux) va dans le dossier caché `.tri_photos/` à l'intérieur du dossier de photos.

```bash
. .venv/bin/activate
# 1. Analyse : ne déplace RIEN, produit .tri_photos/plan.csv + .tri_photos/rapport.html
python tri_photos.py analyse ~/Photos/A_trier

# 2. Ouvrir ~/Photos/A_trier/.tri_photos/rapport.html et regarder chaque groupe

# 3a. Appliquer SUR PLACE : les photos sont rangées dans ~/Photos/A_trier/2024/…, Voyages/…, _A_REVOIR/…
python tri_photos.py apply ~/Photos/A_trier

# 3b. … ou ranger ailleurs (--copy pour copier au lieu de déplacer)
python tri_photos.py apply ~/Photos/A_trier --out ~/Photos/Trie [--copy]

# En cas de regret : annuler le dernier apply (répétable : annule les apply un par un)
python tri_photos.py undo ~/Photos/A_trier
```

- Un `apply` sur place supprime les sous-dossiers qu'il a vidés. `undo` les recrée.
- Annuler un `apply --copy` supprime les copies. Les originaux n'ont pas bougé.

## Ajouter de nouvelles photos à un dossier déjà trié

Dépose les nouvelles photos à la racine du dossier trié, ou dans n'importe quel sous-dossier qui ne fait pas partie de la structure (par exemple `Import/`). Relance ensuite `analyse` puis `apply` :

```bash
cp ~/nouvelles/* ~/Photos/A_trier/Import/
python tri_photos.py analyse ~/Photos/A_trier
python tri_photos.py apply   ~/Photos/A_trier
```

Une photo est **déjà rangée** si elle se trouve dans `_A_REVOIR/…`, `Voyages/…`, `Themes/…` ou `AAAA/AAAA-MM/…`. Ces photos ne bougent plus jamais :

- les corrections faites à la main sont respectées : photo sortie de `_A_REVOIR`, voyage renommé (`Voyages/Sydney avec Paul`), photo supprimée…
- seules les nouvelles photos sont rangées et apparaissent dans le rapport.

Les photos déjà rangées servent quand même de référence :

- **Voyages** : une nouvelle photo prise pendant un voyage existant rejoint son dossier, quel que soit son nom actuel. Si de nouvelles photos font se rejoindre deux voyages, chacune va dans celui dont les dates sont les plus proches. Sinon, un nouveau voyage est créé.
- **Quasi-doublons** (avec `--revoir-auto`) : une nouvelle photo semblable à une photo déjà rangée va dans `_A_REVOIR/similaires/<photo de référence>`. Le rapport signale quand la nouvelle est de meilleure qualité : à toi de choisir laquelle garder.
- **Mois, thèmes, catégories** : les nouvelles photos rejoignent les dossiers existants, ou en créent de nouveaux.
- **Nouveaux groupes** : relance `decouvrir`. Les groupes déjà nommés gardent leur nom.

Ne dépose pas de nouvelles photos directement dans un dossier de la structure (`2024/2024-03/`…) : elles seraient considérées comme déjà rangées et ne seraient pas analysées.

`--tout-refaire` re-trie tout depuis zéro, photos déjà rangées comprises. Dans ce cas, les dossiers renommés à la main et les corrections ne sont plus respectés.

## Découvrir les catégories que tu ne connais pas encore

```bash
# 1. Regroupe les photos qui se ressemblent, SANS catégories prédéfinies
python tri_photos.py decouvrir ~/Photos/A_trier

# 2. Ouvre ~/Photos/A_trier/.tri_photos/decouverte.html et décide quoi faire de chaque groupe,
#    puis clique « Enregistrer categories.json » (le navigateur le télécharge, par ex. dans ~/Téléchargements)

# 3. Analyse avec tes groupes (prioritaires sur les catégories prédéfinies)
python tri_photos.py analyse ~/Photos/A_trier --categories ~/Téléchargements/categories.json
```

Dans la page, chaque groupe a trois choix :

| Choix | Effet sur ses photos |
|---|---|
| **Tri normal** (par défaut) | Aucun : elles sont rangées comme les autres, par voyage ou par mois. |
| **À revoir** | `_A_REVOIR/<nom>/` : probablement à supprimer après un coup d'œil. |
| **Dossier thème** | `Themes/<nom>/`, sauf les photos prises pendant un voyage, qui restent dans le voyage. |

La page sert aussi à ajuster les groupes :

- **Toutes les photos** : 18 photos par groupe sont affichées, le bouton « Voir les N photos » montre le reste.
- **Grosseur des groupes** : le curseur réunit les groupes qui se ressemblent le plus. Il n'y a rien à recalculer. Chaque groupe réuni peut être ressorti avec « séparer ».
- **Fusion à la main** : coche des groupes, puis clique « Fusionner ». Le nom et le choix de la fusion s'appliquent à tous ses groupes.
- **Mémoire** : tes choix sont gardés dans le navigateur. Tu peux fermer la page et revenir plus tard, et ils sont aussi repris après une nouvelle découverte.
- « Charger… » reprend les choix d'un `categories.json` déjà enregistré.

`analyse` recopie `categories.json` dans `.tri_photos/`. Les analyses suivantes l'utilisent donc sans `--categories`, et `decouvrir` y retrouve les noms déjà donnés.

- `--grain` (0.25 par défaut) : plus petit = plus de groupes, plus précis. Pour des groupes **plus fins** que la première découverte, relance `decouvrir --grain 0.2`. L'analyse des images est en cache, donc la relance prend environ une minute. Pour des groupes **plus larges**, utilise plutôt le curseur de la page.
- `--min-group` (6) : taille minimale d'un groupe affiché.
- `--videos DOSSIER` : regroupe aussi les vidéos de ce dossier avec les photos. L'IA analyse 3 images de chaque vidéo, et ces vidéos sont marquées ▶ dans la page. Tes choix s'appliqueront alors aux vidéos (voir « Vidéos »).
- `--page-seule` : régénère seulement `decouverte.html` à partir de la dernière découverte, en une seconde. Rien n'est recalculé et aucun modèle n'est chargé.
- Relancer `decouvrir` (par exemple avec de nouvelles photos) garde les noms et choix déjà donnés aux groupes qui se retrouvent.
- `categories.json` contient le « portrait moyen » de chaque groupe. Il resservira pour les photos des prochaines années : une nouvelle photo de reçu ira toute seule dans `_A_REVOIR/Reçus`.

Pour 5 000 photos, compte 10 à 25 min pour la première analyse sur un CPU correct. Les relances sont quasi instantanées grâce au cache. Tu peux donc ajuster les seuils et relancer.

## Interrompre et reprendre

Toutes les commandes peuvent être interrompues (Ctrl+C, fermeture du terminal, coupure de courant) sans perdre le travail fait. Il suffit de relancer **la même commande** :

| Commande | Ce qui est sauvegardé en continu |
|---|---|
| `analyse`, `decouvrir` | L'analyse des images, toutes les minutes et à l'interruption. Les métadonnées (exiftool), après chaque lot de 500. Les vignettes, une par une. |
| `apply` | Chaque déplacement est noté dans le journal **avant** d'être fait. Relancer `apply` reprend où il s'était arrêté, avec les mêmes options (`--out`, `--copy`). `undo` sait aussi annuler un `apply` interrompu. |

Les fichiers écrits (plan, rapport, caches, vignettes) le sont de façon atomique : un fichier est soit l'ancien, soit le nouveau, jamais à moitié écrit. Une copie, ou un déplacement vers un autre disque, passe par un fichier `.part` renommé à la fin. Une photo n'est donc jamais à moitié copiée.

## Résultat

Dans le dossier de photos (tri sur place) ou dans `--out` :

```
A_trier/            (ou Trie/)
  .tri_photos/      plan, rapport, cache, journaux (dossier de photos uniquement)
  Voyages/
    2024-07-12_Voyage_Sydney/
    2025-01-03_Voyage_FR-JP/      (plusieurs pays = codes pays)
  2024/2024-03/ ...               (tout le reste, par mois)
  Themes/<nom>/                   (groupes « Dossier thème » de la page de découverte)
  _A_REVOIR/<nom>/                (groupes « À revoir » de la page de découverte)
```

### Rien n'est mis à revoir automatiquement

Par défaut, une photo ou une vidéo ne va dans `_A_REVOIR` que si **tu** l'as décidé, en marquant son groupe « À revoir » dans la page de découverte. Tout le reste suit le tri normal : voyage, sinon mois.

`--revoir-auto` (pour `analyse`, photos comme vidéos) réactive les détections automatiques, qui s'ajoutent à tes choix :

```
_A_REVOIR/
  captures_ecran/  documents/  recus_tickets/  articles_magasin/   (photos : modèle d'IA)
  notes_tableaux/  memes_images_web/  photos_ratees/  messagerie/
  floues/  trop_sombres/  surexposees/                             (photos : qualité)
  similaires/IMG_1234/            (quasi-doublons ; la meilleure reste rangée normalement)
  doublons/  live_photos/  enregistrements_ecran/  tres_courtes/  illisibles/   (vidéos)
```

Les options `--category-threshold`, `--blur-threshold`, `--dup-threshold`, `--keep`, `--min-duration`… ne servent qu'avec `--revoir-auto`.

## Réglages utiles

| Option | Défaut | Effet |
|---|---|---|
| `--trip-km` | 100 | Distance à partir de laquelle c'est un voyage. Gold Coast ≈ 70 km, Sunshine Coast ≈ 90 km, Toowoomba ≈ 125 km. |
| `--trip-min-photos` | 8 | En dessous, l'escapade reste dans le dossier du mois. |
| `--trip-gap-days` | 3 | Un trou sans photo plus long coupe le voyage en deux. |
| `--keep` | 1 | Nombre de photos gardées par groupe de similaires (mets 2 si tu veux). |
| `--dup-threshold` | 0.93 | Plus haut = moins de photos considérées comme similaires. |
| `--category-threshold` | 0.5 | Plus haut = moins de photos envoyées dans les catégories à revoir. |
| `--blur-threshold` | 60 | Plus bas = moins de photos jugées floues. |
| `--home-lat/--home-lon` | Brisbane | Ville de référence. |

### Origine des photos et vidéos

La marque de l'appareil est lue dans les métadonnées. Pour les iPhone, c'est « Apple ». Pour les Samsung, c'est « samsung » dans les photos, et une balise propre à Samsung (`SamsungModel`) dans les vidéos. Les autres téléphones Android utilisent leurs propres balises.

Une photo ou une vidéo **sans marque d'appareil** n'a pas été prise par un appareil photo :

- **Elle n'est jamais rattachée à un voyage d'après sa date** : reçue pendant un voyage, elle n'en fait pas partie. Elle va dans le dossier du mois, avec la mention « origine inconnue » dans le rapport.
- Avec `--revoir-auto` seulement :
  - format d'écran de téléphone (1080x2400, 1179x2556…), ou PNG : `_A_REVOIR/captures_ecran` ;
  - ni date ni lieu (image ou vidéo reçue par WhatsApp…, ou téléchargée) : `_A_REVOIR/messagerie`. Pour les photos, c'est le cas sauf si le modèle leur trouve une catégorie plus précise ; pour les vidéos, seulement en basse définition.

Les catégories sont des phrases en anglais en haut du script (`REVIEW_CATEGORIES`). Tu peux en ajouter, par exemple `"plantes_jardin": ["a photo of a plant label", ...]`.

## Vidéos (`tri_videos.py`)

Le tri repose sur les métadonnées (exiftool), le nom et la taille des fichiers : des milliers de vidéos en quelques secondes. `reverse_geocoder` est optionnel et sert à nommer les voyages. Avec `ffmpeg`, chaque vidéo a une vignette dans le rapport (`--sans-vignettes` pour s'en passer).

**Thèmes et groupes « à revoir », comme pour les photos.** Si un `categories.json` existe (celui de la page de découverte), l'IA regarde 3 images de chaque vidéo. Une vidéo qui ressemble à un groupe « Dossier thème » va dans `Themes/<nom>/`, sauf si elle a été filmée pendant un voyage. Une vidéo qui ressemble à un groupe « À revoir » va dans `_A_REVOIR/<nom>/`. Le fichier est cherché à trois endroits, dans cet ordre :

1. `--categories` ;
2. `.tri_photos/` du dossier de vidéos ;
3. `.tri_photos/` du dossier de photos donné par `--plan-photos`, où `tri_photos.py analyse` le recopie.

L'analyse IA est mise en cache et peut être interrompue. Elle demande les mêmes dépendances que `tri_photos.py`. `--sans-ia` s'en passe complètement : dans ce cas, seuls exiftool (et ffmpeg pour les vignettes) sont nécessaires.

Pour que tes choix tiennent compte des vidéos, lance la découverte avec elles :

```bash
python tri_photos.py decouvrir ~/Photos/A_trier --videos ~/Videos/A_trier
# choisis dans la page, enregistre categories.json, puis :
python tri_photos.py analyse ~/Photos/A_trier --categories ~/Téléchargements/categories.json
python tri_videos.py analyse ~/Videos/A_trier --plan-photos ~/Photos/A_trier
```

Mêmes commandes et même fonctionnement que pour les photos, y compris pour ajouter des vidéos à un dossier déjà trié et pour `--tout-refaire`. Le plan (`plan_videos.csv` + `rapport_videos.html`) va dans le même dossier `.tri_photos/`, avec des noms de fichiers distincts, et chaque script n'annule que ses propres `apply`.

```bash
python tri_videos.py analyse ~/Photos/A_trier          # photos et vidéos dans le même dossier
python tri_videos.py apply   ~/Photos/A_trier          # sur place (ou --out DOSSIER [--copy])
python tri_videos.py undo    ~/Photos/A_trier

# Vidéos dans un autre dossier que les photos : indiquer où sont les photos
python tri_videos.py analyse ~/Videos/A_trier --plan-photos ~/Photos/A_trier
```

Si le plan des photos existe, une vidéo prise pendant un voyage trouvé par les photos va dans ce voyage, même sans GPS. Lance donc `tri_photos.py analyse` avant. Le script prend `.tri_photos/plan.csv` dans le dossier analysé, ou ce que désigne `--plan-photos` : un dossier de photos ou directement un `plan.csv`.

Avec `--revoir-auto` seulement :

| À revoir | Comment c'est détecté |
|---|---|
| `illisibles` | pas de durée lisible (fichier tronqué ou corrompu) |
| `doublons` | copie exacte (même taille, puis même hash) |
| `live_photos` | ≤ 4 s, et identifiant Live Photo Apple ou photo du même nom à côté |
| `enregistrements_ecran` | nom (`RPReplay`, `Screen Recording`…), ou format d'écran ≥ 19,5:9 sans marque d'appareil |
| `similaires/` | même durée (±0,15 s), même format, même moment : copie ré-encodée. La meilleure qualité est gardée |
| `messagerie` | dossier ou nom WhatsApp / Telegram / Signal / Messenger, sans marque d'appareil ; ou ni appareil, ni date, et basse définition (vidéo compressée par une messagerie) |
| `tres_courtes` | < 2 s (`--min-duration`) |

Le rapport indique la taille de chaque dossier et liste les 30 vidéos les plus lourdes. Clique sur ▶ pour lire une vidéo : elle ne se charge qu'à ce moment-là.
Les dates QuickTime sont converties de l'UTC vers l'heure de cet ordinateur. Si une vieille caméra enregistre en heure locale, ajoute `--dates-locales`.
