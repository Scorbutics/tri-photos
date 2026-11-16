"""
Code commun à tri_photos.py et tri_videos.py.

Bibliothèque standard uniquement : tri_videos.py doit rester léger
(pas de numpy, PIL ni torch).
"""
from __future__ import annotations

import argparse
import bisect
import collections
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Sequence

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}

# Dossier de travail (plan, rapport, cache, journaux) dans le dossier trié, partagé
# par les deux scripts : leurs fichiers ont des noms distincts.
PLAN_DIR = ".tri_photos"

# Modèle d'IA (CLIP) utilisé par défaut pour les empreintes d'images (voir tri_ia.py)
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"

# Dossiers de premier niveau de la structure triée (plus AAAA/AAAA-MM/)
SORTED_TOP = {"_A_REVOIR", "Voyages", "Themes"}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_command(args) -> None:
    """Lance la sous-commande choisie. Ctrl+C : tout ce qui a été fait est déjà
    sauvegardé (caches, journal d'apply), il suffit de relancer la même commande."""
    try:
        args.func(args)
    except KeyboardInterrupt:
        log("\nInterrompu. Le travail déjà fait est sauvegardé : relance la même "
            "commande pour reprendre là où elle s'est arrêtée.")
        sys.exit(130)


# ---------------------------------------------------------------------------
# Écritures sûres : un fichier est soit l'ancien, soit le nouveau, jamais à moitié
# écrit (interruption, coupure de courant, disque débranché).
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def atomic_open(path: Path, mode: str = "w", **kw) -> Iterator:
    """Écrit dans <path>.tmp puis le renomme en <path> (opération atomique)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, mode, **kw) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    with atomic_open(path, "w", encoding="utf-8") as f:
        f.write(text)


def file_id(path: str | Path) -> str:
    """Identité d'un fichier qui survit à un déplacement (contrairement au chemin)."""
    st = os.stat(path)
    return f"{Path(path).name}|{st.st_size}|{st.st_mtime}"


class Every:
    """Vrai au plus une fois toutes les `seconds` secondes (sauvegardes régulières)."""

    def __init__(self, seconds: float):
        self.seconds, self.last = seconds, time.monotonic()

    def __call__(self) -> bool:
        if time.monotonic() - self.last >= self.seconds:
            self.last = time.monotonic()
            return True
        return False


# ---------------------------------------------------------------------------
# Dossiers
# ---------------------------------------------------------------------------
def work_dir(source: str) -> tuple[Path, Path]:
    """(dossier source, dossier de travail <source>/.tri_photos)."""
    src = Path(source).expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"Dossier introuvable : {src}")
    return src, src / PLAN_DIR


def sorted_folder(rel: Path) -> str | None:
    """Dossier de rangement si le fichier (chemin relatif à la source) est déjà dans
    la structure triée : _A_REVOIR/…, Voyages/…, Themes/… ou AAAA/AAAA-MM/…
    Un fichier ailleurs (racine, autre dossier) est nouveau et sera rangé."""
    parts = rel.parts[:-1]
    if not parts:
        return None
    if parts[0] in SORTED_TOP or (
            len(parts) >= 2 and re.fullmatch(r"\d{4}", parts[0])
            and re.fullmatch(rf"{parts[0]}-\d{{2}}", parts[1])):
        return "/".join(parts)
    return None


# ---------------------------------------------------------------------------
# Métadonnées (exiftool : un appel par lot de 500 fichiers)
# ---------------------------------------------------------------------------
def run_exiftool(files: list[Path], tags: list[str], options: Sequence[str] = (),
                 cache: Path | None = None) -> dict[str, dict]:
    """{chemin absolu: métadonnées}. Suppose exiftool installé.
    Avec `cache`, les métadonnées déjà lues (même fichier, même nom/taille/date,
    mêmes options) sont reprises, et le cache est sauvegardé après chaque lot :
    une analyse interrompue ne relit pas ce qu'elle a déjà lu."""
    signature = hashlib.sha1(json.dumps([list(tags), list(options)]).encode()).hexdigest()
    known: dict[str, dict] = {}
    if cache and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("signature") == signature:
                known = data["items"]
        except (OSError, ValueError, KeyError):
            log(f"  ! cache de métadonnées illisible ({cache.name}), on le reconstruit")
    out: dict[str, dict] = {}
    ids = {str(f): file_id(f) for f in files}
    todo = []
    for f in files:
        item = known.get(ids[str(f)])
        if item is not None:
            out[os.path.abspath(f)] = item
        else:
            todo.append(f)
    current = set(ids.values())
    known = {k: v for k, v in known.items() if k in current}  # oublie les fichiers disparus
    if len(todo) < len(files):
        log(f"  métadonnées en cache : {len(files) - len(todo)}/{len(files)}")
    chunk = 500
    for i in range(0, len(todo), chunk):
        part = [str(f) for f in todo[i:i + chunk]]
        r = subprocess.run(["exiftool", "-json", "-n", "-q", "-q", *options, *tags, *part],
                           capture_output=True, text=True)
        try:
            for item in json.loads(r.stdout or "[]"):
                path = os.path.abspath(item["SourceFile"])
                out[path] = item
                if path in ids:
                    known[ids[path]] = item
        except json.JSONDecodeError:
            pass
        if cache:
            atomic_write_text(cache, json.dumps({"signature": signature, "items": known}))
        log(f"  métadonnées {min(i + chunk, len(todo))}/{len(todo)}")
    return out


def parse_exif_date(s) -> dt.datetime | None:
    """Garde l'heure « murale » (heure locale affichée), sans fuseau."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    for fmt, n in (("%Y:%m:%d %H:%M:%S%z", 25), ("%Y:%m:%d %H:%M:%S", 19),
                   ("%Y-%m-%dT%H:%M:%S%z", 25), ("%Y-%m-%dT%H:%M:%S", 19)):
        try:
            d = dt.datetime.strptime(s[:n], fmt).replace(tzinfo=None)
        except ValueError:
            continue
        return d if d.year >= 1990 else None  # 0000:00:00, horloge pas réglée
    return None


# ---------------------------------------------------------------------------
# Géographie et voyages
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def safe_name(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else " " for c in s)
    return "_".join(keep.split())[:60] or "Voyage"


def trip_label(coords: list[tuple[float, float]]) -> str:
    """Nom d'un voyage : les 2 villes les plus fréquentes, ou les pays s'il y en a plusieurs."""
    if not coords:
        return "Voyage"
    try:
        import reverse_geocoder as rg  # optionnel
        res = rg.search(coords, mode=1, verbose=False)
    except Exception:
        return "Voyage"
    cities = collections.Counter(r["name"] for r in res)
    countries = collections.Counter(r["cc"] for r in res)
    if len(countries) > 1:
        return safe_name("-".join(c for c, _ in countries.most_common(3)))
    return safe_name("-".join(c for c, _ in cities.most_common(2)))


def find_trips(items: Sequence, ts: Sequence[dt.datetime], home: tuple[float, float],
               trip_km: float, gap_days: float) -> list[list[int]]:
    """Découpe une suite chronologique (objets avec .lat/.lon, dates ts) en voyages :
    suites d'éléments loin de chez soi, coupées par un retour à la maison ou par un
    trou de plus de gap_days. Sans GPS : on prend l'élément GPS le plus proche dans
    le temps (12 h max), sinon « inconnu » (ni voyage ni maison).
    Renvoie des listes d'indices dans items."""
    gps = [i for i, x in enumerate(items) if x.lat is not None]
    away: list[bool | None] = [None] * len(items)
    for i in gps:
        away[i] = haversine_km(items[i].lat, items[i].lon, *home) > trip_km
    if gps:
        gps_t = [ts[i] for i in gps]
        for i in range(len(items)):
            if away[i] is not None:
                continue
            k = bisect.bisect_left(gps_t, ts[i])
            d, j = min((abs(gps_t[c] - ts[i]), gps[c]) for c in (k - 1, k) if 0 <= c < len(gps))
            if d <= dt.timedelta(hours=12):
                away[i] = away[j]

    trips: list[list[int]] = []
    cur: list[int] = []
    for i in range(len(items)):
        if away[i]:
            if cur and ts[i] - ts[cur[-1]] > dt.timedelta(days=gap_days):
                trips.append(cur)
                cur = []
            cur.append(i)
        elif away[i] is False and cur:
            trips.append(cur)  # retour à la maison
            cur = []
    if cur:
        trips.append(cur)
    return trips


# ---------------------------------------------------------------------------
# Groupes de copies / quasi-doublons
# ---------------------------------------------------------------------------
def settle_group(members: list, keep: int, gid: int, score: Callable):
    """Décide qui garder dans un groupe de copies (objets avec .pinned, .path,
    .dup_group, .dup_keep). Les éléments déjà rangés ne bougent jamais : si l'un
    d'eux est gardé (hors _A_REVOIR), tous les nouveaux du groupe sont écartés ;
    sinon on garde les `keep` meilleurs nouveaux selon score.
    Renvoie (référence, nouveaux écartés), ou None si le groupe n'a rien de nouveau."""
    news = [x for x in members if not x.pinned]
    if not news:
        return None
    olds = [x for x in members if x.pinned]
    for x in members:
        x.dup_group = gid
    for x in olds:
        x.dup_keep = not x.pinned.startswith("_A_REVOIR")
    old_keepers = [x for x in olds if x.dup_keep]
    if old_keepers:
        ref, losers = max(old_keepers, key=score), news
    else:
        ranked = sorted(news, key=score, reverse=True)
        ref, losers = ranked[0], ranked[keep:]
    for x in losers:
        x.dup_keep = False
    return ref, losers


def similar_folder(members: list, ref) -> str:
    """Dossier _A_REVOIR/similaires/ d'un groupe : celui où des éléments du groupe
    sont déjà rangés, sinon un nouveau au nom de la référence (stable d'une analyse
    à l'autre, contrairement à un numéro)."""
    existing = [x.pinned for x in members if x.pinned.startswith("_A_REVOIR/similaires/")]
    if existing:
        return collections.Counter(existing).most_common(1)[0][0]
    return f"_A_REVOIR/similaires/{Path(ref.path).stem}"


# ---------------------------------------------------------------------------
# Application / annulation du plan
# ---------------------------------------------------------------------------
class Plan:
    """Ce qui distingue le plan des photos de celui des vidéos."""

    def __init__(self, csv_name: str, journal_prefix: str, noun: str):
        self.csv_name = csv_name                # plan.csv / plan_videos.csv
        self.journal_prefix = journal_prefix    # un journal par apply : <prefix><date>.csv
        self.noun = noun                        # « photos » / « vidéos » (féminin pluriel)

    # Journal d'un apply : une ligne par fichier (source, destination, mode, racine),
    # écrite et forcée sur le disque AVANT de toucher au fichier. Tant que l'apply
    # n'est pas terminé, le journal s'appelle <prefix><date>.en_cours.csv : relancer
    # apply le reprend, undo sait l'annuler.
    IN_PROGRESS = ".en_cours.csv"

    def active_journals(self, work: Path) -> list[Path]:
        """Journaux d'apply non annulés (terminés ou en cours), du plus ancien au plus
        récent : c'est la pile d'undo."""
        return sorted((j for j in work.glob(f"{self.journal_prefix}[0-9]*.csv")
                       if not j.name.endswith(".annule.csv")), key=lambda j: j.name)

    def apply(self, args) -> None:
        src_root, work = work_dir(args.source)
        plan = work / self.csv_name
        if not plan.exists():
            sys.exit(f"Aucun plan dans {work} : lance d'abord analyse {src_root}")
        dest_root = Path(args.out).expanduser().resolve() if args.out else src_root
        mode = "copie" if args.copy else "deplacement"

        active = self.active_journals(work)
        resumed: dict[str, Path] = {}  # source -> destination déjà traitées
        if active and active[-1].name.endswith(self.IN_PROGRESS):
            log_path = active[-1]
            with open(log_path, newline="") as f:
                rows = [r for r in csv.reader(f) if len(r) >= 4]
            if rows and (rows[0][2], Path(rows[0][3])) != (mode, dest_root):
                sys.exit(f"Un apply interrompu ({log_path.name}) a été lancé avec "
                         f"{'--copy ' if rows[0][2] == 'copie' else ''}"
                         f"{'--out ' + rows[0][3] if Path(rows[0][3]) != src_root else '(sur place)'} : "
                         "relance-le avec les mêmes options, ou undo pour l'annuler.")
            resumed = {r[0]: Path(r[1]) for r in rows}
            log(f"Reprise de l'apply interrompu ({log_path.name}, {len(resumed)} déjà traités).")
        elif active and active[-1].stat().st_mtime >= plan.stat().st_mtime:
            sys.exit(f"Ce plan a déjà été appliqué ({active[-1].name}). "
                     f"Relance analyse pour un nouveau plan, ou undo pour annuler.")
        else:
            log_path = work / (f"{self.journal_prefix}{dt.datetime.now():%Y%m%d-%H%M%S-%f}"
                               + self.IN_PROGRESS)

        done = in_place = missing = 0
        vacated: set[Path] = set()
        with open(plan, newline="") as f, open(log_path, "a", newline="") as lf:
            w = csv.writer(lf)
            for row in csv.DictReader(f):
                src = Path(row["source"])
                if not row["destination"]:
                    continue
                prev = resumed.get(str(src))
                if prev is not None and prev.exists() and (args.copy or not src.exists()):
                    done += 1  # déjà fait avant l'interruption
                    continue
                if not src.exists():
                    missing += 1
                    continue
                dest_dir = dest_root / row["destination"]
                dest = dest_dir / src.name
                if dest.resolve() == src.resolve():  # déjà rangé (tri sur place relancé)
                    in_place += 1
                    continue
                dest_dir.mkdir(parents=True, exist_ok=True)
                n = 1
                while dest.exists():
                    dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
                    n += 1
                w.writerow([src, dest, mode, dest_root])
                lf.flush()
                os.fsync(lf.fileno())
                transfer(src, dest, copy=args.copy)
                if not args.copy:
                    vacated.add(src.parent)
                done += 1
        if done:
            log_path.rename(log_path.with_name(log_path.name.replace(self.IN_PROGRESS, ".csv")))
        else:
            log_path.unlink()
        remove_empty_dirs(vacated, src_root)
        log(f"{done} {self.noun} {'copiées' if args.copy else 'déplacées'} vers {dest_root}"
            + (f" ({in_place} déjà à leur place)" if in_place else "") + "."
            + (f" Annulation possible : undo {src_root}" if done else ""))
        if missing:
            log(f"! {missing} {self.noun} du plan introuvables (déplacées depuis l'analyse ?) : "
                f"relance analyse {src_root} pour un plan à jour.")

    def undo(self, args) -> None:
        src_root, work = work_dir(args.source)
        active = self.active_journals(work) if work.is_dir() else []
        if not active:
            sys.exit(f"Rien à annuler dans {src_root}.")
        journal = active[-1]
        with open(journal, newline="") as f:
            rows = [r for r in csv.reader(f) if len(r) >= 2]  # ligne tronquée par une coupure
        n = skipped = 0
        emptied: set[Path] = set()
        dest_root = src_root
        for row in reversed(rows):
            src, dest = Path(row[0]), Path(row[1])
            mode = row[2] if len(row) > 2 else "deplacement"
            dest_root = Path(row[3]) if len(row) > 3 else dest_root
            if not dest.exists():
                continue
            if mode == "copie" and src.exists():
                dest.unlink()  # l'original est toujours là : on supprime la copie
            elif src.exists():
                log(f"  ! {src} existe déjà, {dest} laissé en place")
                skipped += 1
                continue
            else:
                src.parent.mkdir(parents=True, exist_ok=True)
                transfer(dest, src, copy=False)
            emptied.add(dest.parent)
            n += 1
        remove_empty_dirs(emptied, dest_root)
        journal.rename(journal.with_name(
            journal.name.replace(self.IN_PROGRESS, ".csv").removesuffix(".csv") + ".annule.csv"))
        log(f"{n} {self.noun} remises à leur état d'origine ({journal.name})"
            + (f", {skipped} ignorées" if skipped else "") + ".")

    def add_commands(self, sub) -> None:
        """Sous-commandes apply et undo."""
        p = sub.add_parser("apply", help="applique le plan (sur place, ou vers --out)")
        p.add_argument("source", help="dossier analysé")
        p.add_argument("--out", help="ranger dans ce dossier au lieu de ranger sur place")
        p.add_argument("--copy", action="store_true", help="copier au lieu de déplacer")
        p.set_defaults(func=self.apply)
        u = sub.add_parser("undo", help="annule le dernier apply")
        u.add_argument("source", help="dossier analysé")
        u.set_defaults(func=self.undo)


def transfer(src: Path, dest: Path, copy: bool) -> None:
    """Copie ou déplace src vers dest sans jamais laisser de fichier à moitié écrit
    à dest : sur le même disque, un renommage (atomique) ; sinon copie vers
    <dest>.part puis renommage, et l'original n'est supprimé qu'ensuite."""
    if not copy:
        try:
            os.rename(src, dest)
            return
        except OSError:  # autre disque : copie puis suppression
            pass
    part = dest.with_name(dest.name + ".part")
    try:
        shutil.copy2(src, part)
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()
    if not copy:
        os.unlink(src)


def remove_empty_dirs(dirs: set[Path], root: Path) -> None:
    """Supprime les dossiers vidés par un déplacement, sans jamais remonter jusqu'à root."""
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        while d != root and root in d.parents:
            try:
                d.rmdir()  # échoue si le dossier n'est pas vide
            except OSError:
                break
            d = d.parent


# ---------------------------------------------------------------------------
# Options communes de la commande analyse
# ---------------------------------------------------------------------------
def add_common_analyse_args(a: argparse.ArgumentParser, what: str) -> None:
    """what : « photos » ou « vidéos »."""
    a.add_argument("source", help=f"dossier de {what} (le plan va dans <source>/{PLAN_DIR})")
    a.add_argument("--tout-refaire", action="store_true",
                   help=f"re-trier aussi les {what} déjà rangées (_A_REVOIR/, Voyages/, "
                        "Themes/, AAAA/AAAA-MM/) au lieu de les laisser en place")
    a.add_argument("--revoir-auto", action="store_true",
                   help=f"mettre aussi de côté dans _A_REVOIR les {what} détectées automatiquement "
                        "(captures d'écran, floues, doublons…). Par défaut, seuls les groupes "
                        "marqués « À revoir » dans la page de découverte y vont.")
    a.add_argument("--home-lat", type=float, default=-27.4698)   # Brisbane
    a.add_argument("--home-lon", type=float, default=153.0251)
    a.add_argument("--trip-km", type=float, default=100,
                   help="distance au domicile au-delà de laquelle c'est un voyage")
    a.add_argument("--trip-gap-days", type=float, default=3,
                   help="trou sans prise de vue qui coupe un voyage en deux")
