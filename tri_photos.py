#!/usr/bin/env python3
"""
Pré-tri de photos 100 % local.

  1. Catégories "à revoir" par modèle CLIP local (captures d'écran, documents,
     reçus, articles en magasin, photos ratées...).
  2. Qualité : flou, trop sombre, surexposé.
  3. Quasi-doublons / rafales : on garde la meilleure (ou les N meilleures).
  4. Rangement : voyages (GPS loin de la ville de référence) sinon par mois.

decouvrir et analyse ne touchent à AUCUNE photo : ils construisent un plan
(plan.csv + rapport HTML avec vignettes) dans <dossier>/.tri_photos/.
apply exécute le plan (sur place, ou vers --out), undo annule le dernier apply.

Ajout de photos : dépose-les dans le dossier déjà trié (hors de la structure, par
exemple à la racine) puis relance analyse + apply. Les photos déjà rangées
(_A_REVOIR/, Voyages/, Themes/, AAAA/AAAA-MM/) ne bougent pas ; les nouvelles
rejoignent les voyages, mois et groupes existants ou en créent de nouveaux.
--tout-refaire re-trie tout depuis zéro.

Usage :
  python tri_photos.py decouvrir ~/Photos/A_trier      # optionnel : trouver des groupes,
                                       # les nommer dans .tri_photos/decouverte.html
  python tri_photos.py analyse   ~/Photos/A_trier [--categories categories.json]
  python tri_photos.py apply     ~/Photos/A_trier      # range sur place selon le plan
  python tri_photos.py apply     ~/Photos/A_trier --out ~/Photos/Trie [--copy]
  python tri_photos.py undo      ~/Photos/A_trier      # annule le dernier apply
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from tri_commun import (IMAGE_EXT, Every, Plan, add_common_analyse_args, atomic_open,
                        atomic_write_text, file_id, find_trips, log, parse_exif_date,
                        run_command, run_exiftool, safe_name, settle_group, similar_folder,
                        sorted_folder, trip_label, work_dir)

PLAN = Plan("plan.csv", "deplacements_", "photos")

try:
    import pillow_heif  # photos iPhone (.heic)
    pillow_heif.register_heif_opener()
except ImportError:
    print("! pillow-heif introuvable : les photos .heic (iPhone) seront illisibles. "
          "Installe-le : pip install pillow-heif", file=sys.stderr)


# ---------------------------------------------------------------------------
# Catégories CLIP. Plusieurs formulations par catégorie, moyennées.
# Les catégories "garder" servent de contrepoids : une photo n'est classée
# "à revoir" que si une catégorie à revoir l'emporte nettement sur elles.
# Ajoute / retire librement des catégories ici.
# ---------------------------------------------------------------------------
REVIEW_CATEGORIES = {
    "captures_ecran": [
        "a screenshot of a phone screen", "a screenshot of a computer screen",
        "a screenshot of a chat conversation", "a screenshot of a website",
        "a screenshot of an app",
    ],
    "documents": [
        "a photo of a document", "a photo of a printed page with text",
        "a photo of a form", "a photo of a letter", "a scanned document",
        "a photo of an id card or passport",
    ],
    "recus_tickets": [
        "a photo of a receipt", "a photo of a shopping receipt",
        "a photo of an invoice", "a photo of a ticket",
    ],
    "articles_magasin": [
        "a photo of a product on a store shelf", "a photo of a price tag",
        "a photo of an item for sale in a shop", "a photo of a product with its price label",
        "a photo of a product box", "a photo of a supermarket aisle",
    ],
    "notes_tableaux": [
        "a photo of a whiteboard", "a photo of handwritten notes",
        "a photo of a presentation slide", "a photo of a computer monitor",
    ],
    "memes_images_web": [
        "a meme with text", "an internet meme", "a cartoon or illustration",
        "a graphic image with text",
    ],
    "photos_ratees": [
        "a blurry accidental photo", "a photo taken by accident inside a pocket",
        "a completely black photo", "an accidental photo of the floor",
        "an accidental photo of a finger over the lens",
    ],
}
KEEP_CATEGORIES = {
    "_photo": [
        "a photo of people", "a portrait of a person", "a group photo of friends",
        "a photo of a child", "a selfie",
        "a landscape photo", "a photo of a beach", "a photo of a city street",
        "a photo of a building", "a photo of nature", "a photo of mountains",
        "a photo of an animal", "a photo of a pet", "a photo of food on a plate",
        "a photo of a meal at a restaurant", "a photo of a party",
        "a travel photo", "a photo of the sea", "a photo of a sunset",
        "a photo of a garden", "a photo of a car", "a photo of a room interior",
    ],
}

# ---------------------------------------------------------------------------
@dataclass
class Photo:
    path: str
    date: str | None = None          # ISO
    date_source: str = ""
    lat: float | None = None
    lon: float | None = None
    make: str | None = None
    width: int = 0
    height: int = 0
    sharpness: float = 0.0
    brightness: float = 0.0
    clipped_dark: float = 0.0
    clipped_bright: float = 0.0
    category: str = ""
    category_score: float = 0.0
    quality_flag: str = ""
    dup_group: int | None = None
    dup_keep: bool = True
    dup_folder: str = ""             # _A_REVOIR/similaires/<photo de référence>
    dup_note: str = ""
    pinned: str = ""                 # dossier actuel si déjà rangée (on n'y touche pas)
    theme: str = ""
    destination: str = ""
    reason: str = ""
    emb: np.ndarray | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Métadonnées (exiftool : un seul appel pour tout le dossier)
# ---------------------------------------------------------------------------


def read_metadata(files: list[Path], cache: Path) -> dict[str, dict]:
    if not shutil.which("exiftool"):
        log("! exiftool introuvable : dates = date du fichier, pas de GPS. "
            "Installe-le : sudo apt install libimage-exiftool-perl")
        return {}
    return run_exiftool(files, ["-DateTimeOriginal", "-CreateDate", "-MediaCreateDate",
                                "-GPSLatitude", "-GPSLongitude", "-Make",
                                "-ImageWidth", "-ImageHeight"], cache=cache)


# ---------------------------------------------------------------------------
# Qualité de l'image
# ---------------------------------------------------------------------------
def load_rgb(path: str, max_side: int = 768) -> Image.Image | None:
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_side, max_side))
        return im.convert("RGB")
    except Exception as e:  # fichier corrompu
        log(f"  ! illisible : {path} ({e})")
        return None


def quality_metrics(im: Image.Image) -> tuple[float, float, float, float]:
    import cv2
    g = np.asarray(im.convert("L"))
    # netteté : variance du laplacien sur une image ramenée à 512 px
    h, w = g.shape
    s = 512 / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return (sharp, float(g.mean()),
            float((g < 15).mean()), float((g > 245).mean()))


# ---------------------------------------------------------------------------
# CLIP local
# ---------------------------------------------------------------------------
class Clip:
    def __init__(self, model_name: str, pretrained: str):
        import open_clip
        import torch
        self.torch = torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Chargement du modèle CLIP {model_name}/{pretrained} sur {self.device} "
            "(premier lancement : téléchargement unique)…")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_images(self, ims: list[Image.Image]) -> np.ndarray:
        t = self.torch.stack([self.preprocess(im) for im in ims]).to(self.device)
        with self.torch.no_grad():
            f = self.model.encode_image(t).float()
        f /= f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    def encode_categories(self, cats: dict[str, list[str]]) -> tuple[list[str], np.ndarray]:
        names, vecs = [], []
        for name, prompts in cats.items():
            with self.torch.no_grad():
                f = self.model.encode_text(self.tokenizer(prompts).to(self.device)).float()
            f /= f.norm(dim=-1, keepdim=True)
            v = f.mean(0)
            vecs.append((v / v.norm()).cpu().numpy())
            names.append(name)
        return names, np.stack(vecs)


# ---------------------------------------------------------------------------
# Géographie
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
def load_photos(args) -> tuple[list[Photo], list[Photo], "Clip | None", Path]:
    """Métadonnées + mesures + embeddings (avec cache). Commun à analyse et decouvrir."""
    src, out = work_dir(args.source)
    out.mkdir(exist_ok=True)
    files = sorted(p for p in src.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXT  # les vidéos : tri_videos.py
                   and out not in p.parents)  # pas les vignettes
    log(f"{len(files)} fichiers trouvés dans {src}")
    if not files:
        sys.exit("Aucune photo trouvée.")

    meta = read_metadata(files, out / ".cache_exif_photos.json")
    photos: list[Photo] = []
    for f in files:
        m = meta.get(str(f), {})
        p = Photo(path=str(f))
        d = None
        for k in ("DateTimeOriginal", "CreateDate", "MediaCreateDate"):
            d = parse_exif_date(m.get(k))
            if d:
                p.date_source = k
                break
        if not d:
            d = dt.datetime.fromtimestamp(f.stat().st_mtime)
            p.date_source = "fichier"
        p.date = d.isoformat()
        lat, lon = m.get("GPSLatitude"), m.get("GPSLongitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and (lat or lon):
            p.lat, p.lon = float(lat), float(lon)
        p.make = m.get("Make")
        photos.append(p)

    # --- cache des embeddings / mesures (relancer avec d'autres seuils = rapide)
    # Sauvegardé toutes les minutes, à la fin et en cas d'interruption (Ctrl+C) :
    # une analyse interrompue reprend où elle s'était arrêtée.
    cache_file = out / ".cache_tri.npz"
    cache: dict[str, tuple] = {}
    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=True) as z:
                cache = z["cache"].item()
        except Exception as e:  # fichier abîmé (ancienne version, disque plein…)
            log(f"! cache illisible ({e}) : mis de côté, l'analyse repart de zéro.")
            cache_file.rename(cache_file.with_name(cache_file.name + ".abime"))

    def save_cache() -> None:
        with atomic_open(cache_file, "wb") as f:
            np.savez(f, cache=np.array(cache, dtype=object))

    clip = None
    images = photos
    todo = []
    for p in images:
        key = f"{file_id(p.path)}|{args.model}/{args.pretrained}"
        if key in cache:
            p.emb, (p.width, p.height, p.sharpness, p.brightness,
                    p.clipped_dark, p.clipped_bright) = cache[key]
        else:
            todo.append((key, p))

    if todo:
        log(f"  {len(images) - len(todo)} images déjà analysées (cache), {len(todo)} à analyser")
        clip = Clip(args.model, args.pretrained)
        bs = 32
        every_minute = Every(60)
        try:
            analyse_batches(todo, bs, clip, cache, save_cache, every_minute)
        finally:
            save_cache()
    return photos, images, clip, out


def analyse_batches(todo, bs, clip, cache, save_cache, every_minute) -> None:
    """Mesures + embeddings CLIP des images pas encore en cache, par lots."""
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        ims, ok = [], []
        for key, p in batch:
            im = load_rgb(p.path)
            if im is None:
                p.quality_flag = "illisible"
                continue
            with Image.open(p.path) as raw:
                p.width, p.height = raw.size
            (p.sharpness, p.brightness,
             p.clipped_dark, p.clipped_bright) = quality_metrics(im)
            ims.append(im)
            ok.append((key, p))
        if ims:
            embs = clip.encode_images(ims)
            for (key, p), e in zip(ok, embs):
                p.emb = e
                cache[key] = (e, (p.width, p.height, p.sharpness, p.brightness,
                                  p.clipped_dark, p.clipped_bright))
        log(f"  analyse images {min(i + bs, len(todo))}/{len(todo)}")
        if every_minute():
            save_cache()


def photo_score(p: Photo) -> tuple:
    """Meilleure photo d'un groupe : pas floue/sombre, puis la plus nette à résolution comparable."""
    return ((0 if p.quality_flag else 1),
            math.log1p(p.sharpness) + 0.5 * math.log1p(p.width * p.height))


def settle_photo_group(members: list[Photo], keep: int, gid: int) -> bool:
    """Groupe de quasi-doublons (règle commune : settle_group) ; les nouvelles
    photos écartées vont dans _A_REVOIR/similaires/<photo de référence>."""
    settled = settle_group(members, keep, gid, photo_score)
    if not settled:
        return False
    ref, losers = settled
    for p in members:
        if not p.pinned and p.dup_keep:
            p.quality_flag = ""  # la gardée d'un groupe n'est pas « à revoir »
    folder = similar_folder(members, ref)
    for p in losers:
        p.dup_folder = folder
        p.dup_note = f"quasi-doublon de {Path(ref.path).name}"
        if ref.pinned:
            p.dup_note += " (déjà rangée)"
            if photo_score(p) > photo_score(ref):
                p.dup_note += ", celle-ci est de meilleure qualité"
    return True


def analyse(args) -> None:
    photos, images, clip, out = load_photos(args)
    src = out.parent

    # --- photos déjà rangées : elles gardent leur place (sauf --tout-refaire)
    if not args.tout_refaire:
        for p in photos:
            p.pinned = sorted_folder(Path(p.path).relative_to(src)) or ""
    new = [p for p in photos if not p.pinned]
    pinned = len(photos) - len(new)
    if pinned:
        log(f"{pinned} photos déjà rangées (laissées en place), {len(new)} nouvelles à ranger.")

    # --- 0. catégories découvertes et nommées par toi (prioritaires).
    # Le fichier donné est recopié dans .tri_photos/ : decouvrir y retrouve les noms
    # et les analyses suivantes l'utilisent sans avoir à le repasser.
    cats_file = out / "categories.json"
    if args.categories:
        given = Path(args.categories).expanduser().resolve()
        if given != cats_file:
            shutil.copyfile(given, cats_file)
    if cats_file.exists() and new:
        apply_discovered(new, cats_file)

    # --- 1. catégories prédéfinies (texte), pour les nouvelles photos
    if new:
        if clip is None:
            clip = Clip(args.model, args.pretrained)
        rev_names, rev_vecs = clip.encode_categories(REVIEW_CATEGORIES)
        keep_names, keep_vecs = clip.encode_categories(KEEP_CATEGORIES)
        all_names = rev_names + keep_names
        all_vecs = np.concatenate([rev_vecs, keep_vecs])
    for p in new:
        if p.emb is None or p.category or p.theme:
            continue
        logits = 100.0 * all_vecs @ p.emb
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        i = int(prob.argmax())
        if all_names[i] in rev_names and prob[i] >= args.category_threshold:
            p.category, p.category_score = all_names[i], float(prob[i])
        # capture d'écran évidente : PNG sans marque d'appareil
        if (not p.category and Path(p.path).suffix.lower() == ".png" and not p.make):
            p.category, p.category_score = "captures_ecran", -1.0

    # --- 2. qualité (seulement pour les nouvelles photos non catégorisées)
    for p in new:
        if p.category or p.emb is None:
            continue
        if p.brightness < 25 or p.clipped_dark > 0.85:
            p.quality_flag = "trop_sombres"
        elif p.clipped_bright > 0.6:
            p.quality_flag = "surexposees"
        elif p.sharpness < args.blur_threshold:
            p.quality_flag = "floues"

    # --- 3. quasi-doublons : nouvelles photos "normales", comparées entre elles ET
    # aux photos déjà rangées (gardées, ou déjà dans un groupe de similaires)
    def comparable(p: Photo) -> bool:
        if p.emb is None:
            return False
        if p.pinned:
            return (not p.pinned.startswith("_A_REVOIR")
                    or p.pinned.startswith("_A_REVOIR/similaires/"))
        return not p.category

    normal = sorted((p for p in photos if comparable(p)), key=lambda p: p.date)
    parent = list(range(len(normal)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    times = [dt.datetime.fromisoformat(p.date) for p in normal]
    if normal and new:
        E = np.stack([p.emb for p in normal])
        win = dt.timedelta(minutes=args.dup_window_min)
        for i in range(len(normal)):
            j = i + 1
            while j < len(normal) and times[j] - times[i] <= win:
                if float(E[i] @ E[j]) >= args.dup_threshold:
                    parent[find(j)] = find(i)
                j += 1
        # copies identiques à n'importe quelle date (photo ré-enregistrée, WhatsApp…)
        sims = E @ E.T
        ii, jj = np.where(np.triu(sims, 1) >= 0.985)
        for i, j in zip(ii, jj):
            parent[find(int(j))] = find(int(i))

    groups = collections.defaultdict(list)
    for i, p in enumerate(normal):
        groups[find(i)].append(p)
    gid = 0
    for members in groups.values():
        if len(members) >= 2 and settle_photo_group(members, args.keep, gid + 1):
            gid += 1

    # --- 4. voyages / mois. Les photos déjà rangées (hors _A_REVOIR) participent à la
    # détection : une nouvelle photo prise pendant un voyage existant rejoint son dossier.
    home = (args.home_lat, args.home_lon)
    kept = sorted((p for p in photos
                   if (p.pinned and not p.pinned.startswith("_A_REVOIR"))
                   or (not p.pinned and not p.category and not p.quality_flag and p.dup_keep)),
                  key=lambda p: p.date)
    ts = [dt.datetime.fromisoformat(p.date) for p in kept]
    trips = find_trips(kept, ts, home, args.trip_km, args.trip_gap_days)

    in_trip: dict[int, str] = {}
    for t in trips:
        fresh = [i for i in t if not kept[i].pinned]
        if not fresh:
            continue
        # voyage déjà rangé : on réutilise son dossier, quel que soit son nom actuel.
        # Si les nouvelles photos font se rejoindre plusieurs voyages rangés, chacune
        # va dans le dossier dont les dates sont les plus proches des siennes.
        spans: dict[str, list[dt.datetime]] = {}
        for i in t:
            if kept[i].pinned.startswith("Voyages/"):
                s = spans.setdefault(kept[i].pinned, [ts[i], ts[i]])
                s[0], s[1] = min(s[0], ts[i]), max(s[1], ts[i])
        if spans:
            for i in fresh:
                in_trip[i] = min(spans, key=lambda f: max(spans[f][0] - ts[i],
                                                          ts[i] - spans[f][1],
                                                          dt.timedelta(0)))
        elif len(t) >= args.trip_min_photos:
            coords = [(kept[i].lat, kept[i].lon) for i in t if kept[i].lat is not None]
            folder = f"Voyages/{ts[t[0]]:%Y-%m-%d}_Voyage_{trip_label(coords)}"
            for i in fresh:
                in_trip[i] = folder

    for i, p in enumerate(kept):
        if p.pinned:
            continue
        if i in in_trip:
            p.destination, p.reason = in_trip[i], "voyage"
        elif p.theme:
            p.destination, p.reason = f"Themes/{p.theme}", "thème"
        else:
            p.destination, p.reason = f"{ts[i]:%Y}/{ts[i]:%Y-%m}", "mois"

    for p in photos:
        if p.destination:
            continue
        if p.pinned:
            p.destination, p.reason = p.pinned, "déjà rangée"
        elif p.category:
            p.destination = f"_A_REVOIR/{p.category}"
            p.reason = ("png sans appareil photo" if p.category_score < 0
                        else "groupe découvert" if p.category_score > 1
                        else f"modèle {p.category_score:.0%}")
        elif p.quality_flag:
            p.destination = f"_A_REVOIR/{p.quality_flag}"
            p.reason = (f"netteté {p.sharpness:.0f}" if p.quality_flag == "floues"
                        else f"luminosité {p.brightness:.0f}")
        elif not p.dup_keep:
            p.destination, p.reason = p.dup_folder, p.dup_note

    write_plan(out, photos)
    write_report(out, photos)
    summary(photos)
    log(f"\nPlan : {out/'plan.csv'}\nRapport : {out/'rapport.html'}\n"
        f"Rien n'a été déplacé. Vérifie le rapport puis : "
        f"python {Path(sys.argv[0]).name} apply {src}  [--out DOSSIER] [--copy]")


def write_plan(out: Path, photos: list[Photo]) -> None:
    with atomic_open(out / "plan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "destination", "raison", "date", "date_source",
                    "groupe_similaires", "gardee"])
        for p in sorted(photos, key=lambda p: (p.destination, p.date)):
            w.writerow([p.path, p.destination, p.reason, p.date, p.date_source,
                        p.dup_group or "", "oui" if p.dup_keep else "non"])


def make_thumb(out: Path, p: Photo) -> str:
    thumbs = out / ".vignettes"
    thumbs.mkdir(exist_ok=True)
    name = hashlib.sha1(file_id(p.path).encode()).hexdigest()[:16] + ".jpg"
    t = thumbs / name
    if not t.exists():
        im = load_rgb(p.path, 240)
        if im:
            with atomic_open(t, "wb") as f:
                im.save(f, format="JPEG", quality=70)
    return f".vignettes/{name}"


def make_thumbs(out: Path, photos: list[Photo]) -> list[str]:
    """Vignettes de plusieurs photos, en parallèle (le décodage HEIC est lent).
    Celles qui existent déjà sont réutilisées : une interruption ne perd rien."""
    from concurrent.futures import ThreadPoolExecutor
    every = Every(15)
    done = [0]

    def one(p: Photo) -> str:
        t = make_thumb(out, p)
        done[0] += 1
        if every():
            log(f"  vignettes {done[0]}/{len(photos)}")
        return t

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        return list(pool.map(one, photos))


def write_report(out: Path, photos: list[Photo]) -> None:
    by_dest = collections.defaultdict(list)
    for p in photos:
        if not p.pinned:  # le rapport ne montre que ce qui va bouger
            by_dest[p.destination].append(p)
    # les groupes de similaires montrent aussi la photo gardée (même déjà rangée)
    keepers = {p.dup_group: p for p in photos if p.dup_group and p.dup_keep}

    def thumb(p: Photo) -> str:
        return make_thumb(out, p)

    order = sorted(by_dest, key=lambda d: (not d.startswith("_A_REVOIR"), d))
    parts = ["<!doctype html><meta charset=utf-8><title>Tri photos</title><style>"
             "body{font-family:system-ui;margin:16px;background:#111;color:#eee}"
             "h2{margin-top:32px;border-bottom:1px solid #444}"
             ".g{display:flex;flex-wrap:wrap;gap:6px}"
             "figure{margin:0;width:160px;font-size:11px;color:#aaa}"
             "img{width:160px;height:120px;object-fit:cover;border-radius:4px}"
             ".keep img{outline:3px solid #3c3}"
             "nav a{color:#8cf;margin-right:12px;line-height:1.8}</style>"
             "<h1>Plan de tri</h1><nav>"]
    for d in order:
        parts.append(f'<a href="#{html.escape(d)}">{html.escape(d)} ({len(by_dest[d])})</a>')
    parts.append("</nav>")
    for d in order:
        items = by_dest[d]
        parts.append(f'<h2 id="{html.escape(d)}">{html.escape(d)} — {len(items)}</h2><div class=g>')
        if d.startswith("_A_REVOIR/similaires/"):
            k = keepers.get(items[0].dup_group)
            if k:
                items = [k] + items
        for p in items[:400]:
            cls = " class=keep" if (p.dup_group and p.dup_keep and d.startswith("_A_REVOIR/sim")) else ""
            label = ("GARDÉE (déjà rangée) " if p.pinned else "GARDÉE ") if cls else ""
            img = f'<img loading=lazy src="{thumb(p)}">'
            parts.append(f'<figure{cls}>{img}<figcaption>{label}{html.escape(Path(p.path).name)}'
                         f'<br>{p.date[:16]} · {html.escape(p.reason)}</figcaption></figure>')
        if len(items) > 400:
            parts.append(f"<p>… et {len(items) - 400} autres</p>")
        parts.append("</div>")
    atomic_write_text(out / "rapport.html", "\n".join(parts))


def summary(photos: list[Photo]) -> None:
    new = [p for p in photos if not p.pinned]
    c = collections.Counter()
    for p in new:
        d = p.destination
        c["voyages" if d.startswith("Voyages/") else
          "thèmes" if d.startswith("Themes/") else
          "similaires" if d.startswith("_A_REVOIR/similaires") else
          d if d.startswith("_A_REVOIR") else "par mois"] += 1
    log(f"\nRésumé ({len(new)} nouvelles photos à ranger) :")
    for k, v in sorted(c.items()):
        log(f"  {k:32s} {v}")
    existing = {p.pinned for p in photos if p.pinned.startswith("Voyages/")}
    trips = {p.destination for p in new if p.destination.startswith("Voyages/")}
    log(f"  voyages : {len(trips - existing)} nouveau(x), {len(trips & existing)} existant(s) complété(s)")
    if len(new) < len(photos):
        log(f"  déjà rangées, laissées en place : {len(photos) - len(new)}")


# ---------------------------------------------------------------------------
# Découverte : regrouper sans connaître les catégories à l'avance
# ---------------------------------------------------------------------------
# Vocabulaire servant uniquement à SUGGÉRER un nom pour chaque groupe trouvé.
# Les groupes eux-mêmes ne dépendent pas de cette liste.
VOCABULAIRE = {
    "captures d'écran": "a screenshot", "documents": "a document with text",
    "reçus": "a shopping receipt", "articles en magasin": "a product on a store shelf",
    "étiquettes de prix": "a price tag", "notes manuscrites": "handwritten notes",
    "tableau blanc": "a whiteboard", "memes": "an internet meme",
    "cartes et plans": "a map", "billets et cartes d'embarquement": "a ticket or boarding pass",
    "plats et repas": "food on a plate", "desserts": "a dessert or cake",
    "boissons et cafés": "a drink or a coffee", "restaurants": "a restaurant",
    "plage": "a beach", "mer et océan": "the ocean", "montagne": "mountains",
    "forêt et randonnée": "a forest hiking trail", "coucher de soleil": "a sunset",
    "ville et rues": "a city street", "monuments": "a famous monument",
    "bâtiments et architecture": "a building", "intérieur de maison": "a room interior",
    "jardin et plantes": "plants in a garden", "fleurs": "flowers",
    "chiens": "a dog", "chats": "a cat", "oiseaux": "a bird",
    "animaux sauvages": "a wild animal", "kangourous et faune australienne": "a kangaroo or koala",
    "bébés": "a baby", "enfants": "children playing", "selfies": "a selfie",
    "portraits": "a portrait of a person", "photos de groupe": "a group of people",
    "fêtes et anniversaires": "a birthday party", "mariages": "a wedding",
    "sport": "people playing sport", "concerts et spectacles": "a concert",
    "voitures": "a car", "bateaux": "a boat", "avions et aéroports": "an airplane or airport",
    "trains et transports": "a train or public transport", "vélo": "a bicycle",
    "piscine": "a swimming pool", "neige": "snow", "nuit et lumières": "city lights at night",
    "ciel et nuages": "the sky with clouds", "écrans et télé": "a tv or computer screen",
    "vêtements": "clothes", "meubles": "furniture", "bricolage et travaux": "home renovation work",
    "voiture abîmée ou réparation": "car damage or repair", "objets à vendre": "an object for sale",
    "nourriture au marché": "a food market", "œuvres d'art et musées": "artwork in a museum",
    "dessins d'enfants": "a child's drawing", "photos floues ou ratées": "a blurry accidental photo",
}


def cluster_id(centre: list[float]) -> str:
    """Identifiant stable d'un groupe (sert à retrouver les choix faits dans la page)."""
    return hashlib.sha1(json.dumps(centre[:32]).encode()).hexdigest()[:12]


def restore_choices(clusters: list[dict], out: Path) -> int:
    """Reprend les noms et choix d'un categories.json précédent (dans .tri_photos/)
    pour les groupes qui se retrouvent (même « portrait moyen »)."""
    prev_file = out / "categories.json"
    if not prev_file.exists():
        return 0
    previous = [(c, np.array(c["centre"])) for c in json.loads(prev_file.read_text())]
    n = 0
    for cl in clusters:
        c = np.array(cl["centre"])
        best = max(previous, key=lambda pc: float(pc[1] @ c), default=None)
        if best and float(best[1] @ c) > 0.9:  # même groupe qu'à la découverte précédente
            cl["nom"], cl["action"] = best[0]["nom"], best[0].get("action", "garder")
            cl["vu"] = best[0].get("vu", True)
            n += 1
    return n


def legacy_discovery_data(page: Path) -> dict:
    """Données d'une page decouverte.html produite par une ancienne version
    (avant decouverte.json) : elles y sont embarquées en JSON."""
    text = page.read_text(encoding="utf-8")
    m = re.search(r"^const C = (.*);$", text, re.M)
    if not m:
        sys.exit(f"Pas de données de groupes dans {page}.")
    clusters = json.loads(m.group(1))
    for cl in clusters:  # cette version ne gardait que 18 exemples par groupe
        cl["photos"], cl["partiel"] = cl.pop("exemples"), True
    t = re.search(r"(\d+) images, \d+ groupes \(\+ (\d+) photos", text)
    total, small = (int(t.group(1)), int(t.group(2))) if t else (0, 0)
    return {"source": str(page.parent.parent), "total": total, "small": small,
            "clusters": clusters}


def discover(args) -> None:
    src, out = work_dir(args.source)
    data_file = out / "decouverte.json"
    if args.page_seule:  # régénérer la page sans rien recalculer
        if data_file.exists():
            data = json.loads(data_file.read_text(encoding="utf-8"))
        elif (out / "decouverte.html").exists():
            data = legacy_discovery_data(out / "decouverte.html")
        else:
            sys.exit(f"Aucune découverte précédente dans {out} : lance d'abord decouvrir {src}")
        for cl in data["clusters"]:
            cl.setdefault("id", cluster_id(cl["centre"]))
            if "photos" not in cl:  # ancienne version : 18 exemples seulement
                cl["photos"], cl["partiel"] = cl.pop("exemples", []), True
        restored = restore_choices(data["clusters"], out)
        if restored:
            log(f"{restored} groupes reprennent les choix de {out/'categories.json'}.")
        atomic_write_text(data_file, json.dumps(data))
        write_discovery_page(out, data)
        log(f"Page régénérée : {out/'decouverte.html'}")
        return

    from scipy.cluster.hierarchy import fcluster, linkage

    photos, images, clip, out = load_photos(args)
    imgs = [p for p in images if p.emb is not None]
    if len(imgs) < 2:
        sys.exit("Pas assez d'images.")
    E = np.stack([p.emb for p in imgs]).astype(np.float64)
    log(f"Regroupement de {len(imgs)} images…")
    Z = linkage(E, method="average", metric="cosine")
    labels = fcluster(Z, t=args.grain, criterion="distance")

    if clip is None:
        clip = Clip(args.model, args.pretrained)
    voc_names, voc_vecs = clip.encode_categories(
        {fr: [f"a photo of {en}", f"{en}"] for fr, en in VOCABULAIRE.items()})

    clusters, ordered = [], []
    members_by = collections.defaultdict(list)
    for p, l in zip(imgs, labels):
        members_by[int(l)].append(p)
    small = 0
    for members in sorted(members_by.values(), key=len, reverse=True):
        if len(members) < args.min_group:
            small += len(members)
            continue
        M = np.stack([p.emb for p in members])
        c = M.mean(0)
        c /= np.linalg.norm(c)
        sims = M @ c
        # seuil d'appartenance pour les futures photos : un peu sous le moins typique
        seuil = float(max(0.5, np.percentile(sims, 10) - 0.02))
        sugg = voc_vecs @ c
        top = [voc_names[i] for i in np.argsort(-sugg)[:3]]
        centre = [round(float(x), 5) for x in c]
        clusters.append({
            "id": cluster_id(centre), "nom": top[0], "action": "garder", "vu": False,
            "nb": len(members), "suggestions": top, "seuil": round(seuil, 4),
            "centre": centre,
        })
        ordered.append([members[i] for i in np.argsort(-sims)])  # les plus typiques d'abord

    # vignettes de TOUTES les photos des groupes (la page en montre 18, puis le reste à la demande)
    log(f"Vignettes de {sum(map(len, ordered))} photos…")
    thumbs = make_thumbs(out, [p for members in ordered for p in members])
    k = 0
    for cl, members in zip(clusters, ordered):
        cl["photos"] = thumbs[k:k + len(members)]
        k += len(members)
    restore_choices(clusters, out)
    data = {"source": str(src), "total": len(imgs), "small": small, "grain": args.grain,
            "clusters": clusters}
    atomic_write_text(out / "decouverte.json", json.dumps(data))
    write_discovery_page(out, data)
    log(f"\n{len(clusters)} groupes trouvés (+ {small} photos isolées ou dans de petits groupes).")
    log(f"Ouvre {out/'decouverte.html'}, décide quoi faire de chaque groupe, puis clique "
        f"« Enregistrer categories.json ».\n"
        f"Ensuite : python {Path(sys.argv[0]).name} analyse {src} "
        f"--categories <chemin du categories.json téléchargé>")


def write_discovery_page(out: Path, data: dict) -> None:
    script = Path(sys.argv[0]).name if sys.argv and sys.argv[0].endswith(".py") else "tri_photos.py"
    payload = json.dumps({**data, "script": script}).replace("</", "<\\/")
    atomic_write_text(out / "decouverte.html", DISCOVERY_PAGE.replace("__DATA__", payload))


DISCOVERY_PAGE = r"""<!doctype html>
<html lang=fr>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Découverte des groupes</title>
<style>
:root{
  --bg:#121212;--panel:#1b1b1b;--panel2:#232323;--line:#333;--text:#eee;--muted:#9a9a9a;
  --normal:#7aa2c8;--revoir:#e0904a;--theme:#4fb8a8;--accent:#8cf;--sel:#f0d264;
}
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text);line-height:1.45}
main{max-width:1400px;margin:0 auto;padding:0 16px 80px}
h1{font-size:22px;margin:20px 0 4px}
h2{font-size:16px;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 16px}
code{background:var(--panel2);padding:1px 5px;border-radius:4px;font-size:.92em}
button{font:inherit}

/* Explication des trois choix */
.choices{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin:0 0 14px}
.choice{background:var(--panel);border:1px solid var(--line);border-left:4px solid;border-radius:8px;padding:10px 12px}
.choice h3{margin:0 0 4px;font-size:15px}
.choice p{margin:0;font-size:13px;color:#ccc}
.choice .dest{margin-top:6px;font-family:ui-monospace,monospace;font-size:12px;color:var(--muted)}
.c-garder{border-left-color:var(--normal)}.c-revoir{border-left-color:var(--revoir)}.c-theme{border-left-color:var(--theme)}
.note{font-size:13px;color:var(--muted);margin:0 0 14px}

/* Regroupement */
.regroup{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:0 0 14px}
.regroup .row{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center}
.regroup input[type=range]{flex:1 1 280px;accent-color:var(--accent)}
.regroup .val{font-weight:600;min-width:150px}
.regroup p{margin:6px 0 0;font-size:13px;color:var(--muted)}
.warn{background:#33291a;border:1px solid #5c4626;border-radius:8px;padding:8px 12px;font-size:13px;margin:0 0 14px}

/* Barre d'outils collante */
.bar{position:sticky;top:0;z-index:5;background:rgba(18,18,18,.96);backdrop-filter:blur(4px);
  border-bottom:1px solid var(--line);padding:10px 0;margin-bottom:12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.progress{min-width:220px;flex:1}
.progress .txt{font-size:13px;color:#ccc}
.progress .track{height:6px;background:var(--panel2);border-radius:3px;overflow:hidden;margin-top:4px}
.progress .fill{height:100%;background:var(--accent);width:0}
.filters{display:flex;gap:6px;flex-wrap:wrap}
.filters button{background:var(--panel2);color:#ddd;border:1px solid var(--line);border-radius:999px;padding:4px 11px;font-size:13px;cursor:pointer}
.filters button[aria-pressed=true]{background:#eee;color:#111;border-color:#eee}
.bar input[type=search]{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 9px;font-size:14px;width:170px}
.btn{background:#2a6;color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer;font-weight:600}
.btn.secondary{background:var(--panel2);color:#ddd;border:1px solid var(--line);font-weight:400}
.btn.merge{background:var(--sel);color:#111}
.btn[hidden]{display:none}

/* Carte d'un groupe */
.group{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--normal);border-radius:8px;padding:12px;margin:0 0 12px}
.group.revoir{border-left-color:var(--revoir)}.group.theme{border-left-color:var(--theme)}
.group.unseen{border-style:dashed;border-left-style:solid}
.group.selected{outline:2px solid var(--sel);outline-offset:-1px}
.ghead{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;margin-bottom:6px}
.num{color:var(--muted);font-size:13px;min-width:110px;display:flex;align-items:center;gap:8px}
.num b{color:var(--text);font-size:15px}
.num input{width:18px;height:18px;accent-color:var(--sel);cursor:pointer;margin:0}
.name{display:flex;flex-direction:column;gap:2px}
.name label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.name input{font-size:15px;padding:5px 8px;width:240px;background:var(--panel2);color:var(--text);border:1px solid #555;border-radius:5px}
.seg{display:inline-flex;border:1px solid #555;border-radius:7px;overflow:hidden}
.seg button{background:var(--panel2);color:#ccc;border:0;border-right:1px solid #555;padding:7px 12px;font-size:14px;cursor:pointer}
.seg button:last-child{border-right:0}
.seg button:hover{background:#2d2d2d}
.seg button[aria-pressed=true][data-a=garder]{background:var(--normal);color:#111;font-weight:600}
.seg button[aria-pressed=true][data-a=revoir]{background:var(--revoir);color:#111;font-weight:600}
.seg button[aria-pressed=true][data-a=theme]{background:var(--theme);color:#111;font-weight:600}
.effect{font-size:13px;color:#ccc;margin:2px 0 6px}
.effect code{color:#fff}
.effect .merge{color:var(--muted)}
.sugg{font-size:12px;color:var(--muted)}
.sugg a{color:var(--accent);cursor:pointer;margin-left:6px;text-decoration:none}
.sugg a:hover{text-decoration:underline}
.parts{font-size:12px;color:var(--muted);margin:0 0 8px;display:flex;flex-wrap:wrap;gap:4px 10px}
.parts span{background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:1px 4px 1px 9px}
.parts a{color:var(--accent);cursor:pointer;margin-left:4px;text-decoration:none;padding:0 4px}
.thumbs{display:flex;flex-wrap:wrap;gap:4px}
.thumbs img{width:104px;height:78px;object-fit:cover;border-radius:3px;background:#222;transition:transform .12s;cursor:zoom-in}
.thumbs img:hover{transform:scale(2.2);position:relative;z-index:3;box-shadow:0 4px 18px #000}
.more{margin-top:8px;background:none;border:1px solid var(--line);color:var(--accent);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:13px}
.empty{color:var(--muted);padding:30px 0;text-align:center}

dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;max-width:640px;padding:18px 20px}
dialog::backdrop{background:rgba(0,0,0,.6)}
dialog pre{background:var(--panel2);padding:10px;border-radius:6px;white-space:pre-wrap;word-break:break-all;font-size:13px}
.restored{background:#1d2a33;border:1px solid #2b4657;border-radius:8px;padding:8px 12px;font-size:13px;margin:0 0 12px;display:none;align-items:center;gap:10px;flex-wrap:wrap}
.restored button{font-size:12px}
@media (max-width:700px){.bar{position:static}.name input{width:100%}.name{flex:1 1 100%}}
</style>
<main>
<h1>Groupes de photos qui se ressemblent</h1>
<p class=sub id=intro></p>

<div class=choices>
  <div class="choice c-garder"><h3>Tri normal</h3>
    <p>Le groupe n'a aucun effet : ses photos sont rangées comme toutes les autres, par voyage ou par mois.
       C'est le choix par défaut, pour les groupes sans intérêt particulier (paysages, repas, amis…).</p>
    <div class=dest>→ Voyages/… ou 2026/2026-05/</div></div>
  <div class="choice c-revoir"><h3>À revoir</h3>
    <p>Photos probablement inutiles (captures d'écran, reçus, photos ratées…) : elles sont mises de côté
       pour que tu les supprimes ou les gardes après un coup d'œil.</p>
    <div class=dest>→ _A_REVOIR/&lt;nom&gt;/</div></div>
  <div class="choice c-theme"><h3>Dossier thème</h3>
    <p>Un sujet que tu veux retrouver ensemble (vélo, bateaux…), quelle que soit la date.
       Exception : une photo prise pendant un voyage reste dans le dossier du voyage.</p>
    <div class=dest>→ Themes/&lt;nom&gt;/</div></div>
</div>
<p class=note>Le <b>nom</b> ne sert que pour « À revoir » et « Dossier thème » : c'est le nom du dossier.
Deux groupes avec le même nom vont dans le même dossier. Tes choix s'appliqueront aussi aux futures photos qui ressemblent à ces groupes.
Les photos isolées (hors groupe) suivent le tri normal.</p>

<div class=warn id=partial hidden>Cette page vient d'une ancienne découverte qui ne gardait que 18 photos par groupe.
Pour voir toutes les photos, relance <code id=rerunCmd></code> : c'est rapide (environ une minute), l'analyse des images
est déjà en cache, et tes choix seront repris.</div>

<section class=regroup>
  <h2>Grosseur des groupes</h2>
  <div class=row>
    <span>Fins</span>
    <input type=range id=level min=0 value=0 aria-label="Regrouper les groupes qui se ressemblent">
    <span>Larges</span>
    <span class=val id=levelTxt></span>
  </div>
  <p>Vers la droite, les groupes qui se ressemblent le plus sont réunis (comme une fusion manuelle, annulable groupe par groupe).
     Pour des groupes plus fins qu'au départ, relance <code id=grainCmd></code> : environ une minute, sans refaire l'analyse des images.
     Tu peux aussi cocher des groupes et cliquer « Fusionner ».</p>
</section>

<div class=restored id=restored><span id=restoredTxt></span>
  <button class="btn secondary" id=resetBtn>Repartir des choix d'origine</button></div>

<div class=bar>
  <div class=progress><div class=txt id=progressTxt></div><div class=track><div class=fill id=progressFill></div></div></div>
  <div class=filters id=filters></div>
  <input type=search id=search placeholder="Chercher un nom…">
  <button class="btn merge" id=mergeBtn hidden>Fusionner</button>
  <button class="btn secondary" id=unselectBtn hidden>Désélectionner</button>
  <label class="btn secondary" title="Reprendre les choix d'un categories.json déjà enregistré">Charger…
    <input type=file id=load accept=".json,application/json" hidden></label>
  <button class=btn id=save>Enregistrer categories.json</button>
</div>

<div id=list></div>
<p class=empty id=empty hidden>Aucun groupe ne correspond au filtre.</p>
</main>

<dialog id=done>
  <h3 style="margin-top:0">categories.json enregistré</h3>
  <p>Le navigateur l'a mis dans ton dossier de téléchargements. Lance ensuite l'analyse en lui donnant ce fichier
     (adapte le chemin si besoin) :</p>
  <pre id=cmd></pre>
  <p style="font-size:13px;color:var(--muted)">L'analyse en garde une copie dans <code>.tri_photos/</code> :
     les suivantes, et la prochaine découverte, la réutilisent automatiquement.</p>
  <form method=dialog style="text-align:right"><button class=btn>OK</button></form>
</dialog>

<script>
const DATA = __DATA__;
const C = DATA.clusters;                       // groupes trouvés par le script (jamais modifiés en structure)
const N = C.length;
const LABEL = {garder: 'Tri normal', revoir: 'À revoir', theme: 'Dossier thème'};
const STORE = 'decouverte:' + DATA.source;     // une mémoire par dossier, qui survit à une nouvelle découverte
const SHOWN = 18;
C.forEach((c, i) => { c.i = i; c.photos = c.photos || c.exemples || []; c.vu = !!c.vu; });
const byId = Object.fromEntries(C.map(c => [c.id, c]));

// État de la page : choix par groupe (dans C), niveau de regroupement, fusions et séparations manuelles.
let level = 0, links = [], detached = new Set(), selected = new Set(), expanded = new Set();
let filter = 'tous', query = '';

function dot(a, b) { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; }
function safeName(s) {  // même règle que le script (nom du dossier)
  const k = s.replace(/[^\p{L}\p{N} _-]/gu, ' ').trim().split(/\s+/).join('_').slice(0, 60);
  return k || 'groupe';
}
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else if (k.startsWith('on')) e[k] = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

// --- Regroupement hiérarchique des groupes (moyenne des similarités entre « portraits moyens »).
// MERGES[k] = deux ensembles de groupes réunis à l'étape k, du plus ressemblant au moins ressemblant.
const SIM = C.map(a => C.map(b => dot(a.centre, b.centre)));
const MERGES = (() => {
  let sets = C.map(c => [c.i]);
  const out = [];
  while (sets.length > 1) {
    let best = -2, bi = 0, bj = 1;
    for (let i = 0; i < sets.length; i++) for (let j = i + 1; j < sets.length; j++) {
      let s = 0;
      for (const a of sets[i]) for (const b of sets[j]) s += SIM[a][b];
      s /= sets[i].length * sets[j].length;
      if (s > best) { best = s; bi = i; bj = j; }
    }
    out.push({a: sets[bi], b: sets[bj], sim: best});
    const merged = sets[bi].concat(sets[bj]);
    sets = sets.filter((_, k) => k !== bi && k !== bj); sets.push(merged);
  }
  return out;
})();

// --- Partition effective : fusions du curseur + fusions manuelles, moins les groupes séparés
let comps = [];  // [{master, members:[...]}], master = le plus gros groupe de la fusion
function partition() {
  const parent = C.map(c => c.i);
  const find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const union = (a, b) => { if (a != null && b != null) parent[find(a)] = find(b); };
  const free = list => list.find(x => !detached.has(C[x].id));
  for (let k = 0; k < level; k++) union(free(MERGES[k].a), free(MERGES[k].b));
  for (const [a, b] of links) if (byId[a] && byId[b]) union(byId[a].i, byId[b].i);
  const groups = {};
  for (const c of C) (groups[find(c.i)] ||= []).push(c);
  comps = Object.values(groups).map(m => {
    m.sort((x, y) => y.nb - x.nb || x.i - y.i);
    return {master: m[0], members: m, nb: m.reduce((s, c) => s + c.nb, 0)};
  }).sort((x, y) => x.master.i - y.master.i);
}

// --- Mémoire du navigateur : rien n'est perdu si l'onglet est fermé ou la découverte relancée
function persist() {
  try {
    localStorage.setItem(STORE, JSON.stringify({
      ids: C.map(c => c.id).join(','), level, links, detached: [...detached],
      choix: C.map(c => ({id: c.id, nom: c.nom, action: c.action, vu: c.vu,
                          centre: c.centre.map(x => Math.round(x * 1000) / 1000)})),
    }));
  } catch (e) {}
}
function restore() {
  let s = null;
  try { s = JSON.parse(localStorage.getItem(STORE) || 'null'); } catch (e) {}
  if (!s || !s.choix) return 0;
  let n = 0;
  const known = Object.fromEntries(s.choix.map(x => [x.id, x]));
  for (const c of C) {
    let prev = known[c.id];
    if (!prev) {  // nouvelle découverte : le groupe d'avant le plus ressemblant, s'il est assez proche
      let bs = 0.9;
      for (const x of s.choix) { if (!x.centre) continue; const v = dot(c.centre, x.centre); if (v > bs) { bs = v; prev = x; } }
    }
    if (prev && (prev.nom !== c.nom || prev.action !== c.action || prev.vu !== c.vu)) {
      Object.assign(c, {nom: prev.nom, action: prev.action, vu: prev.vu}); n++;
    }
  }
  if (s.ids === C.map(c => c.id).join(',')) {  // même découverte : on reprend aussi les fusions
    level = s.level || 0; links = s.links || []; detached = new Set(s.detached || []);
  }
  return n;
}

// --- Rendu
function effectText(comp) {
  const c = comp.master;
  if (c.action === 'garder') return 'Ses photos sont rangées normalement, <b>par voyage ou par mois</b>.';
  const dir = (c.action === 'revoir' ? '_A_REVOIR/' : 'Themes/') + safeName(c.nom) + '/';
  const same = comps.filter(o => o !== comp && o.master.action === c.action && safeName(o.master.nom) === safeName(c.nom))
                    .map(o => '#' + (o.master.i + 1));
  return `Ses photos iront dans <code>${esc(dir)}</code>`
    + (c.action === 'theme' ? ' (sauf celles prises pendant un voyage)' : '')
    + (same.length ? ` <span class=merge>— même dossier que ${same.slice(0, 6).join(', ')}${same.length > 6 ? '…' : ''}</span>` : '');
}
function visible(comp) {
  const c = comp.master;
  return (filter === 'tous' || (filter === 'nonvus' ? !c.vu : c.action === filter))
    && (!query || comp.members.some(m => m.nom.toLowerCase().includes(query) || m.suggestions.some(s => s.toLowerCase().includes(query))));
}
function card(comp) {
  const c = comp.master, multi = comp.members.length > 1;
  const box = el('input', {type: 'checkbox', 'aria-label': 'Sélectionner pour fusionner'});
  box.checked = selected.has(c.id);
  box.onchange = () => { box.checked ? selected.add(c.id) : selected.delete(c.id); updateSelection(); sec.classList.toggle('selected', box.checked); };
  const input = el('input', {value: c.nom, 'aria-label': 'Nom du groupe ' + (c.i + 1)});
  const effect = el('div', {class: 'effect'});
  effect.innerHTML = effectText(comp);
  input.oninput = () => { c.nom = input.value.trim(); c.vu = true; persist(); effect.innerHTML = effectText(comp); updateCounts(); };
  const seg = el('div', {class: 'seg', role: 'group', 'aria-label': 'Que faire de ce groupe ?'});
  for (const a of ['garder', 'revoir', 'theme']) {
    seg.append(el('button', {'data-a': a, type: 'button', 'aria-pressed': c.action === a,
                             onclick: () => { c.action = a; c.vu = true; changed(); }}, LABEL[a]));
  }
  const sugg = el('div', {class: 'sugg'}, 'Suggestions :');
  const sset = [...new Set(comp.members.flatMap(m => m.suggestions))].slice(0, 5);
  for (const s of sset) sugg.append(el('a', {onclick: () => { c.nom = s; c.vu = true; changed(); }}, s));
  const head = el('div', {class: 'ghead'},
    el('div', {class: 'num'}, box, el('b', {}, '#' + (c.i + 1)), comp.nb + ' photos'),
    el('div', {class: 'name'}, el('label', {}, 'Nom (dossier)'), input), seg, sugg);
  const sec = el('section', {class: 'group ' + c.action + (c.vu ? '' : ' unseen') + (selected.has(c.id) ? ' selected' : ''), id: 'g' + (c.i + 1)}, head, effect);
  if (multi) {
    const parts = el('div', {class: 'parts'}, `Fusion de ${comp.members.length} groupes :`);
    for (const m of comp.members) {
      parts.append(el('span', {}, `#${m.i + 1} · ${m.nb} photos`,
        el('a', {title: 'Sortir ce groupe de la fusion', onclick: () => split(m, comp)}, '✕ séparer')));
    }
    sec.append(parts);
  }
  const all = comp.members.flatMap(m => m.photos);
  const thumbs = el('div', {class: 'thumbs'});
  const open = expanded.has(c.id);
  for (const p of open ? all : all.slice(0, SHOWN)) thumbs.append(el('img', {loading: 'lazy', src: p, alt: ''}));
  sec.append(thumbs);
  const partial = comp.members.some(m => m.partiel);
  if (all.length > SHOWN) {
    sec.append(el('button', {class: 'more', type: 'button', onclick: () => { open ? expanded.delete(c.id) : expanded.add(c.id); render(); }},
      open ? 'Réduire' : `Voir les ${all.length} photos`));
  } else if (partial && comp.nb > all.length) {
    sec.append(el('p', {class: 'sugg'}, `${all.length} photos montrées sur ${comp.nb} (ancienne découverte, voir l'encadré en haut).`));
  }
  return sec;
}
function render() {
  partition();
  const list = document.getElementById('list');
  list.replaceChildren(...comps.filter(visible).map(card));
  document.getElementById('empty').hidden = list.children.length > 0;
  updateCounts(); updateSelection();
  document.getElementById('level').value = level;
  document.getElementById('levelTxt').textContent =
    level ? `${comps.length} groupes (au lieu de ${N})` : `${comps.length} groupes (découverte d'origine)`;
}
function updateCounts() {
  const counts = {garder: 0, revoir: 0, theme: 0}; let seen = 0;
  for (const comp of comps) { counts[comp.master.action]++; if (comp.master.vu) seen++; }
  document.getElementById('progressTxt').textContent =
    `${seen} / ${comps.length} groupes passés en revue · ${counts.revoir} à revoir · ${counts.theme} thème(s) · ${counts.garder} tri normal`;
  document.getElementById('progressFill').style.width = (100 * seen / comps.length) + '%';
  const f = document.getElementById('filters');
  f.replaceChildren(...[['tous', `Tous (${comps.length})`], ['nonvus', `Pas encore vus (${comps.length - seen})`],
    ['revoir', `À revoir (${counts.revoir})`], ['theme', `Thème (${counts.theme})`], ['garder', `Tri normal (${counts.garder})`]]
    .map(([k, lab]) => el('button', {type: 'button', 'aria-pressed': filter === k, onclick: () => { filter = k; render(); }}, lab)));
}
function updateSelection() {
  const n = selected.size;
  const mb = document.getElementById('mergeBtn');
  mb.hidden = n < 2; mb.textContent = `Fusionner (${n})`;
  document.getElementById('unselectBtn').hidden = n === 0;
}
function changed() { persist(); render(); }

// --- Fusionner / séparer
document.getElementById('mergeBtn').onclick = () => {
  const ids = [...selected];
  for (let k = 1; k < ids.length; k++) links.push([ids[0], ids[k]]);
  for (const id of ids) detached.delete(id);
  selected.clear(); partition();
  const comp = comps.find(x => x.members.some(m => m.id === ids[0]));
  comp.master.vu = true;
  changed();
  document.getElementById('g' + (comp.master.i + 1))?.scrollIntoView({block: 'center'});
};
document.getElementById('unselectBtn').onclick = () => { selected.clear(); render(); };
function split(m, comp) {
  if (m === comp.master) {  // le plus gros sort : les autres gardent les choix faits sur la fusion
    for (const o of comp.members) if (o !== m) Object.assign(o, {nom: m.nom, action: m.action, vu: m.vu});
  }
  detached.add(m.id);
  links = links.filter(([a, b]) => a !== m.id && b !== m.id);
  changed();
}
document.getElementById('level').oninput = ev => { level = +ev.target.value; detached.clear(); changed(); };

// --- Enregistrer / charger categories.json
document.getElementById('save').onclick = () => {
  partition();
  const out = [];
  for (const comp of comps) for (const m of comp.members) {
    const {photos, exemples, partiel, i, ...r} = m;
    out.push({...r, nom: comp.master.nom, action: comp.master.action, vu: comp.master.vu,
              ...(comp.members.length > 1 ? {fusion: comp.master.id} : {})});
  }
  const a = el('a', {href: URL.createObjectURL(new Blob([JSON.stringify(out)], {type: 'application/json'})), download: 'categories.json'});
  a.click();
  document.getElementById('cmd').textContent =
    `python ${DATA.script} analyse "${DATA.source}" --categories ~/Téléchargements/categories.json`;
  document.getElementById('done').showModal();
};
document.getElementById('load').onchange = async ev => {
  const file = ev.target.files[0]; if (!file) return;
  let prev; try { prev = JSON.parse(await file.text()); } catch (e) { alert('Fichier illisible.'); return; }
  let n = 0;
  for (const c of C) {  // même logique que le script : le groupe le plus ressemblant, si assez proche
    let best = null, bs = 0.9;
    for (const p of prev) { if (!p.centre) continue; const s = dot(c.centre, p.centre); if (s > bs) { bs = s; best = p; } }
    if (best) { c.nom = best.nom; c.action = best.action || 'garder'; c.vu = true; n++; }
  }
  changed();
  alert(`${n} groupe(s) sur ${N} ont repris les choix de ${file.name}.`);
  ev.target.value = '';
};
document.getElementById('search').oninput = ev => { query = ev.target.value.trim().toLowerCase(); render(); };
document.getElementById('resetBtn').onclick = () => {
  if (!confirm('Oublier les choix et fusions faits dans ce navigateur et repartir de la page d\'origine ?')) return;
  try { localStorage.removeItem(STORE); } catch (e) {}
  location.reload();
};

// --- Démarrage
const grain = DATA.grain || 0.25;
document.getElementById('intro').textContent =
  `${DATA.total} photos analysées dans ${DATA.source} : ${N} groupes de photos qui se ressemblent`
  + ` (+ ${DATA.small} photos isolées). Pour chaque groupe, choisis ce qu'il faut en faire.`;
document.getElementById('grainCmd').textContent =
  `python ${DATA.script} decouvrir "${DATA.source}" --grain ${Math.max(0.1, Math.round((grain - 0.05) * 100) / 100)}`;
document.getElementById('rerunCmd').textContent = `python ${DATA.script} decouvrir "${DATA.source}"`;
document.getElementById('partial').hidden = !C.some(c => c.partiel);
document.getElementById('level').max = N - 1;
const n = restore();
if (n) {
  document.getElementById('restoredTxt').textContent = `Tes choix précédents ont été restaurés (${n} groupe(s) repris de ce navigateur).`;
  document.getElementById('restored').style.display = 'flex';
}
render();
</script>
"""


def apply_discovered(images: list[Photo], path: Path) -> None:
    cats = [c for c in json.loads(path.read_text()) if c.get("action") in ("revoir", "theme")]
    if not cats:
        return
    C = np.stack([np.array(c["centre"]) for c in cats])
    seuils = np.array([c["seuil"] for c in cats])
    n = 0
    for p in images:
        if p.emb is None:
            continue
        sims = C @ p.emb
        i = int(np.argmax(sims))
        if sims[i] < seuils[i]:
            continue
        nom = safe_name(cats[i]["nom"]) or "groupe"
        if cats[i]["action"] == "revoir":
            p.category, p.category_score = nom, 2.0
        else:
            p.theme = nom
        n += 1
    log(f"Catégories découvertes : {n} photos rattachées à {len(cats)} groupe(s) nommé(s).")


# ---------------------------------------------------------------------------
# Application / annulation
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse", help="analyse et produit plan.csv + rapport.html")
    add_common_analyse_args(a, "photos")
    a.add_argument("--trip-min-photos", type=int, default=8)
    a.add_argument("--keep", type=int, default=1, help="photos gardées par groupe de similaires")
    a.add_argument("--dup-threshold", type=float, default=0.93,
                   help="similarité CLIP (0-1) pour considérer deux photos comme quasi-identiques")
    a.add_argument("--dup-window-min", type=float, default=10,
                   help="écart max (minutes) entre deux quasi-doublons")
    a.add_argument("--category-threshold", type=float, default=0.5,
                   help="confiance min. pour classer dans une catégorie à revoir")
    a.add_argument("--blur-threshold", type=float, default=60,
                   help="variance du laplacien sous laquelle une photo est floue")
    a.add_argument("--categories", help="categories.json produit par la commande decouvrir "
                   f"(recopié dans .tri_photos/ et réutilisé ensuite automatiquement)")
    a.add_argument("--model", default="ViT-B-32")
    a.add_argument("--pretrained", default="laion2b_s34b_b79k")
    a.set_defaults(func=analyse)
    d = sub.add_parser("decouvrir", help="regroupe les photos par ressemblance pour que tu nommes les groupes")
    d.add_argument("source", help="dossier de photos")
    d.add_argument("--grain", type=float, default=0.25,
                   help="plus petit = groupes plus nombreux et plus précis (0.15-0.35)")
    d.add_argument("--min-group", type=int, default=6, help="taille minimale d'un groupe affiché")
    d.add_argument("--page-seule", action="store_true",
                   help="régénérer seulement decouverte.html à partir de la dernière découverte "
                        "(rien n'est recalculé, aucun modèle chargé)")
    d.add_argument("--model", default="ViT-B-32")
    d.add_argument("--pretrained", default="laion2b_s34b_b79k")
    d.set_defaults(func=discover)
    PLAN.add_commands(sub)
    args = ap.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
