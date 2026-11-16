#!/usr/bin/env python3
"""
Pré-tri de vidéos 100 % local, SANS décoder les vidéos.

Tout repose sur les métadonnées lues par exiftool (un appel pour tout le
dossier), le nom des fichiers et leur taille. Aucune image n'est extraite,
aucun modèle n'est chargé : quelques secondes pour des milliers de vidéos.

  1. À revoir :
       illisibles          exiftool ne trouve pas de durée (fichier tronqué/corrompu)
       doublons            copies exactes (même contenu octet par octet)
       live_photos         petites vidéos des Live Photos iPhone (la photo suffit)
       enregistrements_ecran  captures vidéo de l'écran
       similaires/         ré-encodages d'une même vidéo (copie envoyée par
                           WhatsApp...) : on garde la meilleure qualité
       messagerie          vidéos reçues (WhatsApp, Telegram, Messenger, Signal)
       tres_courtes        vidéos déclenchées par erreur (< 2 s par défaut)
  2. Rangement : mêmes dossiers que tri_photos.py. Si le plan.csv des photos
     est présent, une vidéo prise pendant un voyage détecté par les photos va
     dans le même dossier Voyages/..., même sans GPS. Sinon : voyage détecté
     par le GPS des vidéos, ou dossier du mois.

analyse ne touche à AUCUNE vidéo : il construit un plan (plan_videos.csv +
rapport_videos.html) dans <dossier>/.tri_photos/, le même dossier de travail que
tri_photos.py (noms de fichiers distincts). Si photos et vidéos sont dans le même
dossier, les voyages des photos sont donc retrouvés automatiquement.
apply exécute le plan (sur place, ou vers --out), undo annule le dernier apply.

Ajout de vidéos : comme tri_photos.py, les vidéos déjà rangées (_A_REVOIR/, Voyages/,
Themes/, AAAA/AAAA-MM/) ne bougent pas, les nouvelles rejoignent les voyages et
mois existants. --tout-refaire re-trie tout depuis zéro.

Usage :
  python tri_videos.py analyse ~/Videos/A_trier [--plan-photos ~/Photos/A_trier]
  python tri_videos.py apply   ~/Videos/A_trier      # range sur place selon le plan
  python tri_videos.py apply   ~/Videos/A_trier --out ~/Photos/Trie [--copy]
  python tri_videos.py undo    ~/Videos/A_trier      # annule le dernier apply
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from tri_commun import (IMAGE_EXT, PLAN_DIR, Plan, add_common_analyse_args, atomic_open,
                        atomic_write_text, find_trips, log, parse_exif_date, run_command,
                        run_exiftool, settle_group, similar_folder, sorted_folder, trip_label,
                        work_dir)

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".3gp", ".avi", ".mkv", ".mts", ".m2ts",
             ".wmv", ".mpg", ".mpeg", ".webm"}

PLAN_CSV = "plan_videos.csv"
REPORT = "rapport_videos.html"
PLAN = Plan(PLAN_CSV, "deplacements_videos_", "vidéos")


# Motifs de noms de fichiers (insensibles à la casse). Ajoute les tiens ici.
SCREEN_PATTERNS = [r"screen.?record", r"rpreplay", r"screencast", r"enregistrement.*[ée]cran",
                   r"capture.*vid[ée]o", r"^record_\d", r"screen-\d"]
MESSAGING_PATTERNS = [r"^vid-\d{8}-wa\d+", r"whatsapp", r"telegram", r"^video_\d{4}-\d{2}-\d{2}",
                      r"^signal-\d{4}", r"messenger", r"^received_\d+", r"snapchat"]
# Dossiers dont le nom suffit à dire « reçue par messagerie »
MESSAGING_DIRS = {"whatsapp video", "whatsapp", "telegram", "telegram video", "messenger", "signal"}


@dataclass
class Video:
    path: str
    size: int = 0
    date: str = ""                 # ISO, heure locale de la prise de vue
    date_source: str = ""
    lat: float | None = None
    lon: float | None = None
    make: str = ""
    model: str = ""
    software: str = ""
    duration: float = 0.0          # secondes
    width: int = 0
    height: int = 0
    fps: float = 0.0
    bitrate: float = 0.0           # bits/s
    content_id: str = ""           # identifiant Live Photo (Apple)
    error: str = ""
    category: str = ""
    dup_group: int | None = None
    dup_keep: bool = True
    dup_folder: str = ""           # _A_REVOIR/similaires/<vidéo de référence>
    pinned: str = ""               # dossier actuel si déjà rangée (on n'y touche pas)
    destination: str = ""
    reason: str = ""

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def has_meta_date(self) -> bool:
        return self.date_source not in ("fichier", "nom (jour)", "nom")

    @property
    def aspect(self) -> float:
        w, h = sorted((self.width, self.height))
        return h / w if w else 0.0


# ---------------------------------------------------------------------------
# Métadonnées
# ---------------------------------------------------------------------------


NAME_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_]?(\d{2})[-_]?(\d{2})"
                       r"(?:[-_ T]?(\d{2})[-_.:h]?(\d{2})[-_.:m]?(\d{2}))?(?!\d)")


def date_from_name(name: str) -> tuple[dt.datetime, str] | None:
    """VID_20240712_140322.mp4, PXL_20240712_..., VID-20240712-WA0001.mp4..."""
    for m in NAME_DATE.finditer(name):
        y, mo, d, hh, mi, ss = m.groups()
        try:
            if hh:
                return dt.datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss)), "nom"
            return dt.datetime(int(y), int(mo), int(d), 12, 0, 0), "nom (jour)"
        except ValueError:
            continue
    return None


def read_metadata(files: list[Path], quicktime_utc: bool, cache: Path) -> dict[str, dict]:
    if not shutil.which("exiftool"):
        sys.exit("exiftool est indispensable : sudo apt install libimage-exiftool-perl")
    tags = ["-CreationDate", "-DateTimeOriginal", "-CreateDate", "-MediaCreateDate",
            "-TrackCreateDate", "-GPSLatitude", "-GPSLongitude", "-GPSCoordinates",
            "-Make", "-Model", "-Software", "-Encoder", "-Duration", "-ImageWidth",
            "-ImageHeight", "-Rotation", "-VideoFrameRate", "-AvgBitrate",
            "-ContentIdentifier", "-Error"]
    # Les dates QuickTime (CreateDate...) sont stockées en UTC : exiftool les
    # convertit dans le fuseau de cet ordinateur. CreationDate (iPhone) porte
    # déjà son propre fuseau et reste en heure locale du lieu de prise de vue.
    return run_exiftool(files, tags, ["-api", "QuickTimeUTC=1"] if quicktime_utc else [],
                        cache=cache)


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def build_video(f: Path, m: dict) -> Video:
    v = Video(path=str(f), size=f.stat().st_size)
    d = None
    for k in ("CreationDate", "DateTimeOriginal", "CreateDate", "MediaCreateDate",
              "TrackCreateDate"):
        d = parse_exif_date(m.get(k))
        if d:
            v.date_source = k
            break
    if not d:
        r = date_from_name(f.name)
        if r:
            d, v.date_source = r
    if not d:
        d, v.date_source = dt.datetime.fromtimestamp(f.stat().st_mtime), "fichier"
    v.date = d.isoformat()

    lat, lon = m.get("GPSLatitude"), m.get("GPSLongitude")
    if not isinstance(lat, (int, float)) and isinstance(m.get("GPSCoordinates"), str):
        parts = m["GPSCoordinates"].split()
        if len(parts) >= 2:
            lat, lon = num(parts[0]), num(parts[1])
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and (lat or lon):
        v.lat, v.lon = float(lat), float(lon)

    v.make = str(m.get("Make") or "").strip()
    v.model = str(m.get("Model") or "").strip()
    v.software = str(m.get("Software") or m.get("Encoder") or "").strip()
    v.duration = num(m.get("Duration"))
    v.width, v.height = int(num(m.get("ImageWidth"))), int(num(m.get("ImageHeight")))
    if int(num(m.get("Rotation"))) in (90, 270):
        v.width, v.height = v.height, v.width
    v.fps = num(m.get("VideoFrameRate"))
    v.bitrate = num(m.get("AvgBitrate")) or (v.size * 8 / v.duration if v.duration else 0.0)
    v.content_id = str(m.get("ContentIdentifier") or "")
    v.error = str(m.get("Error") or "")
    return v


# ---------------------------------------------------------------------------
# Doublons
# ---------------------------------------------------------------------------
def file_hash(path: str, full: bool) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        if full:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        else:  # empreinte rapide : début + fin
            h.update(f.read(1 << 20))
            f.seek(max(0, os.path.getsize(path) - (1 << 20)))
            h.update(f.read(1 << 20))
    return h.hexdigest()


def exact_duplicates(videos: list[Video]) -> list[list[Video]]:
    """Même taille -> même empreinte rapide -> même hash complet."""
    by_size = collections.defaultdict(list)
    for v in videos:
        by_size[v.size].append(v)
    groups = []
    for same in by_size.values():
        if len(same) < 2:
            continue
        quick = collections.defaultdict(list)
        for v in same:
            quick[file_hash(v.path, False)].append(v)
        for cand in quick.values():
            if len(cand) < 2:
                continue
            full = collections.defaultdict(list)
            for v in cand:
                full[file_hash(v.path, True)].append(v)
            groups += [g for g in full.values() if len(g) > 1]
    return groups


def near_duplicates(videos: list[Video], tol_s: float, min_dur: float) -> list[list[Video]]:
    """Ré-encodages : même durée (à tol_s près), même orientation/format, et
    prises au même moment (ou le même jour si l'une n'a qu'une date de nom/fichier)."""
    vs = sorted((v for v in videos if v.duration >= min_dur), key=lambda v: v.duration)
    parent = list(range(len(vs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ts = [dt.datetime.fromisoformat(v.date) for v in vs]
    for i in range(len(vs)):
        j = i + 1
        while j < len(vs) and vs[j].duration - vs[i].duration <= tol_s:
            a, b = vs[i], vs[j]
            same_shape = ((a.width >= a.height) == (b.width >= b.height)
                          and abs(a.aspect - b.aspect) < 0.05)
            if a.has_meta_date and b.has_meta_date:
                close = abs(ts[i] - ts[j]) <= dt.timedelta(minutes=2)
            else:
                close = abs(ts[i] - ts[j]) <= dt.timedelta(days=1)
            if same_shape and close:
                parent[find(j)] = find(i)
            j += 1
    groups = collections.defaultdict(list)
    for i, v in enumerate(vs):
        groups[find(i)].append(v)
    return [g for g in groups.values() if len(g) > 1]


def quality_rank(v: Video) -> tuple:
    """La meilleure copie : métadonnées d'origine, puis résolution, puis débit."""
    return (v.has_meta_date, v.lat is not None, v.width * v.height, v.bitrate, -len(v.path))


# ---------------------------------------------------------------------------
# Catégories
# ---------------------------------------------------------------------------
def matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def image_stems(files: list[Path]) -> set[tuple[str, str]]:
    return {(str(f.parent), f.stem.lower()) for f in files}


def classify(v: Video, stems: set, args) -> tuple[str, str]:
    p = Path(v.path)
    if v.error or v.duration <= 0:
        return "illisibles", v.error or "durée introuvable"
    if v.duration <= args.live_max_s and (
            v.content_id or (str(p.parent), p.stem.lower()) in stems):
        return "live_photos", "Live Photo" + (" (photo à côté)" if not v.content_id else "")
    no_camera = not v.make and not v.model
    if matches(SCREEN_PATTERNS, p.name):
        return "enregistrements_ecran", "nom de fichier"
    if no_camera and v.aspect >= 1.95 and not v.lat:
        # caméras : 16:9 (1.78) ou 4:3 ; écrans de téléphone : 19.5:9 à 21:9
        return "enregistrements_ecran", f"format écran {v.width}x{v.height}, sans appareil"
    return "", ""


def is_messaging(v: Video) -> bool:
    p = Path(v.path)
    if any(part.lower() in MESSAGING_DIRS for part in p.parent.parts):
        return True
    return not v.make and matches(MESSAGING_PATTERNS, p.name)


# ---------------------------------------------------------------------------
# Géographie (identique à tri_photos.py)
# ---------------------------------------------------------------------------


def photo_trips(plan: Path, margin_h: float) -> list[tuple[dt.datetime, dt.datetime, str]]:
    """Intervalles [début, fin] des voyages trouvés par tri_photos.py."""
    spans: dict[str, list[dt.datetime]] = {}
    with open(plan, newline="") as f:
        for row in csv.DictReader(f):
            dest = row.get("destination", "")
            if not dest.startswith("Voyages/"):
                continue
            try:
                d = dt.datetime.fromisoformat(row["date"])
            except (KeyError, ValueError):
                continue
            s = spans.setdefault(dest, [d, d])
            s[0], s[1] = min(s[0], d), max(s[1], d)
    m = dt.timedelta(hours=margin_h)
    return sorted((a - m, b + m, dest) for dest, (a, b) in spans.items())


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
def analyse(args) -> None:
    src, out = work_dir(args.source)
    out.mkdir(exist_ok=True)
    all_files = [p for p in src.rglob("*") if p.is_file() and out not in p.parents]
    files = sorted(p for p in all_files if p.suffix.lower() in VIDEO_EXT)
    log(f"{len(files)} vidéos trouvées dans {src}")
    if not files:
        sys.exit("Aucune vidéo trouvée.")
    stems = image_stems([p for p in all_files if p.suffix.lower() in IMAGE_EXT])

    meta = read_metadata(files, not args.dates_locales, out / ".cache_exif_videos.json")
    videos = [build_video(f, meta.get(str(f), {})) for f in files]

    # --- vidéos déjà rangées : elles gardent leur place (sauf --tout-refaire)
    if not args.tout_refaire:
        for v in videos:
            v.pinned = sorted_folder(Path(v.path).relative_to(src)) or ""
    new = [v for v in videos if not v.pinned]
    if len(new) < len(videos):
        log(f"{len(videos) - len(new)} vidéos déjà rangées (laissées en place), "
            f"{len(new)} nouvelles à ranger.")

    def already(ref: Video) -> str:
        return " (déjà rangée)" if ref.pinned else ""

    # --- 1. copies exactes (nouvelles vidéos comparées aussi aux vidéos déjà rangées)
    gid = 0
    for g in exact_duplicates(videos):
        settled = settle_group(g, 1, gid + 1, quality_rank)
        if not settled:
            continue
        gid += 1
        ref, losers = settled
        for v in losers:
            v.category, v.reason = "doublons", f"copie exacte de {ref.name}{already(ref)}"

    # --- 2. illisibles, Live Photos, enregistrements d'écran
    for v in new:
        if not v.category:
            v.category, v.reason = classify(v, stems, args)

    # --- 3. ré-encodages d'une même vidéo (on garde la meilleure). Le dossier du
    # groupe porte le nom de la vidéo de référence, ou réutilise un similaires/ existant.
    pool = [v for v in videos
            if (not v.pinned and not v.category)
            or (v.pinned and (not v.pinned.startswith("_A_REVOIR")
                              or v.pinned.startswith("_A_REVOIR/similaires/")))]
    for g in near_duplicates(pool, args.dup_tolerance, args.dup_min_duration):
        settled = settle_group(g, args.keep, gid + 1, quality_rank)
        if not settled:
            continue
        gid += 1
        ref, losers = settled
        folder = similar_folder(g, ref)
        for v in losers:
            v.dup_folder = folder
            v.reason = (f"copie de {ref.name}{already(ref)} "
                        f"({v.width}x{v.height} vs {ref.width}x{ref.height})")

    # --- 4. messagerie et vidéos très courtes
    for v in new:
        if v.category or not v.dup_keep:
            continue
        if is_messaging(v):
            v.category, v.reason = "messagerie", "reçue par messagerie"
        elif v.duration < args.min_duration:
            v.category, v.reason = "tres_courtes", f"{v.duration:.1f} s"

    # --- 5. voyages / mois (nouvelles vidéos). Une vidéo prise pendant un voyage déjà
    # rangé (photos ou vidéos) rejoint son dossier, quel que soit son nom actuel.
    kept = sorted((v for v in new if not v.category and v.dup_keep), key=lambda v: v.date)
    ts = [dt.datetime.fromisoformat(v.date) for v in kept]
    assigned: dict[int, str] = {}
    margin = dt.timedelta(hours=args.trip_margin_hours)
    own: dict[str, list[dt.datetime]] = {}
    for v in videos:
        if v.pinned.startswith("Voyages/"):
            d = dt.datetime.fromisoformat(v.date)
            s = own.setdefault(v.pinned, [d, d])
            s[0], s[1] = min(s[0], d), max(s[1], d)
    video_spans = sorted((a - margin, b + margin, dest) for dest, (a, b) in own.items())
    for i, t in enumerate(ts):
        for a, b, dest in video_spans:
            if a <= t <= b:
                assigned[i] = dest
                break

    plan_photos = Path(args.plan_photos).expanduser() if args.plan_photos else out / "plan.csv"
    if plan_photos.is_dir():  # dossier de photos analysé plutôt que le plan.csv lui-même
        plan_photos = plan_photos / PLAN_DIR / "plan.csv"
        if not plan_photos.exists():
            sys.exit(f"Pas de plan des photos : {plan_photos} introuvable "
                     f"(lance d'abord tri_photos.py analyse sur ce dossier).")
    if plan_photos.exists():
        spans = photo_trips(plan_photos, args.trip_margin_hours)
        for i, t in enumerate(ts):
            if i in assigned:
                continue
            for a, b, dest in spans:
                if a <= t <= b:
                    assigned[i] = dest
                    break
        log(f"Voyages des photos ({plan_photos}) : {len(spans)} voyage(s), "
            f"{len(assigned)} vidéo(s) rattachée(s).")
    else:
        log("Pas de plan.csv des photos : voyages détectés uniquement par le GPS des vidéos.")

    home = (args.home_lat, args.home_lon)
    # vidéos pas encore rattachées à un voyage : voyages détectés par leur propre GPS
    rest = [i for i in range(len(kept)) if i not in assigned]
    found = find_trips([kept[i] for i in rest], [ts[i] for i in rest], home,
                       args.trip_km, args.trip_gap_days)
    for t in ([rest[k] for k in trip] for trip in found):
        if len(t) < args.trip_min_videos:
            continue
        coords = [(kept[i].lat, kept[i].lon) for i in t if kept[i].lat is not None]
        folder = f"Voyages/{ts[t[0]]:%Y-%m-%d}_Voyage_{trip_label(coords)}"
        for i in t:
            assigned[i] = folder

    for i, v in enumerate(kept):
        if i in assigned:
            v.destination, v.reason = assigned[i], "voyage"
        else:
            v.destination, v.reason = f"{ts[i]:%Y}/{ts[i]:%Y-%m}", "mois"
    for v in videos:
        if v.destination:
            continue
        if v.pinned:
            v.destination, v.reason = v.pinned, "déjà rangée"
        elif v.category:
            v.destination = f"_A_REVOIR/{v.category}"
        else:
            v.destination = v.dup_folder

    write_plan(out, videos)
    write_report(out, videos)
    summary(videos)
    log(f"\nPlan : {out/PLAN_CSV}\nRapport : {out/REPORT}\n"
        f"Rien n'a été déplacé. Vérifie le rapport puis : "
        f"python {Path(sys.argv[0]).name} apply {out.parent}  [--out DOSSIER] [--copy]")


# ---------------------------------------------------------------------------
# Sorties
# ---------------------------------------------------------------------------
def fmt_size(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024 or unit == "Go":
            return f"{n:.0f} {unit}" if unit in ("o", "Ko") else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def fmt_dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def write_plan(out: Path, videos: list[Video]) -> None:
    with atomic_open(out / PLAN_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "destination", "raison", "date", "date_source", "duree_s",
                    "largeur", "hauteur", "taille_octets", "appareil",
                    "groupe_similaires", "gardee"])
        for v in sorted(videos, key=lambda v: (v.destination, v.date)):
            w.writerow([v.path, v.destination, v.reason, v.date, v.date_source,
                        f"{v.duration:.2f}", v.width, v.height, v.size,
                        " ".join(x for x in (v.make, v.model) if x),
                        v.dup_group or "", "oui" if v.dup_keep else "non"])


def write_report(out: Path, all_videos: list[Video]) -> None:
    # le rapport ne montre que ce qui va bouger ; la vidéo gardée d'un groupe de
    # copies est montrée à côté, même si elle est déjà rangée
    keepers = {v.dup_group: v for v in all_videos if v.dup_group and v.dup_keep}
    videos = [v for v in all_videos if not v.pinned]
    by_dest = collections.defaultdict(list)
    for v in videos:
        by_dest[v.destination].append(v)
    order = sorted(by_dest, key=lambda d: (not d.startswith("_A_REVOIR"), d))

    def card(v: Video, keep: bool = False) -> str:
        info = " · ".join(x for x in (
            v.date[:16].replace("T", " "), fmt_dur(v.duration),
            f"{v.width}x{v.height}" if v.width else "", fmt_size(v.size),
            " ".join(x for x in (v.make, v.model) if x)) if x)
        cls = " class=keep" if keep else ""
        tag = ("GARDÉE (déjà rangée) · " if v.pinned else "GARDÉE · ") if keep else ""
        return (f'<figure{cls} data-src="{html.escape(Path(v.path).as_uri())}">'
                f'<div class=play>▶</div><figcaption><b>{tag}{html.escape(v.name)}</b>'
                f'<br>{html.escape(info)}<br><i>{html.escape(v.reason)}</i></figcaption></figure>')

    total = sum(v.size for v in videos)
    parts = ["<!doctype html><meta charset=utf-8><title>Tri vidéos</title><style>"
             "body{font-family:system-ui;margin:16px;background:#111;color:#eee}"
             "h2{margin-top:32px;border-bottom:1px solid #444}"
             ".g{display:flex;flex-wrap:wrap;gap:6px}"
             "figure{margin:0;width:220px;font-size:11px;color:#aaa;background:#1b1b1b;"
             "border-radius:6px;padding:6px;box-sizing:border-box}"
             "figcaption b{color:#ddd;word-break:break-all}"
             ".play{height:110px;display:flex;align-items:center;justify-content:center;"
             "font-size:32px;background:#222;border-radius:4px;cursor:pointer;color:#8cf}"
             "video{width:100%;height:110px;background:#000;border-radius:4px}"
             ".keep{outline:3px solid #3c3}table{border-collapse:collapse;font-size:13px}"
             "td{padding:2px 10px;border-bottom:1px solid #333}"
             "nav a{color:#8cf;margin-right:12px;line-height:1.8}</style>"
             f"<h1>Plan de tri des vidéos</h1><p>{len(videos)} vidéos, {fmt_size(total)}. "
             "Clique sur ▶ pour lire une vidéo (rien n'est chargé avant).</p><nav>"]
    for d in order:
        sz = sum(v.size for v in by_dest[d])
        parts.append(f'<a href="#{html.escape(d)}">{html.escape(d)} '
                     f'({len(by_dest[d])}, {fmt_size(sz)})</a>')
    parts.append('<a href="#lourdes">les plus lourdes</a></nav>')
    for d in order:
        items = by_dest[d]
        parts.append(f'<h2 id="{html.escape(d)}">{html.escape(d)} — {len(items)} · '
                     f'{fmt_size(sum(v.size for v in items))}</h2><div class=g>')
        if d.startswith("_A_REVOIR/similaires/") or d == "_A_REVOIR/doublons":
            shown = set()
            for v in items:
                k = keepers.get(v.dup_group)
                if k and k.dup_group not in shown:
                    shown.add(k.dup_group)
                    parts.append(card(k, keep=True))
                parts.append(card(v))
        else:
            for v in items[:400]:
                parts.append(card(v))
            if len(items) > 400:
                parts.append(f"<p>… et {len(items) - 400} autres</p>")
        parts.append("</div>")
    parts.append('<h2 id=lourdes>Les 30 vidéos les plus lourdes</h2><table>')
    for v in sorted(videos, key=lambda v: -v.size)[:30]:
        parts.append(f"<tr><td>{fmt_size(v.size)}</td><td>{fmt_dur(v.duration)}</td>"
                     f"<td>{html.escape(v.name)}</td><td>{html.escape(v.destination)}</td></tr>")
    parts.append("</table><script>document.querySelectorAll('.play').forEach(p=>p.onclick=()=>{"
                 "const v=document.createElement('video');v.controls=true;v.autoplay=true;"
                 "v.src=p.parentElement.dataset.src;p.replaceWith(v);});</script>")
    atomic_write_text(out / REPORT, "\n".join(parts))


def summary(all_videos: list[Video]) -> None:
    videos = [v for v in all_videos if not v.pinned]
    n, sz = collections.Counter(), collections.Counter()
    for v in videos:
        d = v.destination
        k = ("voyages" if d.startswith("Voyages/") else
             "similaires" if d.startswith("_A_REVOIR/similaires") else
             d if d.startswith("_A_REVOIR") else "par mois")
        n[k] += 1
        sz[k] += v.size
    log(f"\nRésumé ({len(videos)} nouvelles vidéos à ranger) :")
    for k in sorted(n):
        log(f"  {k:32s} {n[k]:5d}  {fmt_size(sz[k]):>9s}")
    review = sum(v.size for v in videos if v.destination.startswith("_A_REVOIR"))
    log(f"  espace dans _A_REVOIR : {fmt_size(review)}")
    if len(videos) < len(all_videos):
        log(f"  déjà rangées, laissées en place : {len(all_videos) - len(videos)}")


# ---------------------------------------------------------------------------
# Application / annulation
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse", help="analyse et produit plan_videos.csv + rapport_videos.html")
    add_common_analyse_args(a, "vidéos")
    a.add_argument("--plan-photos",
                   help="dossier de photos analysé par tri_photos.py, ou son plan.csv "
                        f"(par défaut : <source>/{PLAN_DIR}/plan.csv s'il existe)")
    a.add_argument("--trip-margin-hours", type=float, default=6,
                   help="marge autour d'un voyage des photos pour y rattacher une vidéo")
    a.add_argument("--trip-min-videos", type=int, default=3,
                   help="voyage détecté par les seules vidéos : nombre minimum")
    a.add_argument("--min-duration", type=float, default=2,
                   help="en dessous (secondes) : vidéo très courte, à revoir")
    a.add_argument("--live-max-s", type=float, default=4,
                   help="durée max d'une vidéo de Live Photo")
    a.add_argument("--keep", type=int, default=1, help="copies gardées par groupe de similaires")
    a.add_argument("--dup-tolerance", type=float, default=0.15,
                   help="écart de durée max (s) entre deux copies d'une même vidéo")
    a.add_argument("--dup-min-duration", type=float, default=3,
                   help="durée min pour chercher des copies ré-encodées")
    a.add_argument("--dates-locales", action="store_true",
                   help="les dates QuickTime de tes appareils sont en heure locale, pas en UTC "
                        "(certaines caméras / vieux appareils)")
    a.set_defaults(func=analyse)
    PLAN.add_commands(sub)
    args = ap.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
