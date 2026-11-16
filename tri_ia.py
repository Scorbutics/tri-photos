"""
Partie « intelligence artificielle » commune à tri_photos.py et tri_videos.py :
le modèle CLIP (empreinte d'une image) et la reconnaissance des groupes nommés
dans la page de découverte (categories.json).

Dépendances : numpy, Pillow, torch, open_clip (voir LISEZMOI.md). tri_videos.py
n'importe ce module que s'il a des groupes à reconnaître.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from tri_commun import DEFAULT_MODEL, DEFAULT_PRETRAINED, log, safe_name


class Clip:
    """Modèle CLIP local : une image ou une phrase -> un vecteur (« empreinte »).
    Deux images qui se ressemblent ont des empreintes proches."""

    def __init__(self, model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED):
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


class Groups:
    """Groupes nommés dans la page de découverte (categories.json) dont le choix est
    « À revoir » ou « Dossier thème ». Les groupes « Tri normal » n'ont pas d'effet."""

    def __init__(self, path: Path):
        self.path = path
        self.cats = [c for c in json.loads(Path(path).read_text(encoding="utf-8"))
                     if c.get("action") in ("revoir", "theme")]
        self.centres = (np.stack([np.array(c["centre"]) for c in self.cats])
                        if self.cats else np.zeros((0, 1)))
        self.seuils = np.array([c["seuil"] for c in self.cats])

    def __len__(self) -> int:
        return len(self.cats)

    def match(self, emb: np.ndarray | None) -> tuple[str, str] | None:
        """(action, nom de dossier) du groupe le plus ressemblant, s'il est assez proche
        (au-delà du seuil d'appartenance calculé à la découverte), sinon None."""
        if emb is None or not self.cats:
            return None
        sims = self.centres @ emb
        i = int(np.argmax(sims))
        if sims[i] < self.seuils[i]:
            return None
        return self.cats[i]["action"], safe_name(self.cats[i]["nom"]) or "groupe"
