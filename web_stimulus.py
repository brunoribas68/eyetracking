"""Estímulo a partir de um site: captura a página e resolve AOIs por seletor CSS.

Hoje (modo "print"): o site é aberto num Chromium sem janela (Playwright), na
mesma resolução da tela do teste, e vira uma imagem parada (estimulo.png) que o
participante vê. Junto vai um arquivo lateral estimulo.pagina.json com a URL, a
rolagem, o tamanho da página e as caixas das AOIs encontradas por seletor.

Os dois sistemas de coordenadas são mantidos separados desde já:

- **página** (documento inteiro, origem no topo da página, em px CSS): onde as
  AOIs ficam guardadas; é estável enquanto o layout não muda;
- **vista** (o que está na tela naquele instante = pixels do estímulo): onde o
  olhar é medido. ``PageView`` converte de um para o outro usando a rolagem.

No modo "print" existe uma única vista com rolagem (0, 0), então página e vista
coincidem na parte visível. Num futuro modo "ao vivo" (participante navegando de
verdade) basta registrar uma sequência de ``PageView`` ao longo do tempo (URL +
rolagem por frame), converter cada fixação com ``PageView.view_to_page`` e
agrupar a análise por URL usando uma captura de página inteira
(``full_page=True``) como fundo do heatmap. Nada aqui assume que a vista é fixa.

Formato do aois.json, misturando os dois tipos:
    [{"name": "busca",     "selector": "button[aria-label='Buscar']"},
     {"name": "destaque",  "x": 150, "y": 300, "w": 400, "h": 180}]
AOIs com x/y/w/h são em px do estímulo e passam direto.

Uso:
    python web_stimulus.py capture --study estudos/f1tv          # lê site.json
    python web_stimulus.py capture https://exemplo.com --out estimulo.png --aois aois.json
    python web_stimulus.py login --study estudos/f1tv            # salva login (sites fechados)

Requer:  pip install playwright   (e um Chromium: `playwright install chromium`)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_VIEWPORT = (1920, 1080)
SIDECAR_SUFFIX = ".pagina.json"
STORAGE_STATE_NAME = "navegador.json"


# ---------------------------------------------------------------------------
# coordenadas
# ---------------------------------------------------------------------------
@dataclass
class PageView:
    """O que estava na tela: qual página, rolada até onde, em que tamanho."""
    url: str
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    viewport_w: int = DEFAULT_VIEWPORT[0]
    viewport_h: int = DEFAULT_VIEWPORT[1]
    page_w: int = 0
    page_h: int = 0
    t_ms: float = 0.0  # instante da vista na sessão (usado no modo ao vivo)

    def view_to_page(self, x: float, y: float) -> tuple[float, float]:
        return x + self.scroll_x, y + self.scroll_y

    def page_to_view(self, x: float, y: float) -> tuple[float, float]:
        return x - self.scroll_x, y - self.scroll_y

    def box_to_view(self, box: dict) -> Optional[dict]:
        """Caixa em coordenadas da página -> vista, recortada; None se fora da tela."""
        x0, y0 = self.page_to_view(box["x"], box["y"])
        x1, y1 = x0 + box["w"], y0 + box["h"]
        cx0, cy0 = max(0.0, x0), max(0.0, y0)
        cx1, cy1 = min(float(self.viewport_w), x1), min(float(self.viewport_h), y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return None
        return {"x": cx0, "y": cy0, "w": cx1 - cx0, "h": cy1 - cy0}


@dataclass
class PageSnapshot:
    """Uma captura de página: a vista usada e as AOIs resolvidas (coord. da página)."""
    view: PageView
    screenshot: str = ""
    full_page: bool = False
    captured_at: str = ""
    aois: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PageSnapshot":
        d = dict(d)
        d["view"] = PageView(**d["view"])
        return cls(**d)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PageSnapshot":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def sidecar_path(stimulus: Path) -> Path:
    return stimulus.with_name(stimulus.stem + SIDECAR_SUFFIX)


def load_snapshot_for(stimulus: Optional[Path]) -> Optional[PageSnapshot]:
    if stimulus is None:
        return None
    side = sidecar_path(Path(stimulus))
    return PageSnapshot.load(side) if side.exists() else None


def aois_in_view(aois: list[dict], snapshot: Optional[PageSnapshot]) -> list[dict]:
    """Converte a lista do aois.json em caixas na vista (px do estímulo).

    - AOIs com x/y/w/h passam direto;
    - AOIs com "selector" usam a caixa achada na captura (``snapshot.aois``),
      convertida da página para a vista e recortada ao que estava visível.
    AOIs não encontradas ou fora da tela são avisadas e descartadas.
    """
    resolved = {a["name"]: a for a in (snapshot.aois if snapshot else [])}
    out = []
    for a in aois:
        if all(k in a for k in ("x", "y", "w", "h")):
            out.append(a)
            continue
        if "selector" not in a:
            raise ValueError(f"AOI {a.get('name')!r} precisa de x/y/w/h ou de selector")
        r = resolved.get(a["name"])
        if snapshot is None or r is None:
            print(f"[aviso] AOI {a['name']!r} usa seletor, mas não há captura de página "
                  f"({SIDECAR_SUFFIX}) com ela; recapture o estímulo", file=sys.stderr)
            continue
        if r.get("missing"):
            print(f"[aviso] AOI {a['name']!r}: seletor {a['selector']!r} não encontrado na página",
                  file=sys.stderr)
            continue
        box = snapshot.view.box_to_view(r)
        if box is None:
            print(f"[aviso] AOI {a['name']!r} fica fora da área visível", file=sys.stderr)
            continue
        out.append({**{k: v for k, v in a.items() if k != "selector"}, **box, "selector": a["selector"]})
    return out


# ---------------------------------------------------------------------------
# navegador
# ---------------------------------------------------------------------------
def _chromium_executable() -> Optional[str]:
    """Usa o Chromium já instalado se o do Playwright não bater com a versão."""
    exe = os.environ.get("EYETRACKING_CHROMIUM")
    if exe:
        return exe
    for cand in ("/opt/pw-browsers/chromium",):
        if Path(cand).is_file():
            return cand
    return None


@contextmanager
def open_browser(headless: bool = True) -> Iterator[object]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise SystemExit("Para usar um site como estímulo: pip install playwright "
                         "&& playwright install chromium") from exc
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=headless)
        except Exception:
            exe = _chromium_executable()
            if not exe:
                raise
            browser = pw.chromium.launch(headless=headless, executable_path=exe)
        try:
            yield browser
        finally:
            browser.close()


_BOXES_JS = """
(aois) => aois.map(a => {
  const el = document.querySelector(a.selector);
  if (!el) return {name: a.name, selector: a.selector, missing: true};
  const r = el.getBoundingClientRect();
  return {name: a.name, selector: a.selector,
          x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height};
})
"""

_PAGE_JS = """
() => ({url: location.href, scroll_x: window.scrollX, scroll_y: window.scrollY,
        page_w: document.documentElement.scrollWidth, page_h: document.documentElement.scrollHeight})
"""


def resolve_aois(page, aois: list[dict]) -> list[dict]:
    """Caixas (coord. da página) de cada AOI com seletor, na página já carregada."""
    sel = [{"name": a["name"], "selector": a["selector"]} for a in aois if "selector" in a]
    return page.evaluate(_BOXES_JS, sel) if sel else []


def current_view(page, viewport: tuple[int, int], t_ms: float = 0.0) -> PageView:
    info = page.evaluate(_PAGE_JS)
    return PageView(viewport_w=viewport[0], viewport_h=viewport[1], t_ms=t_ms, **info)


def capture(url: str, out_png: Path, viewport: tuple[int, int] = DEFAULT_VIEWPORT,
            aois: Optional[list[dict]] = None, wait_ms: int = 2000, full_page: bool = False,
            scroll: tuple[int, int] = (0, 0), click: tuple[str, ...] = (), hide: tuple[str, ...] = (),
            storage_state: Optional[Path] = None) -> PageSnapshot:
    """Abre ``url``, prepara a página e salva o print + o arquivo lateral."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with open_browser() as browser:
        ctx_kwargs: dict = {"viewport": {"width": viewport[0], "height": viewport[1]}, "device_scale_factor": 1}
        if storage_state and Path(storage_state).exists():
            ctx_kwargs["storage_state"] = str(storage_state)
        context = browser.new_context(**ctx_kwargs)
        page = context.new_page()
        page.goto(url, wait_until="load")
        for sel in click:  # ex.: botão "aceitar cookies"
            try:
                page.click(sel, timeout=3000)
            except Exception:
                print(f"[aviso] não consegui clicar em {sel!r}", file=sys.stderr)
        if hide:
            page.add_style_tag(content=",".join(hide) + "{display:none !important}")
        if scroll != (0, 0):
            page.evaluate("([x, y]) => window.scrollTo(x, y)", list(scroll))
        page.wait_for_timeout(wait_ms)
        view = current_view(page, viewport)
        boxes = resolve_aois(page, aois or [])
        page.screenshot(path=str(out_png), full_page=full_page)
        context.close()
    if full_page:
        view.scroll_x = view.scroll_y = 0.0
        view.viewport_w, view.viewport_h = view.page_w, view.page_h
    snap = PageSnapshot(view=view, screenshot=out_png.name, full_page=full_page,
                        captured_at=time.strftime("%Y-%m-%dT%H:%M:%S"), aois=boxes)
    snap.save(sidecar_path(out_png))
    return snap


def login(url: str, storage_state: Path, viewport: tuple[int, int] = DEFAULT_VIEWPORT) -> None:
    """Abre um navegador visível para o pesquisador logar e salva cookies/sessão."""
    with open_browser(headless=False) as browser:
        context = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})
        page = context.new_page()
        page.goto(url)
        input("Faça o login no navegador e aperte ENTER aqui para salvar... ")
        storage_state.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(storage_state))
        context.close()
    print(f"Sessão do navegador salva em {storage_state} (contém cookies: não versione)")


# ---------------------------------------------------------------------------
# configuração do estudo (site.json)
# ---------------------------------------------------------------------------
def parse_size(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in text.lower().split("x"))
    return w, h


def load_site_config(study: Path) -> dict:
    """Lê estudos/<nome>/site.json. Campos: url (obrigatório), viewport "1920x1080",
    wait_ms, full_page, scroll [x, y], click [seletores], hide [seletores]."""
    path = study / "site.json"
    if not path.exists():
        raise SystemExit(f"{path} não existe; crie com pelo menos {{\"url\": \"https://...\"}}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if "url" not in cfg:
        raise SystemExit(f"{path} precisa do campo \"url\"")
    return cfg


def capture_study(study: Path, viewport: Optional[tuple[int, int]] = None,
                  out_png: Optional[Path] = None) -> PageSnapshot:
    """Captura o estímulo de um estudo a partir do site.json + aois.json.

    ``viewport`` (se dado) tem prioridade sobre o site.json; o ideal é que seja
    o tamanho da tela do teste, para o print não ser redimensionado.
    """
    cfg = load_site_config(study)
    aois_path = study / "aois.json"
    aois = json.loads(aois_path.read_text(encoding="utf-8")) if aois_path.exists() else []
    vp = viewport or (parse_size(cfg["viewport"]) if cfg.get("viewport") else DEFAULT_VIEWPORT)
    return capture(cfg["url"], out_png or study / "estimulo.png", vp, aois,
                   wait_ms=int(cfg.get("wait_ms", 2000)), full_page=bool(cfg.get("full_page", False)),
                   scroll=tuple(cfg.get("scroll", (0, 0))), click=tuple(cfg.get("click", ())),
                   hide=tuple(cfg.get("hide", ())), storage_state=study / STORAGE_STATE_NAME)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Usa um site como estímulo de eye tracking")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="tira o print do site e resolve as AOIs por seletor")
    c.add_argument("url", nargs="?", help="URL (ou use --study com site.json)")
    c.add_argument("--study", type=Path)
    c.add_argument("--out", type=Path, help="png de saída (padrão: <estudo>/estimulo.png)")
    c.add_argument("--aois", type=Path)
    c.add_argument("--viewport", help="LARGURAxALTURA (use o tamanho da tela do teste)")
    c.add_argument("--wait-ms", type=int, default=2000)
    c.add_argument("--full-page", action="store_true")
    lg = sub.add_parser("login", help="abre o navegador para logar e salva a sessão")
    lg.add_argument("url", nargs="?")
    lg.add_argument("--study", type=Path)
    lg.add_argument("--save", type=Path, help=f"arquivo da sessão (padrão: <estudo>/{STORAGE_STATE_NAME})")
    args = p.parse_args(argv)
    vp = parse_size(args.viewport) if getattr(args, "viewport", None) else None

    if args.cmd == "login":
        url = args.url or (load_site_config(args.study)["url"] if args.study else None)
        save = args.save or (args.study / STORAGE_STATE_NAME if args.study else None)
        if not url or not save:
            p.error("informe a URL e --save, ou --study")
        login(url, save, vp or DEFAULT_VIEWPORT)
        return

    if args.study and not args.url:
        snap = capture_study(args.study, vp, args.out)
        out = args.out or args.study / "estimulo.png"
    else:
        if not args.url or not args.out:
            p.error("informe a URL e --out, ou --study")
        aois = json.loads(args.aois.read_text(encoding="utf-8")) if args.aois else []
        snap = capture(args.url, args.out, vp or DEFAULT_VIEWPORT, aois, args.wait_ms, args.full_page)
        out = args.out
    found = sum(1 for a in snap.aois if not a.get("missing"))
    print(f"Estímulo salvo em {out} ({snap.view.viewport_w}x{snap.view.viewport_h}) | "
          f"AOIs por seletor: {found}/{len(snap.aois)} encontradas")
    for a in snap.aois:
        if a.get("missing"):
            print(f"  - não encontrada: {a['name']} ({a['selector']})")


if __name__ == "__main__":
    main()
