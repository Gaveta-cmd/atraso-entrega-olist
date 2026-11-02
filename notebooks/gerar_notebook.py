"""Gera e executa atraso_entrega.ipynb a partir de atraso_entrega.py (células separadas por "# %%").

Uso: OLIST_DIR=<pasta com os CSVs> python notebooks/gerar_notebook.py
"""
import re
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PASTA = Path(__file__).resolve().parent
FONTE = PASTA / "atraso_entrega.py"
SAIDA = PASTA / "atraso_entrega.ipynb"


def montar() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for bloco in re.split(r"(?m)^(?=# %%)", FONTE.read_text(encoding="utf-8")):
        if not bloco.strip():
            continue
        cabecalho, _, corpo = bloco.partition("\n")
        if "[markdown]" in cabecalho:
            texto = "\n".join(re.sub(r"^# ?", "", linha) for linha in corpo.strip().splitlines())
            nb.cells.append(nbf.v4.new_markdown_cell(texto))
        else:
            nb.cells.append(nbf.v4.new_code_cell(corpo.strip()))
    return nb


def main() -> None:
    nb = montar()
    NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(PASTA)}}).execute()
    nbf.write(nb, SAIDA)
    print(f"{SAIDA.name}: {len(nb.cells)} células, {SAIDA.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
