from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from sota_pipeline import (
    REPORTS,
    ROOT,
)

APP_SRC = ROOT.parents[1] / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from arena_ai.worldcup_model import MatchAnalysis, WorldCupModel, effective_monte_carlo_workers  # noqa: E402


MODEL_PATH = ROOT / "models" / "model_sota.pkl"
REPORT_PATH = REPORTS / "sota_model_report.json"
RAW_HISTORY_PATH = ROOT / "data" / "raw" / "candidates" / "pataterie_all_matches.csv"


console = Console()


MENU_OPTIONS = [
    ("1", "Metricas", "Compara os modelos em accuracy, top-2, log loss, empate e Brier.", "Validacao"),
    ("2", "Odds de campeao", "Snapshot salvo do ultimo build Monte Carlo.", "Torneio"),
    ("3", "Odds por fase", "Mostra chance de uma selecao chegar a cada fase da Copa.", "Torneio"),
    ("4", "Prever confronto", "Mostra classificador, Poisson/Dixon-Coles, mix e placar sorteado.", "Partida"),
    ("5", "Simular uma Copa", "Roda uma Copa completa com seed controlada e sorteio hibrido.", "Monte Carlo"),
    ("6", "Artefatos", "Lista pickle, relatorios e CSVs usados pela aplicacao.", "Arquivos"),
    ("7", "Forca das selecoes", "Ranking de elenco, mercado, caps, lesoes e cobertura por selecao.", "Elenco"),
    ("8", "Jogos de grupos", "Mostra probabilidades e xG dos 72 jogos da fase de grupos.", "Partida"),
    ("9", "Backtest por Copa", "Audita 1994-2022 por Copa, modelo e metricas probabilisticas.", "Validacao"),
    ("10", "Explicar confronto", "Mostra os principais sinais que empurram uma previsao.", "Explicacao"),
    ("11", "Caminho de selecao", "Cruza odds por fase com o caminho da seed 2026.", "Torneio"),
    ("12", "Estabilidade seeds", "Roda varias seeds e mostra campeoes/cenarios de sensibilidade.", "Monte Carlo"),
    ("13", "Monte Carlo ao vivo", "Roda fresh por padrao ou bootstrap turbo, com cenario plausivel.", "Monte Carlo"),
    ("0", "Sair", "Fecha a console.", "Sistema"),
]


def load_report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def show_header() -> None:
    console.print(
        Panel(
            "[bold cyan]Arena AI - Console SOTA da Copa 2026[/bold cyan]\n"
            "[white]Classificador 1X2/XGBoost + Poisson/DC para placar + sorteio estatistico influenciado + Monte Carlo paralelo[/white]",
            border_style="cyan",
            box=box.ROUNDED,
            expand=False,
        )
    )


def show_menu(report: dict[str, object], model: WorldCupModel) -> None:
    table = Table(
        title="[bold]Menu principal[/bold]",
        caption="Escolha pelo numero. Use Enter para aceitar o padrao quando houver.",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Opcao", justify="center", style="bold yellow", no_wrap=True)
    table.add_column("Tela", style="bold white", no_wrap=True)
    table.add_column("Para que serve", style="white")
    table.add_column("Tipo", justify="center", style="green", no_wrap=True)
    for option, title, description, category in MENU_OPTIONS:
        style = "dim" if option == "0" else None
        table.add_row(option, title, description, category, style=style)
    console.print(table)

    raw_rows, raw_start, raw_end = raw_history_summary()
    team_count = len(model.team_names())
    policy = model.simulation_policy()
    classifier_weight = float(policy["classifier_weight"])
    poisson_weight = float(policy["poisson_weight"])
    draw_floor = float(policy["draw_floor"])
    draw_ceiling = float(policy["draw_ceiling"])
    default_workers = effective_monte_carlo_workers()
    summary = (
        f"[bold]Historico bruto/ELO:[/bold] {raw_rows} jogos "
        f"({raw_start} a {raw_end})\n"
        f"[bold]Treino supervisionado:[/bold] {report.get('training_rows', 'n/a')} jogos "
        f"({report.get('train_start', 'n/a')} a {report.get('train_end', 'n/a')})\n"
        f"[bold]Selecoes 2026:[/bold] {team_count} times vindos dos jogos de grupos do fixture 2026\n"
        "[bold]Lista FIFA final:[/bold] validada sem placeholders; usa nomes oficiais como USA, IR Iran, Korea Republic e Congo DR\n"
        f"[bold]Sorteio:[/bold] {classifier_weight:.0%} Classificador 1X2/XGBoost + {poisson_weight:.0%} Poisson/Dixon-Coles\n"
        f"[bold]Empate:[/bold] calibrado por faixa {draw_floor:.0%}-{draw_ceiling:.0%}; sem draw_xgb ativo\n"
        f"[bold]Monte Carlo snapshot:[/bold] {report.get('monte_carlo_runs', 'n/a')} simulacoes "
        f"| [bold]Ao vivo:[/bold] ate {default_workers} workers "
        f"| [bold]Pickle:[/bold] model_sota.pkl"
    )
    console.print(Panel(summary, title="Resumo do pacote carregado", border_style="green", box=box.ROUNDED))


def show_context(title: str, body: str, style: str = "blue") -> None:
    console.print(Panel(body, title=f"[bold]{title}[/bold]", border_style=style, box=box.ROUNDED))


def show_metrics(report: dict[str, object]) -> None:
    show_context(
        "Metricas",
        "Accuracy mede acerto da classe mais provavel. Top-2 aceita a resposta real entre as duas maiores probabilidades.\n"
        "Log loss, Brier e ECE avaliam qualidade probabilistica; menor costuma ser melhor.",
    )
    table = Table(title="Comparativo dos modelos", box=box.ROUNDED, border_style="blue", header_style="bold blue")
    table.add_column("Modelo", style="bold")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Top-2", justify="right", style="green")
    table.add_column("Log loss", justify="right", style="yellow")
    table.add_column("Recall de empate", justify="right", style="magenta")
    table.add_column("Brier", justify="right", style="yellow")
    table.add_column("Teste", justify="right", style="cyan")
    for name, metrics in report["metrics"].items():
        table.add_row(
            name,
            fmt(metrics.get("accuracy")),
            fmt(metrics.get("top2_accuracy")),
            fmt(metrics.get("log_loss")),
            fmt(metrics.get("draw_recall")),
            fmt(metrics.get("brier")),
            str(metrics.get("test_rows", "")),
        )
    console.print(table)


def show_champion_odds() -> None:
    show_context(
        "Odds de campeao - snapshot",
        "Estas probabilidades vem do ultimo build salvo em CSV. Para rodar uma amostra nova agora, use a opcao 'Monte Carlo ao vivo'.",
        "green",
    )
    rows = read_csv_dicts(REPORTS / "sota_champion_odds.csv")
    table = Table(title="Top chances de campeao", box=box.ROUNDED, border_style="green", header_style="bold green")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Selecao", style="bold")
    table.add_column("Titulos simulados", justify="right", style="cyan")
    table.add_column("Prob. ± IC", justify="right")
    runs = max(1, sum(int(row.get("wins", 0)) for row in rows))
    for i, row in enumerate(rows[:16], start=1):
        probability = float(row["champion_probability"])
        table.add_row(str(i), row["team"], row["wins"], color_pct_with_error(probability, runs))
    console.print(table)


def show_live_monte_carlo(model: WorldCupModel) -> None:
    default_workers = effective_monte_carlo_workers()
    show_context(
        "Monte Carlo ao vivo",
        "Roda a mesma experiencia do jogo por padrao: 1000 Copas fresh, sem banco de cenarios. Use bootstrap apenas como modo turbo explicito.",
        "gold1",
    )
    runs = IntPrompt.ask("Simulacoes", default=1000)
    seed = IntPrompt.ask("Seed base", default=2026)
    mode = Prompt.ask("Modo", choices=["fresh", "bootstrap"], default="fresh")
    requested_workers = IntPrompt.ask("Workers", default=default_workers)
    runs = max(1000, int(runs))
    workers = effective_monte_carlo_workers(requested_workers)
    use_scenario_bank = mode == "bootstrap"
    if workers != requested_workers:
        console.print(f"[yellow]Workers ajustados para {workers}; este pipeline limita o Monte Carlo a {default_workers} threads.[/yellow]")
    with Progress(
        TextColumn("[bold gold1]{task.description}"),
        BarColumn(bar_width=42),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_label = "Monte Carlo bootstrap" if use_scenario_bank else f"Monte Carlo fresh ({workers} threads)"
        task_id = progress.add_task(task_label, total=runs)

        def on_progress(done: int, total: int, snapshot: object) -> bool:
            progress.update(task_id, completed=done)
            return True

        champion_odds, story = model.champion_odds_with_representative(
            runs=runs,
            seed=seed,
            workers=workers,
            progress_callback=on_progress,
            progress_with_odds=False,
            use_scenario_bank=use_scenario_bank,
        )
    console.print(
        Panel.fit(
            f"[bold green]Amostra concluida:[/bold green] {runs} simulacoes | modo {mode} | seed base {seed} | {workers} workers",
            border_style="green",
        )
    )
    highlight = str(story.get("representative_for", story.get("champion", ""))) if story else None
    show_champion_table(champion_odds, title="Top campeoes - Monte Carlo ao vivo", highlight_team=highlight)
    if story:
        story["runs"] = runs
        show_representative_story(story)


def show_representative_story(story: dict[str, object]) -> None:
    team = str(story.get("representative_for", story.get("champion", "-")))
    leader = str(story.get("odds_leader", "-"))
    rank = int(story.get("representative_rank", 0))
    probability = float(story.get("representative_probability", 0.0))
    leader_probability = float(story.get("odds_leader_probability", 0.0))
    runs = max(1, int(story.get("runs", 1000) or 1000))
    summary = (
        f"[bold cyan]Campanha em destaque:[/bold cyan] {team} "
        f"(top {rank}, {pct_with_error(probability, runs)})\n"
        f"[bold]Favorito da amostra:[/bold] {leader} lidera com {pct_with_error(leader_probability, runs)}\n"
        f"[bold]Finalista calibrado:[/bold] {story.get('representative_runner_up', '-')} "
        f"(finalista #{int(story.get('representative_runner_up_finalist_rank', 0))}, "
        f"{float(story.get('representative_runner_up_finalist_probability', 0.0)):.1%} das finais)\n"
        f"[bold]Pontuação narrativa:[/bold] {float(story.get('representative_plausibility_score', 0.0)):.0%} | "
        f"{story.get('representative_surprise_level', 'plausivel')}\n"
        "A chave abaixo nao substitui o ranking: ela e uma Copa concreta gerada pelo Monte Carlo, escolhida por plausibilidade narrativa."
    )
    console.print(Panel(summary, title="Campanha em destaque - top 5 + plausibilidade", border_style="cyan", box=box.ROUNDED))
    standings = pd.DataFrame(story["standings"])
    bracket = pd.DataFrame(story["rounds"])
    show_group_stage(standings, bracket)
    show_team_knockout_path(team, bracket, title=f"Caminho do destaque - {team}")


def show_team_knockout_path(team: str, bracket: object, title: str) -> None:
    path = [row for row in bracket.itertuples(index=False) if row.home == team or row.away == team]
    table = Table(title=title, box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("Fase", style="bold")
    table.add_column("Confronto")
    table.add_column("Placar", justify="center")
    table.add_column("Vencedor", style="green")
    table.add_column("Resolucao", style="cyan")
    for row in path:
        table.add_row(
            pretty_stage(row.round),
            f"{row.home} x {row.away}",
            f"{int(row.home_goals)} x {int(row.away_goals)}",
            str(row.winner),
            str(row.resolution),
        )
    console.print(table)


def show_champion_table(champion_odds: object, title: str = "Top chances de campeao", highlight_team: str | None = None) -> None:
    caption = "Linha ciano = Campanha em destaque; o ranking continua ordenado pelas odds da amostra." if highlight_team else None
    table = Table(title=title, caption=caption, box=box.ROUNDED, border_style="green", header_style="bold green")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Selecao", style="bold")
    table.add_column("Titulos simulados", justify="right", style="cyan")
    table.add_column("Prob. ± IC", justify="right")
    if hasattr(champion_odds, "head"):
        rows = list(champion_odds.head(16).itertuples(index=False))
    else:
        rows = [
            type("ChampionRow", (), {"team": team, "wins": wins, "champion_probability": probability})
            for team, wins, probability in list(champion_odds)[:16]
        ]
    if hasattr(champion_odds, "columns") and "wins" in champion_odds.columns:
        runs = max(1, int(champion_odds["wins"].sum()))
    else:
        runs = max(1, sum(int(row.wins) for row in rows))
    for i, row in enumerate(rows, start=1):
        team = str(row.team)
        style = "bold cyan" if highlight_team == team else None
        table.add_row(str(i), team, str(int(row.wins)), color_pct_with_error(float(row.champion_probability), runs), style=style)
    console.print(table)


def show_stage_odds() -> None:
    show_context(
        "Odds por fase",
        "Mostra a probabilidade acumulada de a selecao chegar em cada etapa. A chance de ser campea e sempre menor ou igual a chance de chegar na final.",
        "magenta",
    )
    team = Prompt.ask("Time", default="Brazil")
    rows = [row for row in read_csv_dicts(REPORTS / "sota_stage_odds.csv") if row["team"].lower() == team.lower()]
    if not rows:
        console.print(f"[red]Nao encontrei {team}.[/red]")
        return
    table = Table(title=f"Odds por fase - {rows[0]['team']}", box=box.ROUNDED, border_style="magenta", header_style="bold magenta")
    table.add_column("Fase", style="bold")
    table.add_column("Prob. ± IC", justify="right")
    runs = int(load_report().get("monte_carlo_runs", 1000))
    for row in rows:
        table.add_row(row["stage"], color_pct_with_error(float(row["probability"]), runs))
    console.print(table)


def choose_team(teams: list[str], label: str, default_team: str | None = None, groups: dict[str, str] | None = None) -> str:
    table = Table(
        title=f"[bold]{label}[/bold]",
        caption="Digite o numero da selecao. A numeracao e fixa para facilitar comparacao.",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", style="yellow", no_wrap=True)
    table.add_column("G", justify="center", style="cyan", no_wrap=True)
    table.add_column("Selecao")
    table.add_column("#", justify="right", style="yellow", no_wrap=True)
    table.add_column("G", justify="center", style="cyan", no_wrap=True)
    table.add_column("Selecao")
    table.add_column("#", justify="right", style="yellow", no_wrap=True)
    table.add_column("G", justify="center", style="cyan", no_wrap=True)
    table.add_column("Selecao")
    rows = [teams[index : index + 3] for index in range(0, len(teams), 3)]
    for row_index, row in enumerate(rows):
        cells = []
        for column_index in range(3):
            team_index = row_index * 3 + column_index
            if column_index < len(row):
                team = row[column_index]
                group = groups.get(team) if groups else None
                cells.extend([str(team_index + 1), group or "", team])
            else:
                cells.extend(["", "", ""])
        table.add_row(*cells)
    console.print(table)

    default_index = None
    if default_team in teams:
        default_index = teams.index(str(default_team)) + 1
    while True:
        choice = IntPrompt.ask("Numero", default=default_index)
        if 1 <= choice <= len(teams):
            return teams[choice - 1]
        console.print(f"[red]Escolha um numero entre 1 e {len(teams)}.[/red]")


def predict_game(model: WorldCupModel) -> None:
    show_context(
        "Prever confronto",
        "Escolha duas selecoes do fixture 2026. A tela mostra a sinergia: o Classificador 1X2/XGBoost decide a tendencia, Poisson/Dixon-Coles distribui placares, e o sorteio hibrido gera um placar possivel.",
        "yellow",
    )
    fixture_rows = model.fixture_teams()
    teams = [team for team, _group in fixture_rows]
    groups = dict(fixture_rows)
    home = choose_team(teams, "Selecione o Time A", default_team="Brazil", groups=groups)
    console.clear()
    away_default = "France" if home != "France" else "Brazil"
    while True:
        away = choose_team(teams, f"Selecione o Time B - adversario de {home}", default_team=away_default, groups=groups)
        if away != home:
            break
        console.print("[red]Escolha uma selecao diferente para o Time B.[/red]")
    console.clear()
    score_seed = IntPrompt.ask("Seed do sorteio do placar", default=2026)
    analysis = model.analyze_match(home, away, seed=score_seed)
    table = Table(title=f"{home} x {away}", box=box.ROUNDED, border_style="yellow", header_style="bold yellow")
    table.add_column("Leitura")
    table.add_column("Valor", justify="right")
    table.add_column("O que significa")
    table.add_row("XGBoost 1X2 base", format_probability_tuple(home, away, analysis.base_classifier_probs), "Sinal bruto de vitoria/empate/derrota em 90min")
    table.add_row("Modelo 1X2 final calibrado", format_probability_tuple(home, away, analysis.final_classifier_probs), f"Politica de empate calibrada: {analysis.draw_policy_text}")
    table.add_row("Massa de placares Poisson/DC", format_probability_tuple(home, away, analysis.poisson_outcome_probs), "Soma da matriz de placares por pacote 1X2")
    table.add_row(
        "Mix do sorteio",
        format_probability_tuple(home, away, analysis.blend_probs),
        f"{analysis.classifier_weight:.0%} classificador + {analysis.poisson_weight:.0%} Poisson",
    )
    table.add_row(
        "Pacote sorteado",
        f"[bold gold1]{outcome_label(analysis.outcome_class, home, away)}[/bold gold1]",
        f"Chance do pacote no mix: {analysis.outcome_probability:.1%}",
    )
    table.add_row(
        "Placar sorteado",
        f"[bold cyan]{analysis.sampled_home} x {analysis.sampled_away}[/bold cyan]",
        f"Seed {score_seed}; prob. dentro do pacote: {analysis.score_probability:.1%}",
    )
    best_score = analysis.top_scores[0]
    table.add_row("Placar mais provavel bruto", f"[cyan]{best_score[0]} x {best_score[1]}[/cyan]", f"Poisson/DC sem filtro do pacote: {best_score[2]:.1%}")
    table.add_row("Top placares", format_score_list(list(analysis.top_scores[:5])), "Distribuicao de placares mais provaveis")
    table.add_row("Mais de 2,5 gols", color_pct(analysis.over_25), "Probabilidade de 3 ou mais gols")
    table.add_row("Menos de 2,5 gols", color_pct(analysis.under_25), "Probabilidade de 0, 1 ou 2 gols")
    table.add_row("Ambos marcam", color_pct(analysis.btts), "Probabilidade de os dois times fazerem gol")
    table.add_row(f"{home} avanca/vencedor", color_pct(analysis.home_advances), "Obrigatorio em mata-mata")
    table.add_row(f"{away} avanca/vencedor", color_pct(analysis.away_advances), "Inclui extra time e penaltis")
    table.add_row(f"xG {home}", f"[cyan]{analysis.home_xg:.2f}[/cyan]", "Gols esperados")
    table.add_row(f"xG {away}", f"[cyan]{analysis.away_xg:.2f}[/cyan]", "Gols esperados")
    table.add_row("Diferenca top26 elenco", signed_number(analysis.drivers.squad_top26_diff), "Forca relativa do elenco")
    console.print(table)


def simulate_once(model: WorldCupModel) -> None:
    show_context(
        "Simular uma Copa",
        "A seed controla a aleatoriedade. Cada jogo usa o sorteio estatistico influenciado; mesma seed gera o mesmo torneio, outra seed explora outro universo possivel.",
        "gold1",
    )
    seed = IntPrompt.ask("Seed", default=2026)
    result = model.simulate_tournament(seed)
    champion = str(result["champion"])
    bracket = pd.DataFrame(result["rounds"])
    standings = pd.DataFrame(result["standings"])
    console.print(Panel.fit(f"[bold gold1]Campeao simulado: {champion}[/bold gold1]", border_style="gold1"))
    show_group_stage(standings, bracket)
    table = Table(title="Mata-mata", box=box.ROUNDED, border_style="gold1", header_style="bold gold1")
    table.add_column("Fase", style="bold")
    table.add_column("Confronto")
    table.add_column("Placar", justify="center")
    table.add_column("Vencedor", style="green")
    table.add_column("Resolucao", style="cyan")
    table.add_column("Sorteio 90min", style="yellow")
    for row in bracket.itertuples(index=False):
        sim_outcome = getattr(row, "sim_outcome", None)
        if sim_outcome is None:
            simulation = "-"
        else:
            probability = float(getattr(row, "sim_outcome_probability", 0.0))
            simulation = f"{outcome_label(int(sim_outcome), row.home, row.away)} ({probability:.1%})"
        table.add_row(
            pretty_stage(row.round),
            f"{row.home} x {row.away}",
            f"{row.home_goals} x {row.away_goals}",
            row.winner,
            row.resolution,
            simulation,
        )
    console.print(table)


def show_group_stage(standings: object, bracket: object) -> None:
    table = Table(
        title="Fase de grupos - classificacao simulada",
        caption="Top 2 de cada grupo avancam; os 8 melhores terceiros tambem entram no Round of 32.",
        box=box.ROUNDED,
        border_style="green",
        header_style="bold green",
    )
    table.add_column("G", justify="center", style="cyan", no_wrap=True)
    table.add_column("#", justify="right", style="yellow", no_wrap=True)
    table.add_column("Selecao", style="bold")
    table.add_column("Pts", justify="right", style="green")
    table.add_column("V", justify="right")
    table.add_column("GP", justify="right")
    table.add_column("GC", justify="right")
    table.add_column("SG", justify="right")
    table.add_column("Status", style="cyan")

    round32 = bracket[bracket["round"] == "Round of 32"]
    third_qualifiers = set(round32["home"]).union(set(round32["away"])) - set(standings[standings["rank"] <= 2]["team"])
    for row in standings.sort_values(["group", "rank"]).itertuples(index=False):
        rank = int(row.rank)
        if rank <= 2:
            status = "Avanca"
            status_style = "bold green"
        elif row.team in third_qualifiers:
            status = "Melhor 3o"
            status_style = "bold yellow"
        else:
            status = "Eliminado"
            status_style = "dim"
        table.add_row(
            str(row.group),
            str(rank),
            str(row.team),
            str(int(row.pts)),
            str(int(row.wins)),
            str(int(row.gf)),
            str(int(row.ga)),
            signed_int(int(row.gd)),
            f"[{status_style}]{status}[/{status_style}]",
        )
    console.print(table)


def show_artifacts(report: dict[str, object]) -> None:
    show_context("Artefatos", "Arquivos gerados pelo pipeline SOTA e consumidos pela console ou pelo jogo.", "cyan")
    table = Table(title="Artefatos", box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("Nome", style="bold")
    table.add_column("Arquivo")
    for name, path in report["artifacts"].items():
        table.add_row(name, str(path))
    for name, path in {
        "statistical_report": REPORTS / "sota_statistical_report.json",
        "statistical_report_md": ROOT.parents[1] / "docs" / "STATISTICAL_AUDIT.md",
        "calibration_bins": REPORTS / "sota_calibration_bins.csv",
        "class_calibration_summary": REPORTS / "sota_class_calibration_summary.csv",
        "block_bootstrap_intervals": REPORTS / "sota_block_bootstrap_intervals.csv",
        "ablation_study": REPORTS / "sota_ablation_study.csv",
        "dixon_coles_rho_sensitivity": REPORTS / "sota_dixon_coles_rho_sensitivity.csv",
        "runtime_adjustment_audit": REPORTS / "sota_runtime_adjustment_audit.csv",
        "monte_carlo_stability": REPORTS / "sota_monte_carlo_stability.json",
        "monte_carlo_stage_bracket_stability": REPORTS / "sota_monte_carlo_stage_bracket_stability.csv",
        "raw_data_manifest": REPORTS / "sota_raw_data_manifest.json",
        "raw_data_manifest_csv": REPORTS / "sota_raw_data_manifest.csv",
    }.items():
        if path.exists():
            table.add_row(name, str(path))
    console.print(table)


def show_team_strength(model: WorldCupModel) -> None:
    show_context(
        "Forca das selecoes",
        "Ranking 2026 baseado em FC26/Transfermarkt. Ele nao e convocacao oficial; e uma camada de forca de elenco para alimentar o modelo.",
        "cyan",
    )
    table = Table(title="Top forca de elenco", box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("G", justify="center", style="cyan")
    table.add_column("Selecao", style="bold")
    table.add_column("Top26", justify="right", style="green")
    table.add_column("Setores A/M/D/G", justify="center")
    table.add_column("TM valor", justify="right", style="yellow")
    table.add_column("Caps", justify="right")
    table.add_column("Lesao dias", justify="right", style="red")
    for index, row in enumerate(model.team_strength_rows(limit=20), start=1):
        sectors = (
            f"{float(row['attack_strength']):.0f}/"
            f"{float(row['midfield_strength']):.0f}/"
            f"{float(row['defense_strength']):.0f}/"
            f"{float(row['gk_strength']):.0f}"
        )
        table.add_row(
            str(index),
            str(row["group_letter"]),
            str(row["team_key"]),
            f"{float(row['squad_top26']):.1f}",
            sectors,
            money(float(row["tm_market_value_top25"])),
            f"{float(row['tm_caps_top25']):.0f}",
            f"{float(row['tm_recent_injury_days_top25']):.0f}",
        )
    console.print(table)


def show_group_matches() -> None:
    show_context(
        "Jogos de grupos",
        "Probabilidades de 90min dos 72 jogos de grupos. A ultima coluna mostra o resultado mais provavel, que tambem pode ser empate.",
        "green",
    )
    rows = read_csv_dicts(REPORTS / "sota_match_probabilities.csv")
    table = Table(title="Fase de grupos - probabilidades por jogo", box=box.ROUNDED, border_style="green", header_style="bold green")
    table.add_column("#", justify="right", style="dim")
    table.add_column("G", justify="center", style="cyan")
    table.add_column("Jogo", style="bold")
    table.add_column("Casa", justify="right")
    table.add_column("Empate", justify="right")
    table.add_column("Fora", justify="right")
    table.add_column("xG", justify="center")
    table.add_column("Resultado mais provavel 90min")
    for row in rows:
        probs = {
            row["home_team"]: float(row["p_home_win_90"]),
            "Empate": float(row["p_draw_90"]),
            row["away_team"]: float(row["p_away_win_90"]),
        }
        favorite, favorite_prob = max(probs.items(), key=lambda item: item[1])
        table.add_row(
            row["match_number"],
            row["group"],
            f"{row['home_team']} x {row['away_team']}",
            color_pct(row["p_home_win_90"]),
            color_pct(row["p_draw_90"]),
            color_pct(row["p_away_win_90"]),
            f"{float(row['home_xg']):.2f} x {float(row['away_xg']):.2f}",
            f"{favorite} ({favorite_prob:.1%})",
        )
    console.print(table)


def show_backtest() -> None:
    show_context(
        "Backtest por Copa",
        "Walk-forward: para cada Copa, o modelo treina apenas com jogos anteriores e testa naquela Copa. Log loss, Brier, RPS e ECE avaliam probabilidades.",
        "blue",
    )
    rows = read_csv_dicts(REPORTS / "sota_world_cup_backtest_folds.csv")
    preferred = {"xgb_1x2", "xgb_temperature_calibrated_1x2", "poisson_goal_model_1x2", "baseline_elo_1x2"}
    table = Table(title="Backtest 1994-2022", box=box.ROUNDED, border_style="blue", header_style="bold blue")
    table.add_column("Copa", justify="right")
    table.add_column("Modelo", style="bold")
    table.add_column("Acc", justify="right", style="green")
    table.add_column("Top2", justify="right")
    table.add_column("Log loss", justify="right", style="yellow")
    table.add_column("Brier", justify="right")
    table.add_column("RPS", justify="right")
    table.add_column("ECE", justify="right")
    for row in rows:
        if row["model"] not in preferred:
            continue
        table.add_row(
            row["cup_year"],
            pretty_model(row["model"]),
            pct(row["accuracy"]),
            pct(row["top2_accuracy"]),
            f"{float(row['log_loss']):.4f}",
            f"{float(row['brier']):.4f}",
            f"{float(row['rps']):.4f}",
            f"{float(row['ece']):.4f}",
        )
    console.print(table)


def explain_match(model: WorldCupModel) -> None:
    show_context(
        "Explicar confronto",
        "Esta tela nao substitui SHAP; ela traduz os principais deltas e mostra como o confronto vira sorteio estatistico influenciado.",
        "yellow",
    )
    home, away, analysis = choose_match_analysis(model)
    table = Table(title=f"Explicacao - {home} x {away}", box=box.ROUNDED, border_style="yellow", header_style="bold yellow")
    table.add_column("Sinal")
    table.add_column("Valor", justify="right")
    table.add_column("Leitura")
    drivers = [
        ("XGBoost 1X2 base", format_probability_tuple(home, away, analysis.base_classifier_probs), "Tendencia bruta do resultado em 90min"),
        ("Modelo 1X2 final calibrado", format_probability_tuple(home, away, analysis.final_classifier_probs), f"Politica de empate calibrada: {analysis.draw_policy_text}"),
        ("Massa de placares Poisson/DC", format_probability_tuple(home, away, analysis.poisson_outcome_probs), "Massa acumulada dos placares por pacote"),
        ("Mix sorteio", format_probability_tuple(home, away, analysis.blend_probs), "Distribuicao usada para sortear pacote 1X2"),
        ("Top placares", format_score_list(list(analysis.top_scores[:5])), "Placares brutos mais provaveis na matriz"),
        ("Avanca no desempate", f"{home} {analysis.home_advances:.1%} | {away} {analysis.away_advances:.1%}", "Quem avanca se houver prorrogacao/penaltis"),
        ("xG", f"{analysis.home_xg:.2f} x {analysis.away_xg:.2f}", "Placar esperado usado para simular gols"),
    ]
    for driver in analysis.drivers.rows():
        drivers.append((driver.label, signed_number(driver.value), driver.description))
    for name, value, reading in drivers:
        table.add_row(name, str(value), reading)
    console.print(table)


def show_team_path(model: WorldCupModel) -> None:
    show_context(
        "Caminho de selecao",
        "Mostra odds por fase e, na seed 2026, quais partidas a selecao jogou ate ser eliminada ou campea.",
        "magenta",
    )
    fixture_rows = model.fixture_teams()
    teams = [team for team, _group in fixture_rows]
    groups = dict(fixture_rows)
    team = choose_team(teams, "Selecione a selecao", default_team="Brazil", groups=groups)
    console.clear()
    rows = [row for row in read_csv_dicts(REPORTS / "sota_stage_odds.csv") if row["team"] == team]
    odds = Table(title=f"Odds por fase - {team}", box=box.ROUNDED, border_style="magenta", header_style="bold magenta")
    odds.add_column("Fase")
    odds.add_column("Prob. ± IC", justify="right")
    runs = int(load_report().get("monte_carlo_runs", 1000))
    for row in rows:
        odds.add_row(row["stage"], color_pct_with_error(float(row["probability"]), runs))
    console.print(odds)
    result = model.simulate_tournament(2026)
    bracket = pd.DataFrame(result["rounds"])
    path = [row for row in bracket.itertuples(index=False) if row.home == team or row.away == team]
    path_table = Table(title="Caminho na seed 2026", box=box.ROUNDED, border_style="magenta", header_style="bold magenta")
    path_table.add_column("Fase")
    path_table.add_column("Jogo")
    path_table.add_column("Placar", justify="center")
    path_table.add_column("Resultado")
    if not path:
        path_table.add_row("Grupos", "-", "-", "Nao passou da fase de grupos")
    else:
        for row in path:
            result = "Venceu" if row.winner == team else "Eliminado"
            path_table.add_row(pretty_stage(row.round), f"{row.home} x {row.away}", f"{row.home_goals} x {row.away_goals}", result)
    console.print(path_table)


def show_seed_stability(model: WorldCupModel) -> None:
    show_context(
        "Estabilidade por seeds",
        f"Cada linha e uma Copa completa com uma seed fixa. Isto nao muda as odds salvas do relatorio ({int(load_report().get('monte_carlo_runs', 1000))} runs); serve para inspecionar cenarios reproduziveis.",
        "gold1",
    )
    seeds = [2026, 42, 7, 123, 777, 1001, 10000, 314159]
    table = Table(title="Cenarios reproduziveis", box=box.ROUNDED, border_style="gold1", header_style="bold gold1")
    table.add_column("Seed", justify="right")
    table.add_column("Campeao", style="bold green")
    table.add_column("Final")
    table.add_column("Placar", justify="center")
    table.add_column("Resolucao")
    for seed in seeds:
        result = model.simulate_tournament(seed)
        champion = str(result["champion"])
        bracket = pd.DataFrame(result["rounds"])
        final = bracket[bracket["round"] == "Final"].iloc[0]
        table.add_row(str(seed), champion, f"{final.home} x {final.away}", f"{int(final.home_goals)} x {int(final.away_goals)}", str(final.resolution))
    console.print(table)


def choose_match_analysis(model: WorldCupModel) -> tuple[str, str, MatchAnalysis]:
    fixture_rows = model.fixture_teams()
    teams = [team for team, _group in fixture_rows]
    groups = dict(fixture_rows)
    home = choose_team(teams, "Selecione o Time A", default_team="Brazil", groups=groups)
    console.clear()
    away_default = "France" if home != "France" else "Brazil"
    while True:
        away = choose_team(teams, f"Selecione o Time B - adversario de {home}", default_team=away_default, groups=groups)
        if away != home:
            break
        console.print("[red]Escolha uma selecao diferente para o Time B.[/red]")
    console.clear()
    return home, away, model.analyze_match(home, away, seed=2026)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def raw_history_summary() -> tuple[int | str, str, str]:
    import csv

    if not RAW_HISTORY_PATH.exists():
        return "n/a", "n/a", "n/a"
    rows = 0
    min_date = ""
    max_date = ""
    with RAW_HISTORY_PATH.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            date = row.get("date", "")
            if not date:
                continue
            rows += 1
            min_date = date if not min_date or date < min_date else min_date
            max_date = date if not max_date or date > max_date else max_date
    return rows, min_date, max_date


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_score_list(scores: list[tuple[int, int, float]]) -> str:
    return " | ".join(f"{home}x{away} {probability:.1%}" for home, away, probability in scores)


def format_probability_tuple(home: str, away: str, probs: tuple[float, float, float]) -> str:
    return (
        f"{home} {float(probs[0]):.1%} | "
        f"Empate {float(probs[1]):.1%} | "
        f"{away} {float(probs[2]):.1%}"
    )


def outcome_label(outcome: int, home: str, away: str) -> str:
    if outcome == 0:
        return f"vitoria {home}"
    if outcome == 1:
        return "empate"
    return f"vitoria {away}"


def pct(value: object) -> str:
    if value in (None, "", "nan"):
        return ""
    return f"{float(value):.1%}"


def money(value: float) -> str:
    if value >= 1_000_000_000:
        return f"EUR {value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"EUR {value / 1_000_000:.0f}M"
    if value >= 1_000:
        return f"EUR {value / 1_000:.0f}K"
    return f"EUR {value:.0f}"


def color_pct(value: object) -> str:
    probability = float(value)
    text = f"{probability:.1%}"
    if probability >= 0.65:
        return f"[bold green]{text}[/bold green]"
    if probability >= 0.35:
        return f"[yellow]{text}[/yellow]"
    return f"[red]{text}[/red]"


def monte_carlo_error(probability: float, runs: int) -> float:
    p = max(0.0, min(1.0, float(probability)))
    return 1.96 * math.sqrt(max(0.0, p * (1.0 - p)) / max(1, int(runs)))


def pct_with_error(probability: float, runs: int) -> str:
    return f"{float(probability):.1%} ± {monte_carlo_error(probability, runs):.1%}"


def color_pct_with_error(probability: float, runs: int) -> str:
    text = pct_with_error(probability, runs)
    if probability >= 0.65:
        return f"[bold green]{text}[/bold green]"
    if probability >= 0.35:
        return f"[yellow]{text}[/yellow]"
    return f"[red]{text}[/red]"


def signed_number(value: object) -> str:
    number = float(value)
    style = "green" if number >= 0 else "red"
    return f"[{style}]{number:+.2f}[/{style}]"


def signed_int(value: int) -> str:
    style = "green" if value > 0 else "red" if value < 0 else "white"
    return f"[{style}]{value:+d}[/{style}]"


def pretty_stage(stage: object) -> str:
    stage_map = {
        "Quarterfinals": "Quarter-finals",
        "Semifinals": "Semi-finals",
    }
    return stage_map.get(str(stage), str(stage))


def pretty_model(model: object) -> str:
    model_map = {
        "baseline_elo_1x2": "ELO baseline",
        "xgb_1x2": "XGB 1X2",
        "xgb_temperature_calibrated_1x2": "XGB calibrado",
        "poisson_goal_model_1x2": "Poisson gols",
    }
    return model_map.get(str(model), str(model))


def main() -> None:
    model = WorldCupModel()
    report = model.report if model.report else load_report()
    choices = [option for option, *_rest in MENU_OPTIONS]
    while True:
        show_header()
        show_menu(report, model)
        choice = Prompt.ask("Escolha", choices=choices, default="1")
        console.clear()
        if choice == "0":
            return
        if choice == "1":
            show_metrics(report)
        elif choice == "2":
            show_champion_odds()
        elif choice == "3":
            show_stage_odds()
        elif choice == "4":
            predict_game(model)
        elif choice == "5":
            simulate_once(model)
        elif choice == "6":
            show_artifacts(report)
        elif choice == "7":
            show_team_strength(model)
        elif choice == "8":
            show_group_matches()
        elif choice == "9":
            show_backtest()
        elif choice == "10":
            explain_match(model)
        elif choice == "11":
            show_team_path(model)
        elif choice == "12":
            show_seed_stability(model)
        elif choice == "13":
            show_live_monte_carlo(model)
        Prompt.ask("\nEnter para voltar", default="")
        console.clear()


if __name__ == "__main__":
    main()
