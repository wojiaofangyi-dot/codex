"""基于 Python 的足球比分预测分析。

功能：
1. 使用历史比赛数据估计球队进攻/防守强度。
2. 基于泊松分布计算双方在未来比赛中的进球概率。
3. 输出最可能比分、胜平负概率与期望进球。

说明：
- 这是一个可解释、轻量级的传统统计方法示例。
- 适合做基础分析与快速原型，不依赖第三方库。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class MatchResult:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


@dataclass
class TeamStrength:
    attack: float
    defense: float


def poisson_pmf(k: int, lam: float) -> float:
    """计算泊松分布 P(X=k)。"""
    if k < 0:
        return 0.0
    return exp(-lam) * (lam**k) / factorial(k)


def fit_team_strengths(matches: Iterable[MatchResult]) -> Tuple[Dict[str, TeamStrength], float, float]:
    """从历史比赛数据估计球队强度。

    返回：
    - 每个球队的进攻/防守强度（相对联赛平均值）
    - 联赛场均主队进球
    - 联赛场均客队进球
    """
    matches = list(matches)
    if not matches:
        raise ValueError("比赛数据不能为空")

    teams = sorted({m.home_team for m in matches} | {m.away_team for m in matches})

    home_goals_total = sum(m.home_goals for m in matches)
    away_goals_total = sum(m.away_goals for m in matches)
    n_matches = len(matches)

    avg_home_goals = home_goals_total / n_matches
    avg_away_goals = away_goals_total / n_matches

    # 统计每支球队主场/客场的进球和失球
    home_scored = {t: 0 for t in teams}
    home_conceded = {t: 0 for t in teams}
    away_scored = {t: 0 for t in teams}
    away_conceded = {t: 0 for t in teams}
    home_games = {t: 0 for t in teams}
    away_games = {t: 0 for t in teams}

    for m in matches:
        home_scored[m.home_team] += m.home_goals
        home_conceded[m.home_team] += m.away_goals
        away_scored[m.away_team] += m.away_goals
        away_conceded[m.away_team] += m.home_goals
        home_games[m.home_team] += 1
        away_games[m.away_team] += 1

    strengths: Dict[str, TeamStrength] = {}
    for team in teams:
        if home_games[team] == 0 or away_games[team] == 0:
            raise ValueError(f"球队 {team} 主客场数据不足，无法估计强度")

        team_games = home_games[team] + away_games[team]
        team_scored = home_scored[team] + away_scored[team]
        team_conceded = home_conceded[team] + away_conceded[team]

        league_avg_goals = avg_home_goals + avg_away_goals

        # 进攻强度：球队场均进球 / 联赛场均每队进球
        attack = (team_scored / team_games) / league_avg_goals

        # 防守强度：球队场均失球 / 联赛场均每队失球（越小越强）
        defense = (team_conceded / team_games) / league_avg_goals

        strengths[team] = TeamStrength(attack=attack, defense=defense)

    return strengths, avg_home_goals, avg_away_goals


def predict_score_matrix(
    home_team: str,
    away_team: str,
    strengths: Dict[str, TeamStrength],
    avg_home_goals: float,
    avg_away_goals: float,
    max_goals: int = 6,
) -> List[List[float]]:
    """构建比分概率矩阵，matrix[i][j] 代表主队 i: j 客队概率。"""
    if home_team not in strengths or away_team not in strengths:
        raise KeyError("预测球队不在训练数据中")

    home_lambda = avg_home_goals * strengths[home_team].attack * strengths[away_team].defense
    away_lambda = avg_away_goals * strengths[away_team].attack * strengths[home_team].defense

    matrix: List[List[float]] = []
    for i in range(max_goals + 1):
        row = []
        for j in range(max_goals + 1):
            p = poisson_pmf(i, home_lambda) * poisson_pmf(j, away_lambda)
            row.append(p)
        matrix.append(row)

    # 截断到 max_goals 会丢失尾部概率，这里重归一化保证总和为 1。
    total_prob = sum(sum(row) for row in matrix)
    if total_prob > 0:
        matrix = [[p / total_prob for p in row] for row in matrix]

    return matrix


def summarize_prediction(matrix: List[List[float]]) -> dict:
    """根据比分矩阵汇总胜平负概率与最可能比分。"""
    home_win = draw = away_win = 0.0
    best_score = (0, 0)
    best_prob = -1.0

    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p

            if p > best_prob:
                best_prob = p
                best_score = (i, j)

    exp_home = sum(i * sum(row) for i, row in enumerate(matrix))
    exp_away = sum(j * sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0])))

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "best_score": best_score,
        "best_score_prob": best_prob,
        "expected_goals": (exp_home, exp_away),
    }


def demo() -> None:
    """示例：使用少量虚拟历史数据预测 A vs B。"""
    history = [
        MatchResult("A", "B", 2, 1),
        MatchResult("B", "A", 1, 1),
        MatchResult("A", "C", 3, 0),
        MatchResult("C", "A", 1, 2),
        MatchResult("B", "C", 0, 0),
        MatchResult("C", "B", 2, 1),
        MatchResult("A", "D", 1, 0),
        MatchResult("D", "A", 1, 1),
        MatchResult("B", "D", 2, 2),
        MatchResult("D", "B", 0, 1),
        MatchResult("C", "D", 1, 1),
        MatchResult("D", "C", 0, 2),
    ]

    strengths, avg_home, avg_away = fit_team_strengths(history)
    matrix = predict_score_matrix("A", "B", strengths, avg_home, avg_away, max_goals=6)
    report = summarize_prediction(matrix)

    print("=== A vs B 比分预测 ===")
    print(f"主胜概率: {report['home_win']:.2%}")
    print(f"平局概率: {report['draw']:.2%}")
    print(f"客胜概率: {report['away_win']:.2%}")
    print(f"最可能比分: {report['best_score'][0]}:{report['best_score'][1]} ({report['best_score_prob']:.2%})")
    print(f"期望进球: A {report['expected_goals'][0]:.2f} - B {report['expected_goals'][1]:.2f}")


if __name__ == "__main__":
    demo()
