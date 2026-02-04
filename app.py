import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import textwrap

from datetime import datetime, timedelta
from plotly.subplots import make_subplots


@st.cache_data
def gerar_dados_mock(ano: int | None = None):
    """
    Gera dados mock em nível transacional (para Performance Comercial) e um df diário agregado
    (para Análise Operacional), cobrindo o ano base e o ano anterior.

    Retorna:
    - df_vendas: transações
    - df_diario: agregado diário
    - df_metas: metas mensais por vendedor/equipe
    """

    hoje = datetime.today().date()
    ano_base = ano if ano is not None else (hoje.year - 1)  # último ano completo

    vendedores = [
        "Ana", "Bruno", "Carla", "Diego", "Elisa", "Felipe",
        "Gi", "Henrique", "Isabela", "João", "Karina", "Lucas",
    ]
    equipes = {
        "Ana": "Equipe A",
        "Bruno": "Equipe A",
        "Carla": "Equipe A",
        "Diego": "Equipe B",
        "Elisa": "Equipe B",
        "Felipe": "Equipe B",
        "Gi": "Equipe C",
        "Henrique": "Equipe C",
        "Isabela": "Equipe C",
        "João": "Equipe D",
        "Karina": "Equipe D",
        "Lucas": "Equipe D",
    }
    regioes = {
        "Equipe A": "Sudeste",
        "Equipe B": "Sul",
        "Equipe C": "Nordeste",
        "Equipe D": "Centro-Oeste",
    }

    def gerar_transacoes_ano(ano_ref: int, seed: int) -> pd.DataFrame:
        datas = pd.date_range(start=f"{ano_ref}-01-01", end=f"{ano_ref}-12-31", freq="D")
        rng = np.random.default_rng(seed)

        n_dias = len(datas)
        dow = datas.dayofweek.to_numpy()
        doy = datas.dayofyear.to_numpy()

        fator_dow = np.select([dow == 5, dow == 6], [0.78, 0.72], default=1.0)
        sazonal = 1.0 + 0.18 * np.sin(2 * np.pi * (doy / 365.0))
        pico_maio = np.exp(-0.5 * ((doy - 140) / 18) ** 2) * 0.18
        pico_nov = np.exp(-0.5 * ((doy - 325) / 22) ** 2) * 0.35
        trend = 0.95 + 0.10 * (np.arange(n_dias) / (n_dias - 1))

        lam = 85 * fator_dow * sazonal * (1 + pico_maio + pico_nov) * trend
        lam = np.clip(lam, 10, None)
        qtd_transacoes_dia = rng.poisson(lam).astype(int)

        ticket_base = 65 * sazonal * (1 + 0.15 * pico_nov) * trend
        sigma = 0.55
        mu = np.log(np.maximum(ticket_base, 1)) - 0.5 * sigma**2

        datas_rep = np.repeat(datas.to_numpy(), qtd_transacoes_dia)
        if len(datas_rep) == 0:
            return pd.DataFrame()

        # vendedor por transação (leve diferença de performance)
        perf_vendedor = np.array(
            [1.10, 0.95, 1.00, 0.98, 1.05, 1.02, 0.97, 1.03, 1.01, 0.99, 1.04, 0.96]
        )
        prob_vendedor = perf_vendedor / perf_vendedor.sum()
        vendedor = rng.choice(vendedores, size=len(datas_rep), p=prob_vendedor)
        equipe = np.array([equipes[v] for v in vendedor])
        regiao = np.array([regioes[e] for e in equipe])

        # cliente por transação
        n_clientes = 2400
        cliente_id = rng.integers(1, n_clientes + 1, size=len(datas_rep))

        # ticket por transação (dependente do dia)
        # gera por blocos diários (mais eficiente)
        ticket = np.empty(len(datas_rep), dtype=float)
        idx = 0
        for i, n in enumerate(qtd_transacoes_dia):
            if n <= 0:
                continue
            ticket[idx : idx + n] = rng.lognormal(mean=mu[i], sigma=sigma, size=n)
            idx += n
        # variação por vendedor/região
        fator_regiao = np.where(regiao == "Sudeste", 1.05, 1.0)
        fator_regiao = np.where(regiao == "Nordeste", 0.95, fator_regiao)
        ticket = ticket * fator_regiao

        # itens
        itens_por_tx = rng.normal(loc=2.0, scale=0.4, size=len(datas_rep)).clip(1.0, 4.0)
        qtd_itens = np.maximum(np.round(itens_por_tx), 1).astype(int)

        valor_bruto = (ticket).round(2)

        # descontos (promo em nov + campanhas)
        mes = pd.to_datetime(datas_rep).month
        promo = (mes == 11).astype(float)
        evento = (rng.random(len(datas_rep)) < 0.04).astype(float)
        taxa_desconto = (0.05 + 0.10 * promo + 0.05 * evento + rng.normal(0, 0.01, len(datas_rep))).clip(0.0, 0.30)
        desconto = (valor_bruto * taxa_desconto).round(2)

        imposto = (valor_bruto * 0.12).round(2)

        # devoluções/cancelamentos (em parte das transações)
        p_devol = (0.010 + 0.010 * promo + 0.004 * (valor_bruto > np.quantile(valor_bruto, 0.9))).clip(0.005, 0.05)
        is_devol = rng.random(len(datas_rep)) < p_devol
        devolucao = np.where(is_devol, (valor_bruto * rng.uniform(0.3, 1.0, size=len(datas_rep))), 0.0).round(2)

        valor_liquido = (valor_bruto - desconto - imposto - devolucao).round(2)

        # custo (COGS) para margem bruta (varia por região/produto)
        base_cogs = rng.normal(loc=0.62, scale=0.06, size=len(datas_rep)).clip(0.45, 0.85)
        base_cogs = np.where(regiao == "Sudeste", base_cogs - 0.02, base_cogs)
        base_cogs = np.where(regiao == "Nordeste", base_cogs + 0.02, base_cogs)
        custo = (valor_liquido * base_cogs).round(2)
        lucro_bruto = (valor_liquido - custo).round(2)

        # recebimentos (PMR/DSO): prazo base por região + atrasos
        prazo_base = np.select(
            [regiao == "Sudeste", regiao == "Sul", regiao == "Nordeste", regiao == "Centro-Oeste"],
            [25, 28, 35, 32],
            default=30,
        )
        atraso = rng.normal(loc=3, scale=6, size=len(datas_rep)).round().astype(int)
        atraso = np.clip(atraso, -5, 40)
        dias_receb = np.clip(prazo_base + atraso, 0, 90).astype(int)

        dt_venda = pd.to_datetime(datas_rep)
        dt_receb = dt_venda + pd.to_timedelta(dias_receb, unit="D")

        # algumas faturas em aberto (sem recebimento)
        aberto = rng.random(len(datas_rep)) < 0.06
        dt_receb = dt_receb.where(~aberto, pd.NaT)

        df = pd.DataFrame(
            {
                "id_venda": np.arange(1, len(datas_rep) + 1),
                "data": dt_venda,
                "cliente_id": cliente_id,
                "vendedor": vendedor,
                "equipe": equipe,
                "regiao": regiao,
                "qtd_itens": qtd_itens,
                "valor_bruto": valor_bruto,
                "desconto": desconto,
                "imposto": imposto,
                "devolucao": devolucao,
                "valor_liquido": valor_liquido,
                "custo": custo,
                "lucro_bruto": lucro_bruto,
                "data_recebimento": dt_receb,
            }
        )
        df["ano"] = df["data"].dt.year
        df["mes"] = df["data"].dt.to_period("M").dt.to_timestamp()
        df["semana"] = df["data"].dt.to_period("W").dt.start_time
        df["trimestre"] = df["data"].dt.to_period("Q").dt.start_time
        return df

    df_prev = gerar_transacoes_ano(ano_base - 1, seed=41)
    df_curr = gerar_transacoes_ano(ano_base, seed=42)
    df_vendas = pd.concat([df_prev, df_curr], ignore_index=True)

    # agregado diário (mantém o dashboard operacional atual)
    # OBS: precisamos nomear a coluna do groupby explicitamente para evitar KeyError ('data')
    if df_vendas.empty:
        df_diario = pd.DataFrame(
            columns=[
                "data",
                "faturamento_bruto",
                "descontos",
                "impostos",
                "devolucoes_valor",
                "faturamento_liquido",
                "qtd_transacoes",
                "qtd_itens",
                "ano",
                "mes",
                "semana",
                "trimestre",
            ]
        )
    else:
        df_diario = (
            df_vendas.assign(data_dia=df_vendas["data"].dt.floor("D"))
            .groupby("data_dia", as_index=False)
            .agg(
                faturamento_bruto=("valor_bruto", "sum"),
                descontos=("desconto", "sum"),
                impostos=("imposto", "sum"),
                devolucoes_valor=("devolucao", "sum"),
                faturamento_liquido=("valor_liquido", "sum"),
                qtd_transacoes=("id_venda", "count"),
                qtd_itens=("qtd_itens", "sum"),
            )
            .rename(columns={"data_dia": "data"})
        )
        df_diario["data"] = pd.to_datetime(df_diario["data"])
    df_diario["ano"] = df_diario["data"].dt.year
    df_diario["mes"] = df_diario["data"].dt.to_period("M").dt.to_timestamp()
    df_diario["semana"] = df_diario["data"].dt.to_period("W").dt.start_time
    df_diario["trimestre"] = df_diario["data"].dt.to_period("Q").dt.start_time

    # metas mensais (mock) por vendedor: baseado em histórico (ano anterior) + crescimento
    hist = (
        df_vendas[df_vendas["ano"] == ano_base - 1]
        .assign(mes_num=lambda d: d["data"].dt.month)
        .groupby(["vendedor", "equipe", "mes_num"], as_index=False)
        .agg(faturamento_liquido=("valor_liquido", "sum"))
    )

    rng_meta = np.random.default_rng(123)
    hist["meta"] = (hist["faturamento_liquido"] * rng_meta.normal(1.08, 0.06, size=len(hist))).clip(0).round(2)
    df_metas = hist[["vendedor", "equipe", "mes_num", "meta"]].copy()
    df_metas["ano"] = ano_base
    df_metas = df_metas.rename(columns={"meta": "meta_faturamento_liquido"})

    return df_vendas, df_diario, df_metas


def format_currency(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_currency_compact(v: float) -> str:
    v = float(v or 0)
    abs_v = abs(v)
    if abs_v >= 1_000_000_000:
        return f"R$ {v/1_000_000_000:.2f} bi".replace(".", ",")
    if abs_v >= 1_000_000:
        return f"R$ {v/1_000_000:.2f} mi".replace(".", ",")
    if abs_v >= 1_000:
        return f"R$ {v/1_000:.2f} mil".replace(".", ",")
    return format_currency(v)


def format_percent(v):
    return f"{v:+.1f}%"


def kpi_card(title, value, sublabel=None, delta=None, help_text=None):
    with st.container(border=True):
        if delta is not None:
            st.metric(title, value, format_percent(delta))
        else:
            st.metric(title, value)
        if sublabel:
            st.caption(sublabel)
        if help_text:
            st.caption(help_text)

def inject_typography():
    # Carrega Nunito via Google Fonts (mais confiável que @import em alguns ambientes)
    st.markdown(
        textwrap.dedent(
            """
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700&display=swap" rel="stylesheet">
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        textwrap.dedent(
            """
            <style>
            /* Fonte padrão para o app (exclui ícones do Streamlit) */
            html, body, [data-testid="stAppViewContainer"] {
              font-family: 'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
            }
            [data-testid="stAppViewContainer"] *:not(.material-icons):not([class*="material-symbols"]):not([data-testid="stIconMaterial"]):not([data-testid="stIconMaterial"] *) {
              font-family: 'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
            }

            /* Mantém Material Icons/Symbols funcionando */
            .material-icons,
            [class*="material-icons"] {
              font-family: 'Material Icons' !important;
              font-feature-settings: 'liga' 1;
              -webkit-font-feature-settings: 'liga' 1;
            }
            [class*="material-symbols"] {
              font-family: 'Material Symbols Rounded' !important;
              font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
              font-feature-settings: 'liga' 1;
              -webkit-font-feature-settings: 'liga' 1;
            }

            /* Streamlit usa stIconMaterial com ligatures (texto -> ícone) */
            [data-testid="stIconMaterial"],
            [data-testid="stIconMaterial"] * {
              font-family: 'Material Symbols Rounded' !important;
              font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
              font-feature-settings: 'liga' 1;
              -webkit-font-feature-settings: 'liga' 1;
              font-variant-ligatures: contextual common-ligatures;
              text-transform: none;
              letter-spacing: normal;
              white-space: nowrap;
              word-wrap: normal;
              direction: ltr;
              display: inline-block;
              line-height: 1;
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def inject_dashboard_style():
    # Importante: o Streamlit reexecuta o script a cada interação.
    # Se o CSS for injetado só uma vez, o layout pode "quebrar" em reruns.
    st.markdown(
        textwrap.dedent(
            """
            <style>
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 16px;
                margin-bottom: 8px;
                width: 100%;
            }
            @media (max-width: 1200px) {
                .kpi-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 720px) {
                .kpi-grid {
                    grid-template-columns: 1fr;
                }
            }
            .kpi-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 16px 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
                min-height: 140px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                position: relative;
                overflow: hidden;
            }
            .kpi-card::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                height: 4px;
                width: 100%;
                background: var(--kpi-accent, #2563eb);
            }
            .kpi-top {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 12px;
            }
            .kpi-icon {
                width: 40px;
                height: 40px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: color-mix(in srgb, var(--kpi-accent, #2563eb) 14%, #ffffff);
                border: 1px solid color-mix(in srgb, var(--kpi-accent, #2563eb) 22%, #e2e8f0);
                font-size: 1.1rem;
                line-height: 1;
                flex: 0 0 auto;
            }
            .kpi-title {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #64748b;
                font-weight: 600;
            }
            .kpi-value {
                font-size: 1.45rem;
                font-weight: 700;
                color: #0f172a;
                margin-top: 2px;
            }
            .kpi-sub {
                font-size: 0.78rem;
                color: #94a3b8;
                margin-top: 4px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .kpi-delta-pos {
                font-size: 0.8rem;
                color: #16a34a;
                font-weight: 600;
            }
            .kpi-delta-neg {
                font-size: 0.8rem;
                color: #dc2626;
                font-weight: 600;
            }
            .kpi-help {
                font-size: 0.72rem;
                color: #94a3b8;
                margin-top: 6px;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def render_kpi_grid(kpis):
    inject_dashboard_style()
    cards_html = []
    for item in kpis:
        title = item.get("title", "")
        value = item.get("value", "")
        sublabel = item.get("sublabel")
        delta = item.get("delta")
        help_text = item.get("help_text")
        icon = item.get("icon", "•")
        accent = item.get("accent", "#2563eb")

        delta_html = ""
        if delta is not None:
            css_class = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
            delta_html = f'<div class="{css_class}">{format_percent(delta)}</div>'

        sub_html = (
            f'<div class="kpi-sub" title="{sublabel}">{sublabel}</div>'
            if sublabel
            else ""
        )
        help_html = (
            f'<div class="kpi-help" title="{help_text}">{help_text}</div>'
            if help_text
            else ""
        )

        card_html = (
            f'<div class="kpi-card" style="--kpi-accent:{accent}">'
            f'<div class="kpi-top">'
            f'<div>'
            f'<div class="kpi-title">{title}</div>'
            f'<div class="kpi-value">{value}</div>'
            f"{sub_html}"
            f"</div>"
            f'<div class="kpi-icon" aria-hidden="true">{icon}</div>'
            f"</div>"
            f"<div>"
            f"{delta_html}"
            f"{help_html}"
            f"</div>"
            f"</div>"
        )
        cards_html.append(card_html)

    grid_html = '<div class="kpi-grid">' + "".join(cards_html) + "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)


def calcular_periodos(df: pd.DataFrame, periodo: str):
    """Retorna df agregado pelo período selecionado."""
    if periodo == "Diário":
        group_col = "data"
    elif periodo == "Semanal":
        group_col = "semana"
    elif periodo == "Mensal":
        group_col = "mes"
    elif periodo == "Trimestral":
        group_col = "trimestre"
    elif periodo == "Anual":
        group_col = "ano"
    else:
        group_col = "data"

    agg = (
        df.groupby(group_col)
        .agg(
            faturamento_bruto=("faturamento_bruto", "sum"),
            faturamento_liquido=("faturamento_liquido", "sum"),
            devolucoes_valor=("devolucoes_valor", "sum"),
            qtd_transacoes=("qtd_transacoes", "sum"),
            qtd_itens=("qtd_itens", "sum"),
        )
        .reset_index()
        .rename(columns={group_col: "periodo"})
    )

    # Normaliza "periodo" para datetime quando for anual (para facilitar gráficos)
    if periodo == "Anual":
        agg["periodo"] = pd.to_datetime(agg["periodo"].astype(str) + "-01-01")

    agg["ticket_medio"] = agg["faturamento_liquido"] / agg["qtd_transacoes"]
    agg["taxa_devolucao"] = (
        agg["devolucoes_valor"] / agg["faturamento_bruto"]
    ).replace([np.inf, -np.inf], np.nan)

    agg["faturamento_bruto_cum"] = agg["faturamento_bruto"].cumsum()
    agg["faturamento_liquido_cum"] = agg["faturamento_liquido"].cumsum()

    agg["var_faturamento"] = agg["faturamento_liquido"].pct_change() * 100

    return agg


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(0).astype(float)
    v = values.fillna(0).astype(float)
    denom = w.sum()
    return float((v * w).sum() / denom) if denom > 0 else 0.0


def render_performance_comercial(
    df_vendas_periodo: pd.DataFrame,
    df_vendas_full: pd.DataFrame,
    df_metas: pd.DataFrame,
    inicio: datetime.date,
    fim: datetime.date,
    plotly_font: dict,
    chart_template: str,
):
    # Controles específicos
    st.sidebar.subheader("Performance Comercial")
    with st.sidebar.container(border=True):
        inatividade_dias = st.slider(
            "Inatividade para reativação (dias)",
            min_value=15,
            max_value=180,
            value=60,
            step=5,
        )
        top_n = st.slider("Top N", min_value=5, max_value=20, value=10, step=1)

    fatur_liq = float(df_vendas_periodo["valor_liquido"].sum())
    fatur_bruto = float(df_vendas_periodo["valor_bruto"].sum())
    margem_total = float(df_vendas_periodo["lucro_bruto"].sum())
    margem_pct = (margem_total / fatur_liq * 100) if fatur_liq > 0 else 0.0
    margem_por_tx = float(df_vendas_periodo["lucro_bruto"].mean()) if len(df_vendas_periodo) else 0.0

    # Clientes
    first_purchase = df_vendas_full.groupby("cliente_id")["data"].min()
    new_ids = first_purchase[(first_purchase.dt.date >= inicio) & (first_purchase.dt.date <= fim)].index
    novos_clientes = int(len(new_ids))
    vol_novos = float(df_vendas_periodo[df_vendas_periodo["cliente_id"].isin(new_ids)]["valor_liquido"].sum())

    ativos_ids = df_vendas_periodo["cliente_id"].unique()
    clientes_ativos = int(len(ativos_ids))
    returning_mask = df_vendas_periodo["cliente_id"].map(lambda cid: first_purchase.loc[cid].date() < inicio)
    vol_ativos_base = float(df_vendas_periodo[returning_mask]["valor_liquido"].sum()) if len(df_vendas_periodo) else 0.0

    # Reativados: compra no período após inatividade > X dias
    df_before = df_vendas_full[df_vendas_full["data"].dt.date < inicio]
    last_before = df_before.groupby("cliente_id")["data"].max()
    last_before_dt = last_before.reindex(ativos_ids)
    dias_desde_ultima = (pd.Timestamp(inicio) - last_before_dt).dt.days
    reativados_ids = pd.Index(ativos_ids)[(dias_desde_ultima > inatividade_dias).fillna(False)]
    reativados = int(len(reativados_ids))

    # Inativos no início (base para taxa de reativação)
    clientes_com_hist = last_before.index
    inativos_no_inicio = clientes_com_hist[(pd.Timestamp(inicio) - last_before).dt.days > inatividade_dias]
    taxa_reativacao = (reativados / len(inativos_no_inicio) * 100) if len(inativos_no_inicio) > 0 else 0.0

    # Frequência média de compra (transações por cliente ativo)
    freq_media = (len(df_vendas_periodo) / clientes_ativos) if clientes_ativos > 0 else 0.0

    # Churn: clientes ativos no período anterior (mesma duração) que não compraram no atual
    dur = (pd.Timestamp(fim) - pd.Timestamp(inicio)).days + 1
    prev_inicio = (pd.Timestamp(inicio) - pd.Timedelta(days=dur)).date()
    prev_fim = (pd.Timestamp(inicio) - pd.Timedelta(days=1)).date()
    df_prev = df_vendas_full[
        (df_vendas_full["data"].dt.date >= prev_inicio)
        & (df_vendas_full["data"].dt.date <= prev_fim)
    ]
    prev_ativos = set(df_prev["cliente_id"].unique())
    curr_ativos = set(ativos_ids)
    churn_qtd = len(prev_ativos - curr_ativos) if prev_ativos else 0
    churn_rate = (churn_qtd / len(prev_ativos) * 100) if prev_ativos else 0.0

    # Projeção simples: run-rate do período selecionado para os próximos 30 dias
    daily = (
        df_vendas_periodo.assign(data_dia=df_vendas_periodo["data"].dt.floor("D"))
        .groupby("data_dia", as_index=False)
        .agg(faturamento_liquido=("valor_liquido", "sum"))
        .rename(columns={"data_dia": "data"})
    )
    daily["data"] = pd.to_datetime(daily["data"])
    daily = daily.sort_values("data")
    horizonte = 30
    if len(daily) >= 7:
        dias_obs = max((daily["data"].dt.date.max() - daily["data"].dt.date.min()).days + 1, 1)
        run_rate = daily["faturamento_liquido"].sum() / dias_obs
    else:
        run_rate = 0.0
    proj_30d = run_rate * horizonte

    # PMR/DSO (dias): usa recebimento ou "hoje" para abertos
    ref_today = pd.Timestamp(df_vendas_full["data"].max().date())
    receb = df_vendas_periodo["data_recebimento"].fillna(ref_today)
    dso_dias = (receb - df_vendas_periodo["data"]).dt.days.clip(lower=0)
    dso_medio = float(dso_dias.mean()) if len(dso_dias) else 0.0

    # Metas (somente para o ano base das metas)
    ano_meta = int(df_metas["ano"].max()) if len(df_metas) else None
    df_meta_periodo = df_metas.copy()
    meses_periodo = sorted(df_vendas_periodo["data"].dt.month.unique().tolist())
    if len(df_vendas_periodo) and ano_meta is not None:
        ano_sel = int(df_vendas_periodo["ano"].max())
    else:
        ano_sel = None

    atingimento_geral = None
    if ano_sel == ano_meta and len(meses_periodo):
        meta_total = float(df_meta_periodo[df_meta_periodo["mes_num"].isin(meses_periodo)]["meta_faturamento_liquido"].sum())
        atingimento_geral = (fatur_liq / meta_total * 100) if meta_total > 0 else None

    st.subheader("Visão geral")
    kpis = [
        {
            "title": "Faturamento Líquido",
            "value": format_currency(fatur_liq),
            "sublabel": "Receita no período",
            "icon": "💼",
            "accent": "#2563eb",
        },
        {
            "title": "Margem Bruta Total",
            "value": format_currency(margem_total),
            "sublabel": f"Margem: {margem_pct:.1f}%",
            "icon": "🧮",
            "accent": "#10b981",
        },
        {
            "title": "Margem Bruta por Transação",
            "value": format_currency(margem_por_tx),
            "sublabel": "Média por venda",
            "icon": "🧾",
            "accent": "#f97316",
        },
        {
            "title": "Faturamento vs Meta (Geral)",
            "value": f"{atingimento_geral:.1f}%" if atingimento_geral is not None else "—",
            "sublabel": "Atingimento no período",
            "icon": "🎯",
            "accent": "#6366f1",
        },
        {
            "title": "Novos Clientes",
            "value": f"{novos_clientes:,}".replace(",", "."),
            "sublabel": f"Vendas novos: {format_currency_compact(vol_novos)}",
            "icon": "🆕",
            "accent": "#0ea5e9",
        },
        {
            "title": "Clientes Ativos",
            "value": f"{clientes_ativos:,}".replace(",", "."),
            "sublabel": f"Base existente: {format_currency_compact(vol_ativos_base)}",
            "icon": "👥",
            "accent": "#8b5cf6",
        },
        {
            "title": "Churn (Perda de Clientes)",
            "value": f"{churn_rate:.1f}%",
            "sublabel": f"{churn_qtd} perdidos vs período anterior",
            "icon": "📉",
            "accent": "#ef4444",
        },
        {
            "title": "Reativação",
            "value": f"{reativados:,}".replace(",", "."),
            "sublabel": f"Taxa: {taxa_reativacao:.1f}%",
            "icon": "🔄",
            "accent": "#22c55e",
        },
    ]
    render_kpi_grid(kpis)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Receita & Meta", "Margem", "Clientes", "Projeção & PMR/DSO"]
    )

    with tab1:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Faturamento por Vendedor/Equipe/Região**")
            dim = st.selectbox("Dimensão", ["vendedor", "equipe", "regiao"], index=0)
            df_dim = (
                df_vendas_periodo.groupby(dim, as_index=False)
                .agg(faturamento_liquido=("valor_liquido", "sum"))
                .sort_values("faturamento_liquido", ascending=False)
                .head(top_n)
            )
            fig = px.bar(
                df_dim,
                x=dim,
                y="faturamento_liquido",
                template=chart_template,
                color_discrete_sequence=["#2563eb"],
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), font=plotly_font)
            fig.update_yaxes(tickprefix="R$ ", separatethousands=True, rangemode="tozero")
            st.plotly_chart(fig, width="stretch")

        with col_b:
            st.markdown("**Faturamento vs. Meta**")
            nivel = st.selectbox("Nível", ["Vendedor", "Equipe", "Geral"], index=0)
            if ano_sel != ano_meta:
                st.info("Metas mock disponíveis apenas para o último ano completo. Ajuste o período para esse ano.")
            else:
                atual_v = (
                    df_vendas_periodo.assign(mes_num=lambda d: d["data"].dt.month)
                    .groupby(["vendedor", "equipe", "mes_num"], as_index=False)
                    .agg(faturamento_liquido=("valor_liquido", "sum"))
                )
                meta_v = df_meta_periodo[df_meta_periodo["mes_num"].isin(meses_periodo)]
                df_vm = atual_v.merge(meta_v, on=["vendedor", "equipe", "mes_num"], how="left")
                if nivel == "Vendedor":
                    grp_cols = ["vendedor"]
                    label_col = "vendedor"
                elif nivel == "Equipe":
                    grp_cols = ["equipe"]
                    label_col = "equipe"
                else:
                    grp_cols = []
                    label_col = "label"

                if grp_cols:
                    df_vm = (
                        df_vm.groupby(grp_cols, as_index=False)
                        .agg(
                            faturamento_liquido=("faturamento_liquido", "sum"),
                            meta_faturamento_liquido=("meta_faturamento_liquido", "sum"),
                        )
                    )
                    df_vm["atingimento"] = np.where(
                        df_vm["meta_faturamento_liquido"] > 0,
                        df_vm["faturamento_liquido"] / df_vm["meta_faturamento_liquido"] * 100,
                        np.nan,
                    )
                    df_vm = df_vm.sort_values("atingimento", ascending=False).head(top_n)
                    x_vals = df_vm[label_col]
                    y_real = df_vm["faturamento_liquido"]
                    y_meta = df_vm["meta_faturamento_liquido"]
                else:
                    real_total = float(df_vm["faturamento_liquido"].sum())
                    meta_total = float(df_vm["meta_faturamento_liquido"].sum())
                    x_vals = ["Geral"]
                    y_real = [real_total]
                    y_meta = [meta_total]

                figm = go.Figure()
                figm.add_trace(
                    go.Bar(
                        x=x_vals,
                        y=y_real,
                        name="Realizado",
                        marker_color="#2563eb",
                    )
                )
                figm.add_trace(
                    go.Bar(
                        x=x_vals,
                        y=y_meta,
                        name="Meta",
                        marker_color="#94a3b8",
                        opacity=0.65,
                    )
                )
                figm.update_layout(
                    template=chart_template,
                    height=360,
                    barmode="group",
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=plotly_font,
                )
                figm.update_yaxes(tickprefix="R$ ", separatethousands=True, rangemode="tozero")
                st.plotly_chart(figm, width="stretch")

    with tab2:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Margem Bruta por Vendedor/Equipe/Região**")
            dim2 = st.selectbox("Dimensão ", ["vendedor", "equipe", "regiao"], index=1)
            df_m = (
                df_vendas_periodo.groupby(dim2, as_index=False)
                .agg(
                    margem_bruta=("lucro_bruto", "sum"),
                    faturamento_liquido=("valor_liquido", "sum"),
                )
            )
            df_m["margem_pct"] = np.where(df_m["faturamento_liquido"] > 0, df_m["margem_bruta"] / df_m["faturamento_liquido"] * 100, 0)
            df_m = df_m.sort_values("margem_bruta", ascending=False).head(top_n)
            figm2 = px.bar(
                df_m,
                x=dim2,
                y="margem_bruta",
                template=chart_template,
                color_discrete_sequence=["#10b981"],
            )
            figm2.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), font=plotly_font)
            figm2.update_yaxes(tickprefix="R$ ", separatethousands=True, rangemode="tozero")
            st.plotly_chart(figm2, width="stretch")

        with col_b:
            st.markdown("**Margem % (Top)**")
            df_mp = df_m.sort_values("margem_pct", ascending=False).head(top_n)
            figmp = px.bar(
                df_mp,
                x=dim2,
                y="margem_pct",
                template=chart_template,
                color_discrete_sequence=["#f97316"],
            )
            figmp.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), font=plotly_font)
            figmp.update_yaxes(ticksuffix="%", rangemode="tozero")
            st.plotly_chart(figmp, width="stretch")

    with tab3:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Novos vs Ativos (vendas)**")
            df_pie = pd.DataFrame(
                {
                    "segmento": ["Novos", "Base existente"],
                    "faturamento_liquido": [vol_novos, vol_ativos_base],
                }
            )
            figp = px.pie(
                df_pie,
                names="segmento",
                values="faturamento_liquido",
                template=chart_template,
                color="segmento",
                color_discrete_map={"Novos": "#0ea5e9", "Base existente": "#6366f1"},
                hole=0.55,
            )
            figp.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), font=plotly_font, legend_title_text="")
            st.plotly_chart(figp, width="stretch")

        with col_b:
            st.markdown("**Frequência média de compra**")
            df_freq = (
                df_vendas_periodo.assign(
                    mes=df_vendas_periodo["data"].dt.to_period("M").dt.to_timestamp()
                )
                .groupby("mes", as_index=False)
                .agg(transacoes=("id_venda", "count"), clientes=("cliente_id", "nunique"))
            )
            df_freq["freq_media"] = np.where(df_freq["clientes"] > 0, df_freq["transacoes"] / df_freq["clientes"], 0.0)
            figf = px.bar(
                df_freq,
                x="mes",
                y="freq_media",
                template=chart_template,
                color_discrete_sequence=["#8b5cf6"],
            )
            figf.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), font=plotly_font)
            figf.update_yaxes(rangemode="tozero")
            st.plotly_chart(figf, width="stretch")

    with tab4:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Projeção de faturamento (run-rate)**")
            df_proj = pd.DataFrame(
                {
                    "item": ["Faturamento (período)", f"Projeção próximos {horizonte} dias"],
                    "valor": [fatur_liq, proj_30d],
                }
            )
            figpr = px.bar(
                df_proj,
                x="item",
                y="valor",
                template=chart_template,
                color_discrete_sequence=["#2563eb", "#10b981"],
            )
            figpr.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), font=plotly_font, showlegend=False)
            figpr.update_yaxes(tickprefix="R$ ", separatethousands=True, rangemode="tozero")
            st.plotly_chart(figpr, width="stretch")

        with col_b:
            st.markdown("**PMR/DSO por Vendedor e Região**")
            df_dso = df_vendas_periodo.copy()
            receb = df_dso["data_recebimento"].fillna(ref_today)
            df_dso["dso_dias"] = (receb - df_dso["data"]).dt.days.clip(lower=0)
            dso_vend = (
                df_dso.groupby(["regiao", "vendedor"], as_index=False)
                .apply(lambda g: pd.Series({"dso_medio": _weighted_mean(g["dso_dias"], g["valor_liquido"])}))
                .reset_index(drop=True)
            )
            dso_vend = dso_vend.sort_values("dso_medio", ascending=True).head(top_n)
            figd = px.bar(
                dso_vend,
                x="dso_medio",
                y="vendedor",
                color="regiao",
                orientation="h",
                template=chart_template,
                color_discrete_sequence=["#2563eb", "#10b981", "#f97316", "#6366f1"],
            )
            figd.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), font=plotly_font, legend_title_text="")
            figd.update_xaxes(title="Dias (média ponderada)")
            st.plotly_chart(figd, width="stretch")


def main():
    st.set_page_config(
        page_title="Dashboards de Vendas",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_typography()

    df_vendas, df, df_metas = gerar_dados_mock()
    color_sequence = ["#2563eb", "#10b981", "#f97316", "#6366f1"]
    chart_template = "plotly_white"
    plotly_font = dict(family="Nunito", color="#0f172a", size=12)

    # Menu lateral (sidebar)
    st.sidebar.subheader("Dashboard")
    dashboard = st.sidebar.radio(
        "Selecione",
        options=["Análise Operacional", "Performance Comercial"],
        label_visibility="collapsed",
    )

    st.sidebar.subheader("Filtros")
    with st.sidebar.container(border=True):
        periodo_visualizacao = st.selectbox(
            "Granularidade do período",
            ["Diário", "Semanal", "Mensal", "Trimestral", "Anual"],
            index=2,
        )

        data_inicio = df["data"].min().date()
        data_fim = df["data"].max().date()
        ano_padrao = int(df["ano"].max())
        inicio_padrao = datetime(ano_padrao, 1, 1).date()
        fim_padrao = datetime(ano_padrao, 12, 31).date()
        intervalo = st.date_input(
            "Período",
            value=(inicio_padrao, fim_padrao),
            min_value=data_inicio,
            max_value=data_fim,
        )

        st.toggle(
            "Exibir valores brutos e líquidos",
            value=True,
            key="show_bruto_liq",
        )

    if isinstance(intervalo, tuple) and len(intervalo) == 2:
        inicio, fim = intervalo
        mask = (df["data"].dt.date >= inicio) & (df["data"].dt.date <= fim)
        df_periodo = df.loc[mask].copy()
        mask_v = (df_vendas["data"].dt.date >= inicio) & (df_vendas["data"].dt.date <= fim)
        df_vendas_periodo = df_vendas.loc[mask_v].copy()
    else:
        df_periodo = df.copy()
        df_vendas_periodo = df_vendas.copy()
        inicio = df_periodo["data"].min().date()
        fim = df_periodo["data"].max().date()

    if df_periodo.empty:
        st.warning("Nenhum dado para o período selecionado.")
        return

    if dashboard == "Performance Comercial":
        st.title("Dashboard de Performance Comercial")
        st.caption("Receita, metas, margem, clientes e eficiência de recebimento.")
        render_performance_comercial(
            df_vendas_periodo=df_vendas_periodo,
            df_vendas_full=df_vendas,
            df_metas=df_metas,
            inicio=inicio,
            fim=fim,
            plotly_font=plotly_font,
            chart_template=chart_template,
        )
        return

    st.title("Dashboard Operacional de Vendas")
    st.caption("Visão geral do volume de vendas, faturamento e indicadores operacionais.")

    df_agg = calcular_periodos(df_periodo, periodo_visualizacao)

    # KPIs gerais
    faturamento_bruto_total = df_periodo["faturamento_bruto"].sum()
    faturamento_liquido_total = df_periodo["faturamento_liquido"].sum()
    qtd_transacoes_total = df_periodo["qtd_transacoes"].sum()
    ticket_medio = (
        faturamento_liquido_total / qtd_transacoes_total
        if qtd_transacoes_total > 0
        else 0
    )

    # períodos para comparativos
    df_agg_sorted = df_agg.sort_values("periodo")
    if len(df_agg_sorted) >= 2:
        atual = df_agg_sorted.iloc[-1]
        anterior = df_agg_sorted.iloc[-2]
        crescimento_periodo = (
            (atual["faturamento_liquido"] / anterior["faturamento_liquido"] - 1) * 100
            if anterior["faturamento_liquido"] > 0
            else np.nan
        )
    else:
        crescimento_periodo = np.nan

    ano_atual = df_periodo["ano"].max()
    ano_anterior = ano_atual - 1
    ytd_atual = df[df["ano"] == ano_atual]["faturamento_liquido"].sum()
    ytd_anterior = df[df["ano"] == ano_anterior]["faturamento_liquido"].sum()
    crescimento_ytd = (
        (ytd_atual / ytd_anterior - 1) * 100 if ytd_anterior > 0 else None
    )

    taxa_devolucoes = (
        df_periodo["devolucoes_valor"].sum() / faturamento_bruto_total * 100
        if faturamento_bruto_total > 0
        else 0
    )

    st.subheader("Visão geral")

    total_itens = df_periodo["qtd_itens"].sum()
    kpis = [
        {
            "title": "Faturamento Bruto",
            "value": format_currency(faturamento_bruto_total),
            "sublabel": "Total do período selecionado",
            "icon": "💰",
            "accent": "#2563eb",
        },
        {
            "title": "Faturamento Líquido",
            "value": format_currency(faturamento_liquido_total),
            "sublabel": "Após descontos, impostos e devoluções",
            "delta": crescimento_periodo if not np.isnan(crescimento_periodo) else None,
            "help_text": "Variação vs. período imediatamente anterior",
            "icon": "📈",
            "accent": "#10b981",
        },
        {
            "title": "Ticket Médio",
            "value": format_currency(ticket_medio),
            "sublabel": f"{int(qtd_transacoes_total)} transações no período",
            "icon": "🧾",
            "accent": "#f97316",
        },
        {
            "title": "Taxa de Devoluções",
            "value": f"{taxa_devolucoes:.2f}%",
            "sublabel": "(valor devolvido / faturamento bruto)",
            "icon": "↩️",
            "accent": "#ef4444",
        },
        {
            "title": "Faturamento Acumulado no Ano (YTD)",
            "value": format_currency(ytd_atual),
            "sublabel": f"Ano {ano_atual}",
            "delta": crescimento_ytd,
            "help_text": "Variação vs. YTD do ano anterior"
            if ytd_anterior > 0
            else "Sem dados do ano anterior no mock",
            "icon": "📊",
            "accent": "#6366f1",
        },
        {
            "title": "YTD Ano Anterior",
            "value": format_currency(ytd_anterior),
            "sublabel": f"Ano {ano_anterior}",
            "icon": "🗓️",
            "accent": "#64748b",
        },
        {
            "title": "Volume de Vendas (itens)",
            "value": f"{int(total_itens):,}".replace(",", "."),
            "sublabel": "Quantidade de itens vendidos no período",
            "icon": "🧺",
            "accent": "#0ea5e9",
        },
        {
            "title": "Granularidade atual",
            "value": periodo_visualizacao,
            "sublabel": "Aplicada aos gráficos",
            "icon": "⏱️",
            "accent": "#8b5cf6",
        },
    ]
    render_kpi_grid(kpis)

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Faturamento", "Crescimento e Comparativos", "Indicadores Operacionais"]
    )

    with tab1:
        col_fat, col_cum = st.columns((2, 1.6), gap="large")

        with col_fat:
            st.markdown("**Faturamento (bruto) + volume + líquido**")
            df_line = df_agg_sorted.copy()

            volume_col = "qtd_transacoes"
            volume_nome = "Volume (Transações)"

            # Combo: barras (bruto + volume lado a lado) + linha (líquido)
            fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

            fig_combo.add_trace(
                go.Bar(
                    x=df_line["periodo"],
                    y=df_line["faturamento_bruto"],
                    name="Faturamento Bruto",
                    marker_color="#2563eb",
                    opacity=0.95,
                    offsetgroup="bruto",
                    alignmentgroup="fat_vol",
                ),
                secondary_y=False,
            )

            fig_combo.add_trace(
                go.Bar(
                    x=df_line["periodo"],
                    y=df_line[volume_col],
                    name=volume_nome,
                    marker_color="#6366f1",
                    opacity=0.9,
                    offsetgroup="volume",
                    alignmentgroup="fat_vol",
                ),
                secondary_y=True,
            )

            fig_combo.add_trace(
                go.Scatter(
                    x=df_line["periodo"],
                    y=df_line["faturamento_liquido"],
                    name="Faturamento Líquido",
                    mode="lines+markers",
                    line=dict(color="#10b981", width=3),
                    marker=dict(size=6),
                ),
                secondary_y=False,
            )

            fig_combo.update_layout(
                template=chart_template,
                height=360,
                barmode="group",
                bargap=0.25,
                bargroupgap=0.08,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=40, b=10),
                hovermode="x unified",
                font=plotly_font,
            )

            fig_combo.update_yaxes(
                title_text="Faturamento",
                tickprefix="R$ ",
                separatethousands=True,
                secondary_y=False,
                rangemode="tozero",
                gridcolor="rgba(148,163,184,0.25)",
            )
            fig_combo.update_yaxes(
                title_text=volume_nome,
                secondary_y=True,
                rangemode="tozero",
                showgrid=False,
                tickformat=",d",
            )

            st.plotly_chart(fig_combo, width="stretch")

        with col_cum:
            st.markdown("**Faturamento acumulado**")
            fig_area = px.area(
                df_agg_sorted,
                x="periodo",
                y="faturamento_liquido_cum",
                template=chart_template,
                color_discrete_sequence=["#2563eb"],
            )
            fig_area.update_layout(
                height=340,
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                font=plotly_font,
            )
            fig_area.update_yaxes(tickprefix="R$ ", separatethousands=True)
            st.plotly_chart(fig_area, width="stretch")

    with tab2:
        col_growth, col_comp = st.columns(2, gap="large")

        with col_growth:
            st.markdown("**Crescimento % do faturamento**")
            df_growth = df_agg_sorted.copy()
            df_growth["direcao"] = np.where(
                df_growth["var_faturamento"].fillna(0) < 0, "Queda", "Crescimento"
            )
            fig_growth = px.bar(
                df_growth,
                x="periodo",
                y="var_faturamento",
                color="direcao",
                template=chart_template,
                color_discrete_map={
                    "Crescimento": "#16a34a",
                    "Queda": "#dc2626",
                },
            )
            fig_growth.update_layout(
                height=340,
                legend_title_text="",
                margin=dict(l=10, r=10, t=30, b=10),
                font=plotly_font,
            )
            fig_growth.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor="#94a3b8")
            st.plotly_chart(fig_growth, width="stretch")

        with col_comp:
            st.markdown("**Comparativo (Ano atual x Ano anterior) — barras lado a lado**")

            # Usa o período selecionado como referência e compara com o mesmo intervalo do ano anterior
            if isinstance(intervalo, tuple) and len(intervalo) == 2:
                inicio_sel, fim_sel = intervalo
            else:
                inicio_sel, fim_sel = df_periodo["data"].min().date(), df_periodo["data"].max().date()

            inicio_prev = (pd.Timestamp(inicio_sel) - pd.DateOffset(years=1)).date()
            fim_prev = (pd.Timestamp(fim_sel) - pd.DateOffset(years=1)).date()

            df_curr = df[(df["data"].dt.date >= inicio_sel) & (df["data"].dt.date <= fim_sel)].copy()
            df_prev = df[(df["data"].dt.date >= inicio_prev) & (df["data"].dt.date <= fim_prev)].copy()

            if df_prev.empty:
                st.info("Sem dados do ano anterior no mock para o intervalo selecionado.")
            else:
                ano_curr = int(df_curr["ano"].max()) if not df_curr.empty else int(df["ano"].max())
                ano_prev = int(df_prev["ano"].max())

                curr_m = (
                    df_curr.assign(mes_num=df_curr["data"].dt.month)
                    .groupby("mes_num", as_index=False)
                    .agg(faturamento_liquido=("faturamento_liquido", "sum"))
                    .assign(ano=str(ano_curr))
                )
                prev_m = (
                    df_prev.assign(mes_num=df_prev["data"].dt.month)
                    .groupby("mes_num", as_index=False)
                    .agg(faturamento_liquido=("faturamento_liquido", "sum"))
                    .assign(ano=str(ano_prev))
                )

                df_comp = pd.concat([prev_m, curr_m], ignore_index=True)
                meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
                df_comp["mes"] = df_comp["mes_num"].apply(lambda m: meses[m - 1])

                fig_comp = px.bar(
                    df_comp,
                    x="mes",
                    y="faturamento_liquido",
                    color="ano",
                    barmode="group",
                    template=chart_template,
                    color_discrete_sequence=["#6366f1", "#2563eb"],
                )
                fig_comp.update_layout(
                    height=340,
                    legend_title_text="",
                    margin=dict(l=10, r=10, t=30, b=10),
                    hovermode="x unified",
                    font=plotly_font,
                )
                fig_comp.update_yaxes(tickprefix="R$ ", separatethousands=True, rangemode="tozero")
                st.plotly_chart(fig_comp, width="stretch")

    with tab3:
        col_ticket, col_devol = st.columns(2, gap="large")

        with col_ticket:
            st.markdown("**Ticket médio por período**")
            fig_ticket = px.bar(
                df_agg_sorted,
                x="periodo",
                y="ticket_medio",
                template=chart_template,
                height=360,
                color_discrete_sequence=["#f97316"],
            )
            fig_ticket.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                font=plotly_font,
            )
            fig_ticket.update_yaxes(
                tickprefix="R$ ",
                separatethousands=True,
                rangemode="tozero",
                gridcolor="rgba(148,163,184,0.25)",
            )
            st.plotly_chart(fig_ticket, width="stretch")

        with col_devol:
            st.markdown("**Taxa de devoluções/cancelamentos**")
            df_devol = df_agg_sorted.copy()
            df_devol["taxa_devolucao_pct"] = df_devol["taxa_devolucao"] * 100
            fig_devol = px.bar(
                df_devol,
                x="periodo",
                y="taxa_devolucao_pct",
                template=chart_template,
                color_discrete_sequence=["#ef4444"],
            )
            fig_devol.update_layout(
                height=340,
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                font=plotly_font,
            )
            fig_devol.update_yaxes(ticksuffix="%", rangemode="tozero")
            st.plotly_chart(fig_devol, width="stretch")


if __name__ == "__main__":
    main()

