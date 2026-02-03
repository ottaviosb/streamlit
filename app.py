import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import textwrap

from datetime import datetime, timedelta
from plotly.subplots import make_subplots


@st.cache_data
def gerar_dados_mock(ano: int | None = None) -> pd.DataFrame:
    """Gera dados mock de vendas (agregado diário) para ano atual + ano anterior."""
    hoje = datetime.today().date()
    ano_base = ano if ano is not None else (hoje.year - 1)  # último ano completo

    def gerar_ano(ano_ref: int, seed: int) -> pd.DataFrame:
        datas = pd.date_range(start=f"{ano_ref}-01-01", end=f"{ano_ref}-12-31", freq="D")
        rng = np.random.default_rng(seed)

        n_dias = len(datas)
        dow = datas.dayofweek.to_numpy()  # 0=Seg ... 6=Dom
        doy = datas.dayofyear.to_numpy()

        # Efeito dia da semana (fim de semana reduz)
        fator_dow = np.select([dow == 5, dow == 6], [0.78, 0.72], default=1.0)

        # Sazonalidade anual suave + "picos" (ex.: maio, novembro)
        sazonal_anual = 1.0 + 0.18 * np.sin(2 * np.pi * (doy / 365.0))
        pico_maio = np.exp(-0.5 * ((doy - 140) / 18) ** 2) * 0.18
        pico_nov = np.exp(-0.5 * ((doy - 325) / 22) ** 2) * 0.35

        # Tendência leve (pequena diferença entre anos)
        trend = 0.95 + 0.10 * (np.arange(n_dias) / (n_dias - 1))

        # Volume de transações: Poisson com lambdas ajustados
        lam = 85 * fator_dow * sazonal_anual * (1 + pico_maio + pico_nov) * trend
        lam = np.clip(lam, 10, None)
        qtd_transacoes = rng.poisson(lam).astype(int)

        # Ticket: lognormal ajustado por sazonalidade
        ticket_base = 65 * sazonal_anual * (1 + 0.15 * pico_nov) * trend
        sigma = 0.45
        mu = np.log(np.maximum(ticket_base, 1)) - 0.5 * sigma**2
        ticket_medio_dia = rng.lognormal(mean=mu, sigma=sigma)

        bruto = (qtd_transacoes * ticket_medio_dia).round(2)

        itens_por_transacao = rng.normal(loc=2.0, scale=0.35, size=n_dias).clip(1.2, 2.8)
        qtd_itens = np.maximum((qtd_transacoes * itens_por_transacao).round(0), 0).astype(int)

        promo = (pico_nov > 0.10).astype(float)
        evento = (rng.random(n_dias) < 0.04).astype(float)
        taxa_desconto = (
            0.06 + 0.10 * promo + 0.05 * evento + rng.normal(0, 0.01, n_dias)
        ).clip(0.02, 0.25)
        descontos = (bruto * taxa_desconto).round(2)

        impostos = (bruto * 0.12).round(2)

        taxa_devol = (
            0.012
            + 0.010 * promo
            + 0.006 * (qtd_transacoes > np.quantile(qtd_transacoes, 0.9))
        ).clip(0.005, 0.06)
        devolucoes_valor = (bruto * taxa_devol).round(2)

        liquido = (bruto - descontos - impostos - devolucoes_valor).round(2)

        df = pd.DataFrame(
            {
                "data": datas,
                "faturamento_bruto": bruto,
                "descontos": descontos,
                "impostos": impostos,
                "devolucoes_valor": devolucoes_valor,
                "faturamento_liquido": liquido,
                "qtd_transacoes": qtd_transacoes,
                "qtd_itens": qtd_itens,
            }
        )
        df["ano"] = df["data"].dt.year
        df["mes"] = df["data"].dt.to_period("M").dt.to_timestamp()
        df["semana"] = df["data"].dt.to_period("W").dt.start_time
        df["trimestre"] = df["data"].dt.to_period("Q").dt.start_time
        return df

    # ano anterior + ano base (último ano completo)
    df_prev = gerar_ano(ano_base - 1, seed=41)
    df_curr = gerar_ano(ano_base, seed=42)
    return pd.concat([df_prev, df_curr], ignore_index=True)


def format_currency(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


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


def inject_dashboard_style():
    if st.session_state.get("dashboard_style_injected"):
        return
    st.session_state["dashboard_style_injected"] = True

    st.markdown(
        textwrap.dedent(
            """
            <style>
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 16px;
                margin-bottom: 8px;
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

        sub_html = f'<div class="kpi-sub">{sublabel}</div>' if sublabel else ""
        help_html = f'<div class="kpi-help">{help_text}</div>' if help_text else ""

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


def main():
    st.set_page_config(
        page_title="Dashboard Operacional - Vendas",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Dashboard Operacional de Vendas")
    st.caption("Visão geral do volume de vendas, faturamento e indicadores operacionais.")

    df = gerar_dados_mock()
    color_sequence = ["#2563eb", "#10b981", "#f97316", "#6366f1"]
    chart_template = "plotly_white"

    with st.container(border=True):
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            periodo_visualizacao = st.selectbox(
                "Granularidade do período",
                ["Diário", "Semanal", "Mensal", "Trimestral", "Anual"],
                index=2,
            )

        with col2:
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

        with col3:
            st.write("")
            st.toggle(
                "Exibir valores brutos e líquidos",
                value=True,
                key="show_bruto_liq",
            )

    if isinstance(intervalo, tuple) and len(intervalo) == 2:
        inicio, fim = intervalo
        mask = (df["data"].dt.date >= inicio) & (df["data"].dt.date <= fim)
        df_periodo = df.loc[mask].copy()
    else:
        df_periodo = df.copy()

    if df_periodo.empty:
        st.warning("Nenhum dado para o período selecionado.")
        return

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
            )
            fig_devol.update_yaxes(ticksuffix="%", rangemode="tozero")
            st.plotly_chart(fig_devol, width="stretch")


if __name__ == "__main__":
    main()

