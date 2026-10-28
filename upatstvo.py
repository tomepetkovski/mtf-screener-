# -*- coding: utf-8 -*-
"""Упатство за користење на апликацијата MTF EMA Screener Pro.

Овој модул содржи опис на сетинзите од левата страна (sidebar) со кратко
појаснување за што служи секој од нив.
"""

GUIDE = """
## Упатство за користење

### Основни сетинзи
- **Market** — избор на пазар: крипто или акции (US stocks).
- **Use live price** — користи тековна цена (исклучено = последна затворена свеќа).
- **EMA fast / EMA slow** — периоди на брзата и бавната EMA (пр. 50 / 200).
- **Max distance from EMA (%)** — макс. оддалеченост од EMA во % (0 = исклучено).
- **Top N by volume** — колку инструменти (по волумен) да се скенираат.
- **Min daily $ volume / Min 24h volume** — минимален дневен/24ч волумен за филтрирање.
- **Exchanges** — берзи за акции (NASDAQ, NYSE...).
- **Data sources (priority order)** — приоритет на извори на податоци за крипто.
- **Fast mode** — запира по 2 неуспешни timeframe-и (помалку API повици).
- **Auto-refresh (minutes)** — автоматско освежување (0 = исклучено).
- **Telegram alerts** — известувања преку Telegram (бара secrets).

### Дополнителни сигнали (Extra signals)
- **Breakout·crash·flow·OI·book·funding·RS·earnings** — вклучува дополнителни сигнали за MATCH/WATCH редовите.
- **Signal timeframe** — timeframe за сигналите.
- **Breakout lookback (bars)** — број свеќи за breakout.
- **Breakout volume x average** — множител на волумен за breakout.
- **OI / price change lookback** — lookback за OI/цена.
- **Crash protector** — крие MATCH/WATCH со ризик ≥ праг (100 = исклучено).
- **Only breakouts** — само breakout сетапи.
- **Hide setups crowded in my direction** — крие преполни сетапи.
- **Only OI rising with price up** — само растечки OI со цена нагоре (крипто).
- **Min analyst rating / Hide stocks with earnings within** — филтри за акции (Finnhub).

### Макро настани (UTC)
- **Macro events** — еден настан по ред: `YYYY-MM-DD HH:MM label`. Настани во следните 24ч го намалуваат conviction score.

### Сигнал дневник (Signal journal)
- **Log MATCH signals to journal** — логира MATCH сигнали во локална SQLite база.
- **Journal SL (ATR multiple)** — стоп-лос како ATR множител.
- **Journal TPs in R** — take-profit нивоа во R (пр. 1,2,3).
- **Time stop (bars)** — временски стоп по број свеќи.
- **Don't re-log same symbol within (hours)** — cooldown за повторно логирање.

### EMA критериуми и правила по timeframe
- **Preset** — готов пресет на правила (или Custom за рачно).
- **Rules per timeframe** — за секој timeframe: Price/EMA50, Price/EMA200, EMA50/EMA200 услови.
"""


def render_upatstvo(st):
    with st.expander("📘 Упатство за користење", expanded=False):
        st.markdown(GUIDE)
"""


def upatstvo_markdown():
    return GUIDE
"""


# keep simple API
def show(st):
    render_upatstvo(st)
