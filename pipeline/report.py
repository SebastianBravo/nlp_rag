"""Escritura de reportes: tablas CSV, figuras PNG y JSON de metadatos. Solo agregados."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

# Paleta categórica validada (slots 1 y 2) con un color fijo por colección, más tinta y superficie clara.
COLLECTION_COLORS = {'bluesky_posts': '#2a78d6', 'url_crawls': '#eb6834'}
COLLECTION_LABELS = {'bluesky_posts': 'Bluesky', 'url_crawls': 'Prensa (url_crawls)'}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SEQUENTIAL = LinearSegmentedColormap.from_list('azul', ['#f4f8fe', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

def _percent_axis(axis):
    axis.set_major_formatter(PercentFormatter(1.0, decimals=None))

def _share_label(share):
    return '<0.01%' if 0 < share < 0.0001 else f'{share:.2%}'

def to_jsonable(obj):
    if isinstance(obj, dict): return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame): return to_jsonable(obj.to_dict(orient='records'))
    if isinstance(obj, (pd.Timestamp, np.datetime64)): return None if pd.isna(obj) else pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (float, np.floating)): return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if obj is pd.NaT or obj is pd.NA: return None
    return obj

def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def write_tables(tables, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f'{name}.csv', index=False, float_format='%.6g')
    return sorted(f'{name}.csv' for name in tables)

def _figure(width, height, **kwargs):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'axes.titlelocation': 'left', 'axes.titlecolor': INK, 'axes.labelcolor': INK_2,
                         'axes.edgecolor': AXIS, 'axes.facecolor': SURFACE, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
                         'legend.frameon': False, 'legend.labelcolor': INK_2, 'figure.facecolor': SURFACE})
    fig, ax = plt.subplots(figsize=(width, height), **kwargs)
    return fig, ax

def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return Path(path).name

def token_ecdf(features, thresholds, max_tokens, path):
    """Distribución acumulada de tokens del encoder por colección, con los umbrales marcados."""
    fig, ax = _figure(8, 4.5)
    for collection, color in COLLECTION_COLORS.items():
        values = np.sort(features.loc[features['collection_name'] == collection, 'n_tokens'].to_numpy())
        if not len(values): continue
        above = (values > max_tokens).mean()
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where='post', color=color, lw=2,
                label=f'{COLLECTION_LABELS[collection]}: {above:.1%} supera {max_tokens}')
    for t in thresholds:
        ax.axvline(t, color=MUTED, lw=1, ls='--')
        ax.text(t, 0.5, f'{t} tokens ', rotation=90, ha='right', va='center', color=INK_2, fontsize=9)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    _percent_axis(ax.yaxis)
    ax.set_xlabel('Tokens por documento (prefijo y tokens especiales incluidos, escala log)')
    ax.set_ylabel('Documentos con ≤ x tokens')
    ax.set_title('Longitud en tokens del encoder')
    ax.legend(loc='lower right')
    return _save(fig, path)

def length_histograms(features, path, column='char_count_calc'):
    """Histograma de longitud en caracteres, un panel por colección (escala log)."""
    fig, axes = _figure(8, 5, nrows=2, sharex=True)
    low, high = max(features[column].min(), 1), max(features[column].max(), 10)
    bins = np.logspace(math.floor(math.log10(low)), math.log10(high) + 0.05, 60)
    for ax, (collection, color) in zip(axes, COLLECTION_COLORS.items()):
        values = features.loc[features['collection_name'] == collection, column]
        ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(f'{COLLECTION_LABELS[collection]} · {len(values):,} documentos'.replace(',', '.'), fontsize=10)
        ax.set_ylabel('Documentos')
    axes[-1].set_xscale('log')
    axes[-1].set_xlabel('Caracteres por documento (escala log)')
    fig.suptitle('Longitud de los documentos', x=0.01, ha='left', fontweight='bold', color=INK)
    fig.tight_layout()
    return _save(fig, path)

def language_heatmap(matrix, path):
    """Matriz idioma declarado × detectado, normalizada por fila."""
    shares = matrix.div(matrix.sum(axis=1), axis=0)
    fig, ax = _figure(1.1 * shares.shape[1] + 2.5, 0.7 * shares.shape[0] + 1.8)
    ax.imshow(shares.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect='auto')
    ax.grid(False)
    for (i, j), value in np.ndenumerate(shares.to_numpy()):
        if matrix.iat[i, j] == 0: continue
        ax.text(j, i, f'{value:.1%}', ha='center', va='center', fontsize=9, color='#ffffff' if value > 0.55 else INK)
    ax.set_xticks(range(shares.shape[1]), shares.columns)
    ax.set_yticks(range(shares.shape[0]), [f'{lang} ({n:,})'.replace(',', '.') for lang, n in matrix.sum(axis=1).items()])
    ax.set_xlabel('Idioma detectado (lingua)')
    ax.set_ylabel('Idioma declarado (language_base)')
    ax.set_title('Acuerdo de idioma (porcentaje por fila)')
    return _save(fig, path)

def agreement_by_length_plot(table, path, min_n=30):
    """Tasa de acuerdo de idioma por rango de longitud y colección (solo rangos con al menos `min_n` documentos)."""
    fig, ax = _figure(8, 4)
    bins = list(dict.fromkeys(table['length_bin'].astype(str)))
    for collection, color in COLLECTION_COLORS.items():
        sub = table[(table['collection_name'] == collection) & (table['n'] >= min_n)]
        if sub.empty: continue
        x = [bins.index(b) for b in sub['length_bin'].astype(str)]
        ax.plot(x, sub['agreement'], color=color, lw=2, marker='o', markersize=6, label=COLLECTION_LABELS[collection])
    ax.set_xticks(range(len(bins)), bins)
    ax.set_ylim(0, 1.02)
    _percent_axis(ax.yaxis)
    ax.set_xlabel('Longitud del documento (caracteres)')
    ax.set_ylabel('Acuerdo declarado = detectado')
    ax.set_title('Acuerdo de idioma según la longitud')
    ax.legend(loc='lower right')
    return _save(fig, path)

def monthly_counts_plot(monthly, path):
    """Documentos por mes de `created_at`, un panel por colección con su propia escala."""
    fig, axes = _figure(9, 5, nrows=2, sharex=True)
    for ax, (collection, color) in zip(axes, COLLECTION_COLORS.items()):
        sub = monthly[monthly['collection_name'] == collection]
        dates = pd.to_datetime(sub['period'] + '-01')
        ax.fill_between(dates, sub['documents'], step='mid', color=color, alpha=0.25, linewidth=0)
        ax.step(dates, sub['documents'], where='mid', color=color, lw=1.5)
        ax.set_title(COLLECTION_LABELS[collection], fontsize=10)
        ax.set_ylabel('Documentos por mes')
        ax.set_ylim(bottom=0)
    fig.suptitle('Documentos por mes de publicación (created_at)', x=0.01, ha='left', fontweight='bold', color=INK)
    fig.tight_layout()
    return _save(fig, path)

def top_domains_plot(top, path):
    fig, ax = _figure(8, 0.32 * len(top) + 1.2)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top['share'], color=COLLECTION_COLORS['url_crawls'], height=0.7)
    for yi, share in zip(y, top['share']):
        ax.annotate(f'{share:.1%}', (share, yi), xytext=(4, 0), textcoords='offset points', va='center', fontsize=9, color=INK_2)
    ax.set_yticks(y, top['url_domain'])
    _percent_axis(ax.xaxis)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Proporción de documentos de prensa')
    ax.set_title(f'Los {len(top)} dominios con más documentos de prensa')
    return _save(fig, path)

def noise_plot(noise, path):
    """Proporción de documentos con cada señal de ruido, por colección."""
    noise = noise[noise['collection_name'] != 'total']
    labels = list(dict.fromkeys(noise['label']))
    fig, ax = _figure(8, 0.55 * len(labels) + 1.2)
    height = 0.38
    y = np.arange(len(labels))[::-1]
    for offset, (collection, color) in zip((height / 2 + 0.01, -height / 2 - 0.01), COLLECTION_COLORS.items()):
        sub = noise[noise['collection_name'] == collection].set_index('label').reindex(labels)
        ax.barh(y + offset, sub['share'], height=height, color=color, label=COLLECTION_LABELS[collection])
        for yi, share in zip(y + offset, sub['share']):
            if share > 0:
                ax.annotate(_share_label(share), (share, yi), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=INK_2)
    ax.set_yticks(y, labels)
    _percent_axis(ax.xaxis)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Proporción de documentos de la colección')
    ax.set_title('Señales de ruido en el texto')
    ax.legend(loc='lower right')
    return _save(fig, path)

# --- Segmentación (Fase B): un color fijo por estrategia, distinto de los de colección ---
STRATEGY_COLORS = {'A': '#1baf7a', 'B': '#6250d6'}
STRATEGY_LABELS = {'A': 'A · ventana de tokens', 'B': 'B · estructura'}

def chunk_tokens_ecdf(chunk_data, max_tokens, budget, path):
    """Distribución acumulada de tokens del encoder por chunk, un panel por colección y una línea por estrategia."""
    fig, axes = _figure(9, 4, ncols=2, sharey=True)
    for ax, collection in zip(axes, COLLECTION_LABELS):
        for strategy, data in sorted(chunk_data.items()):
            values = np.sort(data.loc[data['collection_name'] == collection, 'n_tokens'].to_numpy())
            if not len(values): continue
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where='post', lw=2,
                    color=STRATEGY_COLORS.get(strategy, INK_2), label=f'{STRATEGY_LABELS.get(strategy, strategy)} ({len(values):,})'.replace(',', '.'))
        for t, label in ((budget, f'presupuesto {budget}'), (max_tokens, f'tope {max_tokens}')):
            ax.axvline(t, color=MUTED, lw=1, ls='--')
            ax.text(t, 0.5, f'{label} ', rotation=90, ha='right', va='center', color=INK_2, fontsize=8)
        ax.set_title(COLLECTION_LABELS[collection], fontsize=10)
        ax.set_xlabel('Tokens por chunk (con prefijo)')
        ax.legend(loc='upper left', fontsize=8)
    axes[0].set_ylabel('Chunks con ≤ x tokens')
    _percent_axis(axes[0].yaxis)
    fig.suptitle('Tamaño de los chunks en tokens del encoder', x=0.01, ha='left', fontweight='bold', color=INK)
    fig.tight_layout()
    return _save(fig, path)

def chunks_per_document_plot(doc_data, path, collection='url_crawls', cap=30):
    """Distribución de chunks por documento de prensa, por estrategia (los valores mayores que `cap` se agrupan)."""
    fig, ax = _figure(8, 4)
    width = 0.42
    for offset, (strategy, docs) in zip((-width / 2, width / 2), sorted(doc_data.items())):
        values = docs.loc[docs['collection_name'] == collection, 'n_chunks'].clip(upper=cap)
        shares = values.value_counts(normalize=True).sort_index()
        ax.bar(shares.index + offset, shares.to_numpy(), width=width, color=STRATEGY_COLORS.get(strategy, INK_2),
               label=f'{STRATEGY_LABELS.get(strategy, strategy)} · media {docs.loc[docs["collection_name"] == collection, "n_chunks"].mean():.1f}')
    ax.set_xlabel(f'Chunks por documento de prensa ({cap} = {cap} o más)')
    ax.set_ylabel('Documentos')
    _percent_axis(ax.yaxis)
    ax.grid(axis='x', visible=False)
    ax.set_title('Chunks por documento de prensa')
    ax.legend(loc='upper right')
    return _save(fig, path)

def strategy_metrics_plot(table, path, collection='total'):
    """Métricas intrínsecas (proporciones) por estrategia."""
    metrics = {'mid_sentence_share': 'chunks que cortan una oración', 'removed_share': 'chunks eliminados por duplicados',
               'overlap_ratio': 'texto repetido por solapamiento', 'boilerplate_excluded_share': 'texto excluido como boilerplate',
               'single_chunk_docs_share': 'documentos de un solo chunk'}
    data = table[table['collection_name'] == collection].set_index('strategy')
    labels = list(metrics.values())
    fig, ax = _figure(8, 0.6 * len(labels) + 1.2)
    height = 0.38
    y = np.arange(len(labels))[::-1]
    for offset, strategy in zip((height / 2 + 0.01, -height / 2 - 0.01), sorted(data.index)):
        values = [float(data.loc[strategy, m]) for m in metrics]
        ax.barh(y + offset, values, height=height, color=STRATEGY_COLORS.get(strategy, INK_2), label=STRATEGY_LABELS.get(strategy, strategy))
        for yi, value in zip(y + offset, values):
            ax.annotate(_share_label(value) if value else '0%', (value, yi), xytext=(4, 0), textcoords='offset points',
                        va='center', fontsize=8, color=INK_2)
    ax.set_yticks(y, labels)
    _percent_axis(ax.xaxis)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Proporción')
    ax.set_title('Comparación intrínseca de estrategias (todas las colecciones)')
    ax.legend(loc='lower right')
    return _save(fig, path)

# --- Encoder (Fase C) ---
METRIC_COLORS = {'recall@5': '#2a78d6', 'recall@10': '#eb6834', 'ndcg@10': '#1baf7a', 'mrr': '#eda100'}
VARIANT_LABELS = {'perfil': 'formato del perfil', 'prefijo_e5': 'con prefijo "query: "', 'sin_filtro_fecha': 'sin filtro de fecha',
                  'parafrasis': 'paráfrasis', 'traduccion': 'traducción'}

def variant_metrics_plot(summary, path):
    """Métricas de recuperación (dev) por variante de consulta."""
    variants = list(summary['variant'])
    fig, ax = _figure(9, 4.5)
    width = 0.8 / len(METRIC_COLORS)
    x = np.arange(len(variants))
    for offset, (metric, color) in enumerate(METRIC_COLORS.items()):
        values = summary[metric].to_numpy(dtype=float)
        ax.bar(x + (offset - (len(METRIC_COLORS) - 1) / 2) * width, values, width=width * 0.92, color=color, label=metric)
    ax.set_xticks(x, [VARIANT_LABELS.get(v, v) for v in variants], rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    _percent_axis(ax.yaxis)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel(f'Promedio en dev (n = {int(summary["n"].iloc[0])})')
    ax.set_title('Recuperación por variante de consulta')
    ax.legend(loc='upper right', ncols=4, fontsize=8)
    return _save(fig, path)

def matryoshka_plot(mrl, path):
    """Recall@10 y nDCG@10 según la dimensión, con y sin renormalizar."""
    fig, ax = _figure(8, 4)
    for metric, color in (('recall@10', METRIC_COLORS['recall@10']), ('ndcg@10', METRIC_COLORS['ndcg@10'])):
        for renorm, style in ((True, '-'), (False, '--')):
            sub = mrl[mrl['renormalized'] == renorm].sort_values('dimension')
            if sub.empty: continue
            ax.plot(sub['dimension'], sub[metric], style, color=color, lw=2, marker='o', markersize=6,
                    label=f'{metric} {"renormalizado" if renorm else "sin renormalizar"}')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(mrl['dimension'].unique()), [str(d) for d in sorted(mrl['dimension'].unique())])
    ax.set_ylim(0, 1.05)
    _percent_axis(ax.yaxis)
    ax.set_xlabel('Dimensiones conservadas al recortar el vector')
    ax.set_ylabel('Promedio en dev')
    ax.set_title('Sensibilidad a la dimensión del vector')
    ax.legend(loc='lower right', fontsize=8)
    return _save(fig, path)

def language_recall_plot(by_language, path, metric='recall@10'):
    """Recall@10 por idioma de la evidencia, con la pregunta original y traducida."""
    data = by_language[by_language['group'] != 'total']
    groups = list(dict.fromkeys(data['group']))
    fig, ax = _figure(7, 4)
    width = 0.38
    x = np.arange(len(groups))
    for offset, (consulta, color) in zip((-width / 2, width / 2), (('original', '#2a78d6'), ('traduccion', '#eb6834'))):
        sub = data[data['consulta'] == consulta].set_index('group').reindex(groups)
        ax.bar(x + offset, sub[metric], width=width * 0.95, color=color,
               label='pregunta original' if consulta == 'original' else 'pregunta traducida')
        for xi, value in zip(x + offset, sub[metric]):
            if pd.notna(value):
                ax.annotate(f'{value:.0%}', (xi, value), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, color=INK_2)
    counts = data[data['consulta'] == 'original'].set_index('group').reindex(groups)['n']
    ax.set_xticks(x, [f'evidencia {g}\n(n = {int(n)})' for g, n in zip(groups, counts)])
    ax.set_ylim(0, 1.1)
    _percent_axis(ax.yaxis)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel(metric)
    ax.set_title('Recuperación entre idiomas (dev)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncols=2, fontsize=8)
    return _save(fig, path)
