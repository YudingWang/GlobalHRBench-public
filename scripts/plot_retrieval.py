"""Render Figure 4 from released scores without changing observations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'results/broad/summary.json').read_text())['retrieval_ordered']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.2,4.6))
y=np.arange(len(rows));blue='#0072B2';orange='#D55E00'
for yi,r in zip(y,rows):
    left,right=100*r['left'],100*r['right']
    ax.plot([left,right],[yi,yi],color='#c3c9cf',lw=2,zorder=1)
    # Place numbers outside the segment so short gains remain readable.
    if left != right:
        ax.annotate(f'{left:.1f}',(left,yi),xytext=(-6 if left<right else 6,0),textcoords='offset points',ha='right' if left<right else 'left',va='center',fontsize=8,color=blue)
        ax.annotate(f'{right:.1f}',(right,yi),xytext=(6 if right>left else -6,0),textcoords='offset points',ha='left' if right>left else 'right',va='center',fontsize=8,color=orange)
    else:
        ax.annotate(f'{left:.1f}',(left,yi),xytext=(-6,0),textcoords='offset points',ha='right',va='center',fontsize=8,color=blue)
        ax.annotate(f'{right:.1f}',(right,yi),xytext=(6,0),textcoords='offset points',ha='left',va='center',fontsize=8,color=orange)

ax.scatter([100*r['left'] for r in rows],y,color=blue,marker='o',label='BM25',s=27,zorder=3)
ax.scatter([100*r['right'] for r in rows],y,color=orange,marker='D',label='Public-country route',s=26,zorder=4)
labels=[f"{r['score_tied_position']}{'=' if sum(x['right']==r['right'] for x in rows)>1 else ''}  {r['display_name']}" for r in rows]
ax.set_yticks(y,labels);ax.set_ylim(len(rows)-.35,-.65);ax.set_xlim(-3,100);ax.set_xticks(range(0,101,20));ax.set_xlabel('Target-answer score (%)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='lower right',fontsize=8,frameon=False)
fig.tight_layout()
fig.savefig(ROOT/'paper/source/assets/retrieval_models.pdf')
plt.close(fig)
