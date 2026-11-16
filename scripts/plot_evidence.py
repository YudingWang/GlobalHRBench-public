"""Render Figure 5 from the released fourteen-model cell scores."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'results/broad/summary.json').read_text())
cells=summary['evidence_cells'];names={r['model_id']:r['display_name'] for r in cells}
ordered=sorted(names,key=lambda m:(-next(x['metrics']['correct']['question'] for x in cells if x['model_id']==m and x['condition']=='oracle'),names[m]))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.2,6.7));y=np.arange(len(ordered))
for j,(condition,label,color) in enumerate([('closed','Closed book','#9da9b6'),('rag','BM25','#0072B2'),('oracle','Designated evidence','#7651A8')]):
    values=[100*next(x['metrics']['correct']['question'] for x in cells if (x['model_id'],x['condition'])==(m,condition)) for m in ordered]
    ax.barh(y+(j-1)*.24,values,height=.21,color=color,label=label)
    for yi,v in zip(y+(j-1)*.24,values):ax.text(v+1,yi,f'{v:.1f}',va='center',fontsize=7.5)
ax.set_yticks(y,[names[m] for m in ordered]);ax.invert_yaxis();ax.set_xlim(0,106);ax.set_xticks(range(0,101,20));ax.set_xlabel('Target-answer score (%)');ax.legend(loc='lower right',fontsize=8,frameon=False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout()
fig.savefig(ROOT/'paper/source/assets/evidence_models.pdf');plt.close(fig)
