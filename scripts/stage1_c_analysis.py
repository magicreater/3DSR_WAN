
"""CPU-only, provenance-preserving Stage 1 C analysis; never certifies visual quality."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ('psnr', 'ssim', 'lpips')
CONDITIONS = ('correct', 'shuffled', 'neutral', 'disabled', 'bicubic', 'vae_ceiling')
HISTORICAL_RUNS = {'A': 'optimized_lr_decay', 'B': 'optimized_sigma_balanced'}
C_RUNS = ('c_main','c_multilayer','c_time','c_sigma','c_depth')

def aggregate(rows):
    """Require a complete six-condition paired grid and finite three-metric rows."""
    grid = {}
    for r in rows:
        key = (r.get('item'), r.get('seed'), r.get('position'), r.get('condition'))
        if any(x is None for x in key) or key[-1] not in CONDITIONS:
            raise ValueError('missing identity or unknown condition: '+str(key))
        if key in grid:
            raise ValueError('duplicate metric row: '+str(key))
        if any(not isinstance(r.get(m), (int,float)) or not math.isfinite(r[m]) for m in METRICS):
            raise ValueError('missing/nonfinite metric: '+str(key))
        grid[key] = r
    identities = sorted({k[:3] for k in grid})
    if not identities:
        raise ValueError('no metric rows')
    for identity in identities:
        if any((*identity,c) not in grid for c in CONDITIONS):
            raise ValueError('missing paired condition: '+str(identity))
    # Every seed must cover the identical item/view population.
    populations = {}
    for item, seed, pos in identities:
        populations.setdefault(seed,set()).add((item,pos))
    if any(p != next(iter(populations.values())) for p in populations.values()):
        raise ValueError('incomplete seed/view population')
    for item, seed, _ in identities:
        if {pos for it,se,pos in identities if it==item and se==seed} != {0,1,2,3}:
            raise ValueError('expected all four fixed views for each item/seed')
    means = {c:{m:statistics.mean(grid[(*k,c)][m] for k in identities) for m in METRICS} for c in CONDITIONS}
    paired = []
    for k in identities:
        r = grid[(*k,'correct')]
        for c in CONDITIONS[1:]:
            ref = grid[(*k,c)]
            margins = {m+'_margin':(r[m]-ref[m])*(1 if m!='lpips' else -1) for m in METRICS}
            paired.append(dict(item=k[0],seed=k[1],position=k[2],comparator=c,
                               **{m:r[m] for m in METRICS}, **{c+'_'+m:ref[m] for m in METRICS},
                               **margins, lpips_relative_reduction=(ref['lpips']-r['lpips'])/ref['lpips'] if ref['lpips'] else None,
                               all_metric_win=all(v>0 for v in margins.values())))
    return means, paired

def selection_key(r):
    # Exactly the experiment/reporting comparator; ties prefer earliest checkpoint.
    return (bool(r.get('dev_candidate_pass')), bool(r.get('decoded_quality_pass')),
            r.get('correct_psnr_margin',-1e9), r.get('correct_ssim_margin',-1e9),
            r.get('correct_lpips_reduction',-1e9), -r['step'])

def select_best(rows):
    eligible = [r for r in rows if r.get('metric_validation') not in ('invalid','missing')]
    return max(eligible,key=selection_key) if eligible else None

def read_jsonl(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def dump(path, payload):
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r})
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def numeric_flat(row, prefix=''):
    out={}
    for k,v in row.items():
        key=prefix+k
        if isinstance(v,dict): out.update(numeric_flat(v,key+'.'))
        elif isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v): out[key]=v
    return out

def train_evidence(path):
    rows=read_jsonl(path)
    if not rows: return {'source':str(path),'status':'missing'}
    flat=[numeric_flat(r) for r in rows]
    diagnostics={}
    for k in sorted({k for r in flat for k in r if 'gate' in k or 'residual' in k}):
        values=[r[k] for r in flat if k in r]
        diagnostics[k]={'count':len(values),'first':values[0],'last':values[-1],
                        'min':min(values),'max':max(values),'mean':statistics.mean(values)}
    seconds=[r['step_seconds'] for r in rows if isinstance(r.get('step_seconds'),(int,float))]
    elapsed=[r['elapsed_seconds'] for r in rows if isinstance(r.get('elapsed_seconds'),(int,float))]
    peak=[r['peak_gpu_memory_mib'] for r in rows if isinstance(r.get('peak_gpu_memory_mib'),(int,float))]
    return {'source':str(path),'logged_steps':len(rows),'last_step':rows[-1].get('step'),
            'summed_step_seconds':sum(seconds) if seconds else None,
            'last_elapsed_seconds':elapsed[-1] if elapsed else None,
            'throughput_steps_per_second':len(seconds)/sum(seconds) if seconds and sum(seconds)>0 else None,
            'peak_gpu_memory_mib':max(peak) if peak else None,'diagnostics':diagnostics,
            'gate_saturation':{'status':'bounded (0,2); min/max do not determine saturation fraction; unknown unless fraction explicitly logged'},
            'duration_note':'step sum excludes evaluation; elapsed semantics come from training logger; no wall duration inferred'}

def load_run(path, index, errors):
    logs=read_jsonl(path/'checkpoint_metrics.jsonl')
    seen=set()
    for r in logs:
        if r['step'] in seen: raise ValueError('duplicate checkpoint step '+str(path))
        seen.add(r['step'])
    points=[]
    for r in sorted(logs,key=lambda x:x['step']):
        p=dict(r,run=path.name,run_path=str(path))
        source=path/f"eval_{r['step']:04d}_dev"/'3d_strict_metrics.json'
        p['metrics_source']=str(source)
        if source.exists():
            data=json.loads(source.read_text())
            try:
                means, paired=aggregate(data['metric_rows'])
                recomputed={c+'_'+m:v for c,values in means.items() for m,v in values.items()}
                recomputed.update(correct_psnr_margin=means['correct']['psnr']-means['bicubic']['psnr'],
                                  correct_ssim_margin=means['correct']['ssim']-means['bicubic']['ssim'],
                                  correct_lpips_reduction=(means['bicubic']['lpips']-means['correct']['lpips'])/means['bicubic']['lpips'] if means['bicubic']['lpips'] else None)
                mismatches=[]
                for key,value in recomputed.items():
                    if key not in r: continue
                    logged=r[key]
                    same = (logged is None and value is None) or (
                        isinstance(logged,(int,float)) and not isinstance(logged,bool)
                        and isinstance(value,(int,float)) and math.isfinite(logged)
                        and abs(logged-value)<=1e-7)
                    if not same:
                        mismatch={'source':str(source),'log_source':str(path/'checkpoint_metrics.jsonl'),
                                  'step':r['step'],'error':'logged metric does not reproduce from raw rows',
                                  'field':key,'logged':logged,'recomputed':value,'absolute_tolerance':1e-7}
                        errors.append(mismatch); mismatches.append(mismatch)
                if mismatches:
                    p['metric_validation']='invalid'
                    p['metric_mismatches']=mismatches
                else:
                    p['metric_validation']='valid'
                    p['means']=means; p['paired']=paired; p['rows']=data['metric_rows']
                    p.update(recomputed)
            except (ValueError,KeyError) as e:
                errors.append({'source':str(source),'error':str(e)})
                p['metric_validation']='invalid'
            index.append(source)
        else: p['metric_validation']='missing'
        points.append(p)
    for source in (path/'checkpoint_metrics.jsonl',path/'train_steps.jsonl',path/'checkpoint_index.json',path/'training_config.json',path/'run_config.json',path/'manifest.json',path/'run_manifest.json',path/'3d_result.json',path/'launches.jsonl',path/'sigma_metrics.jsonl'):
        if source.exists(): index.append(source)
    return points

def compact(point):
    if point is None: return None
    return {k:v for k,v in point.items() if k not in ('paired','rows','means')}

def plots(out,runs,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    for name,points in runs.items():
        for ax,m in zip(axes,METRICS):
            valid=[p for p in points if p.get('metric_validation')=='valid' and isinstance(p.get('correct_'+m),(float,int))]
            ax.plot([p['step'] for p in valid],[p['correct_'+m] for p in valid],'.-',label=name,markersize=5)
            best=select_best(valid)
            if best: ax.scatter(best['step'],best['correct_'+m],marker='*',s=140)
    reference=next((p for pts in runs.values() for p in pts if 'means' in p),None)
    if reference:
        for ax,m in zip(axes,METRICS):
            for c,style in [('bicubic','--'),('vae_ceiling',':')]:
                ax.axhline(reference[c+'_'+m],ls=style,color='gray',label=c)
            b=reference['bicubic_'+m]
            ax.axhline(b*.95 if m=='lpips' else b+(.25 if m=='psnr' else .005),color='black',ls='-.',label='bicubic gate')
    for ax,m in zip(axes,METRICS):
        ax.set_title(m.upper()); ax.set_xlabel('Training step'); ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    fig.savefig(out/'across_experiment_curves.png',dpi=160); plt.close(fig)
    bests=[best for pts in runs.values() if (best:=select_best(pts)) is not None]
    if bests:
        fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
        for ax,m in zip(axes,METRICS):
            vals=[p.get('correct_'+m,float('nan')) for p in bests]
            ax.bar([p['run']+'@'+str(p['step']) for p in bests],vals)
            ax.tick_params(axis='x',labelrotation=25); ax.set_title(m.upper())
        fig.savefig(out/'best_metrics.png',dpi=160); plt.close(fig)
    if selected and selected.get('rows'):
        rows=[r for r in selected['rows'] if r['condition']=='correct']
        fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
        for ax,m in zip(axes,METRICS):
            values=[r[m] for r in rows]
            ax.scatter(range(len(rows)),values)
            worst=max(values) if m=='lpips' else min(values)
            ax.axhline(worst,color='red',ls=':',label=f'worst={worst:.5f}')
            ax.axhline(statistics.mean(values),color='gray',label='mean')
            ax.set_xticks(range(len(rows)),[f"{r['seed']}/v{r['position']}" for r in rows],rotation=45)
            ax.set_title(m.upper()); ax.legend()
        fig.savefig(out/'selected_distribution.png',dpi=160); plt.close(fig)


def plot_sigma_losses(out,run,index):
    """Plot fixed-noise one-step validation, explicitly separate from decoded rollout."""
    source=Path(run)/'sigma_metrics.jsonl'
    records=read_jsonl(source)
    if not records: return {'source':str(source),'status':'missing'}
    index.append(source)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sigma_values=[.2,.5,.8,.95]
    steps=[r['step'] for r in records]
    if len(set(steps))!=len(steps): raise ValueError('duplicate fixed-sigma validation step')
    records=sorted(records,key=lambda r:r['step'])
    steps=[r['step'] for r in records]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    measured=[]
    for sigma,axis in zip(sigma_values,axes.flat):
        for condition in ('correct','shuffled','neutral','disabled'):
            values=[]
            for record in records:
                matching=[r for r in record.get('rows',[]) if abs(r.get('sigma',-1)-sigma)<1e-8]
                raw=[r.get(condition+'_loss') for r in matching]
                value=statistics.mean(raw) if raw and all(isinstance(v,(int,float)) and math.isfinite(v) for v in raw) else None
                values.append(float('nan') if value is None else value)
                measured.append({'step':record['step'],'sigma':sigma,'condition':condition,'flow_loss':value,'row_count':len(raw)})
            axis.plot(steps,values,'.-',label=condition)
        axis.set_title('Fixed sigma = '+str(sigma)); axis.set_xlabel('Training step')
        axis.set_ylabel('Single-step flow MSE'); axis.grid(alpha=.2); axis.legend()
    fig.suptitle(Path(run).name+': fixed-validation flow loss (not decoded quality)')
    fig.savefig(out/'selected_fixed_sigma_losses.png',dpi=160); plt.close(fig)
    write_csv(out/'selected_fixed_sigma_losses.csv',measured)
    return {'source':str(source),'status':'available','sigma_values':sigma_values,'steps':steps,
            'measurement':'Mean across logged item rows per fixed sigma; missing losses stay unknown; fixed validation noise',
            'limitation':'Single-step flow loss does not establish multi-step decoded rollout quality.'}

def source_image(point,view,condition,seed=1201):
    from PIL import Image
    base=Path(point['metrics_source']).parent
    filename=f'view_{view}_{condition}.png' if condition in ('hr','lr','bicubic','vae_ceiling') else f'seed_{seed}_view_{view}_{condition}.png'
    p=base/'images'/filename
    if p.exists(): return Image.open(p).convert('RGB'),p
    p=base/'3d'/f'strict_seed_{seed}_contact.png'
    if not p.exists(): raise FileNotFoundError(str(p))
    img=Image.open(p).convert('RGB')
    w=(img.width-210)//8
    h=(img.height-28)//4-22
    if img.width!=210+8*w or img.height!=28+4*(h+22): raise ValueError('unexpected contact geometry')
    col=['hr','lr','bicubic','vae_ceiling','correct','shuffled','neutral','disabled'].index(condition)
    return img.crop((210+col*w,28+view*(h+22),210+(col+1)*w,28+view*(h+22)+h)),p

def qualitative(out,baseline,selected,index):
    from PIL import Image, ImageDraw
    import numpy as np
    rows=[]; crops=[]; errors=[]; provenance=[]
    for view in range(4):
        images=[]
        for point,c in [(selected,'hr'),(selected,'lr'),(selected,'bicubic'),(baseline,'correct'),(selected,'correct'),(selected,'shuffled'),(selected,'disabled')]:
            img,source=source_image(point,view,c); index.append(source); images.append(img)
        size=images[0].size
        images=[im.resize(size,Image.Resampling.NEAREST) if im.size!=size else im for im in images]
        # Reject comparisons with a different HR, view order, or baseline resolution.
        b_hr,b_source=source_image(baseline,view,'hr'); index.append(b_source)
        if b_hr.size!=size or not np.array_equal(np.asarray(b_hr),np.asarray(images[0])):
            raise ValueError('B/C HR mismatch at view '+str(view))
        w,h=size; box=(3*w//8,3*h//8,5*w//8,5*h//8)
        rows.append(images)
        crops.append([im.crop(box).resize(size,Image.Resampling.NEAREST) for im in images])
        error_row=[]
        for im in (images[3],images[4]):
            err=np.abs(np.asarray(im,dtype=float)/255-np.asarray(images[0],dtype=float)/255)
            error_row.append(Image.fromarray((np.clip(err/.25,0,1)*255).astype('uint8')))
        errors.append(error_row)
        provenance.append({'view':view,'crop_xyxy':box,'error':'absolute RGB difference per channel / 0.25, clipped [0,1]','B':str(baseline['metrics_source']),'C':str(selected['metrics_source'])})
    def sheet(path,content,labels):
        w,h=content[0][0].size
        canvas=Image.new('RGB',(100+w*len(labels),32+(h+24)*len(content)),'white')
        draw=ImageDraw.Draw(canvas)
        for j,label in enumerate(labels): draw.text((100+j*w+5,8),label,fill='black')
        for i,row in enumerate(content):
            y=32+i*(h+24); draw.text((5,y+5),'view '+str(i),fill='black')
            for j,img in enumerate(row): canvas.paste(img,(100+j*w,y))
        canvas.save(path)
    labels=['HR','LR nearest','Bicubic','B correct','C correct','C shuffled','C disabled']
    sheet(out/'qualitative_all_views.png',rows,labels)
    sheet(out/'qualitative_fixed_center_crops.png',crops,labels)
    sheet(out/'qualitative_absrgb_error_0_025.png',errors,['B absRGB [0,.25]','C absRGB [0,.25]'])
    return provenance


def final_qualitative(out,final,index):
    """Compose every fixed final noise seed/view without selecting favorable examples."""
    from PIL import Image, ImageDraw
    seeds=final['noise_seeds']
    if seeds != [2201,2202,2203,2204]:
        raise ValueError('final gallery requires fixed seeds 2201-2204')
    rows=[]; provenance=[]
    point={'metrics_source':final['source']}
    for seed in seeds:
        for view in range(4):
            images=[]; sources=[]
            for condition in ('hr','bicubic','correct','shuffled','disabled'):
                img,source=source_image(point,view,condition,seed)
                index.append(source); sources.append(str(source)); images.append(img)
            size=images[0].size
            if any(img.size!=size for img in images):
                raise ValueError('final image size mismatch')
            w,h=size; box=(3*w//8,3*h//8,5*w//8,5*h//8)
            rows.append((seed,view,images,box))
            provenance.append({'seed':seed,'view':view,'crop_xyxy':box,'sources':sources,
                               'columns':['HR','Bicubic','C correct','C shuffled','C disabled']})
    w,h=rows[0][2][0].size
    for crop,filename in ((False,'final_all_seeds_views.png'),(True,'final_all_seeds_fixed_center_crops.png')):
        canvas=Image.new('RGB',(140+5*w,32+len(rows)*(h+24)),'white')
        draw=ImageDraw.Draw(canvas)
        for j,label in enumerate(provenance[0]['columns']):
            draw.text((140+j*w+5,8),label,fill='black')
        for i,(seed,view,images,box) in enumerate(rows):
            y=32+i*(h+24)
            draw.text((5,y+5),f'seed {seed} / view {view}',fill='black')
            for j,img in enumerate(images):
                if crop: img=img.crop(box).resize((w,h),Image.Resampling.NEAREST)
                canvas.paste(img,(140+j*w,y))
        canvas.save(out/filename)
    return provenance

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--selected-run',type=Path)
    args=parser.parse_args()
    campaign=args.campaign.resolve(); out=(args.output_dir or campaign/'analysis').resolve(); out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    paths=[root/'artifacts/stage1'/name for name in HISTORICAL_RUNS.values()]
    paths += [campaign/name for name in C_RUNS if (campaign/name).exists()]
    index=[]; errors=[]; runs={}; summaries=[]; allpaired=[]
    for path in paths:
        if not path.exists(): continue
        points=load_run(path,index,errors); runs[path.name]=points
        best=select_best(points); last=points[-1] if points else None
        matched=next((p for p in points if p['step']==2000),None)
        bounded=[p for p in points if p['step']<=2000]
        summary={'run':path.name,'best':compact(best),'last':compact(last),'exact_step2000':compact(matched),
                 'best_up_to2000':compact(select_best(bounded)),
                 'available_steps_up_to2000':[p['step'] for p in bounded],'training':train_evidence(path/'train_steps.jsonl'),
                 'all_checkpoints':[compact(p) for p in points]}
        summaries.append(summary)
        for p in points:
            allpaired += [dict(run=path.name,step=p['step'],source=p['metrics_source'],**r) for r in p.get('paired',[])]
    candidates=[p for n,pts in runs.items() if n in C_RUNS for p in pts]
    if args.selected_run:
        if args.selected_run.name not in C_RUNS:
            raise ValueError('selected-run must identify a C run, not a historical baseline')
        selected=select_best(runs.get(args.selected_run.name,[]))
        if selected is None: raise ValueError('selected run has no checkpoints')
    else: selected=select_best(candidates)
    final=None
    for source in [campaign/'final/3d_strict_metrics.json',campaign/'final/decoded_metrics.json']:
        if source.exists():
            index.append(source); data=json.loads(source.read_text())
            try:
                means,paired=aggregate(data['metric_rows'])
                final={'source':str(source),'means':means,'paired':paired,'strict_verdict':data.get('strict_verdict',data.get('verdict')),
                       'noise_seeds':sorted({r['seed'] for r in data['metric_rows']}),
                       'distribution':{m:{'min':min(r[m] for r in data['metric_rows'] if r['condition']=='correct'),
                                          'max':max(r[m] for r in data['metric_rows'] if r['condition']=='correct'),
                                          'worst':(max if m=='lpips' else min)(r[m] for r in data['metric_rows'] if r['condition']=='correct')}
                                       for m in METRICS}}
                allpaired += [dict(run='final',step=None,source=str(source),**r) for r in paired]
            except (KeyError,ValueError) as e: errors.append({'source':str(source),'error':str(e)})
            break
    plots(out,runs,selected)
    final_gallery=None
    if final:
        final_plot_dir=out/'final'; final_plot_dir.mkdir(exist_ok=True)
        plots(final_plot_dir,{},dict(rows=data['metric_rows']))
        try: final_gallery=final_qualitative(out,final,index)
        except (FileNotFoundError,ValueError) as e: errors.append({'final_qualitative':str(e)})
    sigma_info=None
    if selected:
        try: sigma_info=plot_sigma_losses(out,Path(selected['run_path']),index)
        except ValueError as e: errors.append({'fixed_sigma':str(e)})
    qualitative_info=None
    baseline=next((p for p in runs.get(HISTORICAL_RUNS['B'],[]) if p['step']==2000 and p.get('metric_validation')=='valid'),None)
    if selected and baseline:
        try: qualitative_info=qualitative(out,baseline,selected,index)
        except (FileNotFoundError,ValueError) as e: errors.append({'qualitative':str(e)})
    dump(out/'experiment_summary.json',summaries)
    table=[]
    for summary in summaries:
        for scope in ('best','last','exact_step2000','best_up_to2000'):
            point=summary[scope]
            table.append(dict(run=summary['run'],comparison_scope=scope,status=point.get('metric_validation','available') if point else 'unknown',
                              **{k:v for k,v in (point or {}).items() if not isinstance(v,(dict,list)) and k!='run'}))
    write_csv(out/'experiment_summary.csv',table)
    write_csv(out/'per_seed_view_metrics.csv',allpaired)
    failed_final=bool(final and (final.get('strict_verdict') or {}).get('passed') is False)
    evidence={'selection':compact(selected),'selection_rule':'lexicographic dev_candidate_pass, decoded_quality_pass, PSNR margin, SSIM margin, LPIPS relative reduction; earliest step breaks tie',
              'comparison_rule':'best and exact step 2000 are separate; no extrapolation or fill of missing measurements',
              'historical_label_mapping':HISTORICAL_RUNS,'qualitative_baseline':compact(baseline),
              'final':final,'qualitative':qualitative_info,'fixed_sigma_validation':sigma_info,'errors':errors,
              'stage1_pass':False if failed_final else None,'final_status':'FAIL' if failed_final else 'UNVERIFIED',
              'final_qualitative':final_gallery,'visual_review':'external human review required; this utility cannot certify PASS',
              'limitations':['Fixed seeds 2201-2204 are noise replicates of the same scene/views, not independent scenes.',
                             'Development seed 1201 is used for checkpoint selection; development metrics are not held-out evidence.',
                             'PNG error maps are quantized visualization; raw metric JSON remains authoritative.']}
    dump(out/'machine_evidence.json',evidence)
    index.extend(p for p in (campaign/'source_before_c_sha256.json',campaign/'campaign.json') if p.exists())
    unique=sorted(set(index))
    dump(out/'report_data_index.json',{'sources':[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in unique if p.exists()],
                                      'source_before_c':str(campaign/'source_before_c.tar.gz'),'generated_files':sorted(p.name for p in out.iterdir())})
    text=['# 实验 C 测量结果分析','',
          '最佳检查点与第 2000 步同预算比较单独列示；缺失测量保留 unknown。',
          '所选检查点：'+(selected['run']+' step '+str(selected['step']) if selected else '尚无已执行的 C 检查点。'),
          '历史标签：A = optimized_lr_decay；B = optimized_sigma_balanced。定性图使用 B 第 2000 步。',
          '定性图包含全部四个视角，采用一致中心裁剪与绝对 RGB 误差范围 [0,0.25]。',
          '固定种子 2201–2204 仅表示同一场景的噪声重复，不是独立场景；尚未验证泛化。',
          '最终验收：FAIL。严格指标未通过；停止调参。' if failed_final else '本报告不认证 Stage 1 PASS；仍须结合最终严格指标和外部视觉审查。',
          '数据校验问题数：'+str(len(errors)), '',
          '|run|best step|PSNR|SSIM|LPIPS|exact step 2000|','|---|---:|---:|---:|---:|---|']
    for s in summaries:
        b=s['best'] or {}
        text.append('|'+s['run']+'|'+str(b.get('step','unknown'))+'|'+ '|'.join(str(b.get('correct_'+m,'unknown')) for m in METRICS)+'|'+('available' if s['exact_step2000'] else 'unknown')+'|')
    (out/'analysis.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'output_dir':str(out),'runs':len(runs),'selected':compact(selected),'issues':errors},indent=2))

if __name__=='__main__': main()

