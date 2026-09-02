import importlib.util
from pathlib import Path
import pytest
spec = importlib.util.spec_from_file_location('c_analysis', Path(__file__).parents[1] / 'scripts/stage1_c_analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

def rows():
    return [dict(item='chair', position=v, seed=1201, condition=c, psnr=20., ssim=.8, lpips=.2) for v in range(4) for c in a.CONDITIONS]

def test_complete_and_paired():
    r=rows()
    r[0]['psnr']=21.
    means, paired=a.aggregate(r)
    assert means['correct']['psnr']==20.25
    p=next(x for x in paired if x['position']==0 and x['comparator']=='bicubic')
    assert p['psnr_margin']==1.
    assert p['all_metric_win'] is False

@pytest.mark.parametrize('change', ['duplicate','missing','metric'])
def test_reject_incomplete(change):
    r=rows()
    if change=='duplicate': r.append(dict(r[0]))
    if change=='missing': r.pop()
    if change=='metric': r[0].pop('lpips')
    with pytest.raises(ValueError): a.aggregate(r)

def test_selection_exact_and_tie():
    low=dict(step=250, dev_candidate_pass=False, decoded_quality_pass=True, correct_psnr_margin=5., correct_ssim_margin=.1, correct_lpips_reduction=.5)
    high=dict(low, step=500, dev_candidate_pass=True, correct_psnr_margin=-1.)
    assert a.select_best([low,high])==high
    assert a.select_best([dict(high,step=750),high])['step']==500
    assert a.select_best([]) is None

def test_reject_missing_whole_view():
    with pytest.raises(ValueError, match='four fixed views'):
        a.aggregate([r for r in rows() if r['position']!=3])

def test_contact_extraction_and_all_view_composition(tmp_path):
    from PIL import Image
    base=tmp_path/'eval_2000_dev'
    (base/'3d').mkdir(parents=True)
    width,height=16,16
    contact=Image.new('RGB',(210+8*width,28+4*(height+22)),'white')
    for view in range(4):
        for col in range(8):
            contact.paste(Image.new('RGB',(width,height),(view*30,col*20,50)),(210+col*width,28+view*(height+22)))
    contact.save(base/'3d/strict_seed_1201_contact.png')
    point={'metrics_source':str(base/'3d_strict_metrics.json')}
    img,_=a.source_image(point,3,'correct')
    assert img.getpixel((0,0))==(90,80,50)
    out=tmp_path/'out'; out.mkdir()
    result=a.qualitative(out,point,point,[])
    assert [r['view'] for r in result]==[0,1,2,3]
    assert all(r['crop_xyxy']==(6,6,10,10) for r in result)
    assert Image.open(out/'qualitative_all_views.png').size==(100+7*width,32+4*(height+24))

@pytest.mark.parametrize('key', ['correct_psnr','bicubic_ssim','correct_lpips','correct_psnr_margin','correct_ssim_margin','correct_lpips_reduction'])
def test_tampered_checkpoint_log_rejected(tmp_path,key):
    import json
    record=dict(step=250,dev_candidate_pass=True,correct_psnr=20.,bicubic_ssim=.8,correct_lpips=.2,
                correct_psnr_margin=0.,correct_ssim_margin=0.,correct_lpips_reduction=0.)
    record[key]+=0.01
    (tmp_path/'checkpoint_metrics.jsonl').write_text(json.dumps(record)+'\n')
    evaluation=tmp_path/'eval_0250_dev'; evaluation.mkdir()
    (evaluation/'3d_strict_metrics.json').write_text(json.dumps({'metric_rows':rows()}))
    errors=[]
    points=a.load_run(tmp_path,[],errors)
    assert points[0]['metric_validation']=='invalid'
    assert a.select_best(points) is None
    assert any(e.get('field')==key and e.get('logged')==record[key] for e in errors)

def test_valid_checkpoint_log_matches_raw(tmp_path):
    import json
    (tmp_path/'checkpoint_metrics.jsonl').write_text(json.dumps(dict(step=250, correct_psnr=20.+5e-8))+'\n')
    evaluation=tmp_path/'eval_0250_dev'; evaluation.mkdir()
    (evaluation/'3d_strict_metrics.json').write_text(json.dumps({'metric_rows':rows()}))
    errors=[]
    points=a.load_run(tmp_path,[],errors)
    assert not errors
    assert points[0]['correct_psnr']==20.
    assert points[0]['correct_psnr_margin']==0.
    assert points[0]['metric_validation']=='valid'
    assert a.select_best(points)['step']==250

def test_invalid_candidate_cannot_win():
    assert a.select_best([dict(step=250,metric_validation='invalid',dev_candidate_pass=True),
                          dict(step=500,metric_validation='valid')])['step']==500

def test_sigma_plot_uses_fixed_sigma_rows(tmp_path):
    import json
    run=tmp_path/'c_main'; run.mkdir()
    out=tmp_path/'plots'; out.mkdir()
    source=run/'sigma_metrics.jsonl'
    source.write_text(json.dumps({'step':250,'rows':[dict(sigma=s,correct_loss=.1,shuffled_loss=.2,neutral_loss=.3,disabled_loss=.4) for s in (.2,.5,.8,.95)]})+'\n')
    index=[]
    result=a.plot_sigma_losses(out,run,index)
    assert result['sigma_values']==[.2,.5,.8,.95]
    assert result['steps']==[250]
    assert index==[source]
    assert (out/'selected_fixed_sigma_losses.png').exists()

def setup_main_runs(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(a,'__file__',str(tmp_path/'scripts/stage1_c_analysis.py'))
    campaign=tmp_path/'artifacts/stage1/campaign'
    for name in ('optimized_lr_decay','optimized_sigma_balanced','c_main'):
        run=(campaign if name=='c_main' else tmp_path/'artifacts/stage1')/name
        evaluation=run/'eval_2000_dev'; evaluation.mkdir(parents=True)
        (evaluation/'3d_strict_metrics.json').write_text(json.dumps({'metric_rows':rows()}))
        (run/'checkpoint_metrics.jsonl').write_text(json.dumps({'step':2000})+'\n')
    monkeypatch.setattr(a,'plots',lambda *args:None)
    monkeypatch.setattr(a,'plot_sigma_losses',lambda *args:None)
    monkeypatch.setattr(a,'qualitative',lambda out,baseline,selected,index:[{'B':baseline['metrics_source'],'C':selected['metrics_source']}])
    return campaign

def test_main_labels_sigma_balanced_as_b_not_lr_decay(tmp_path,monkeypatch):
    import json
    import sys
    campaign=setup_main_runs(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['analysis','--campaign',str(campaign)])
    a.main()
    evidence=json.loads((campaign/'analysis/machine_evidence.json').read_text())
    assert '/optimized_sigma_balanced/' in evidence['qualitative'][0]['B']
    assert '/optimized_lr_decay/' not in evidence['qualitative'][0]['B']
    assert '/c_main/' in evidence['qualitative'][0]['C']

def test_main_refuses_historical_run_labeled_c(tmp_path,monkeypatch):
    import sys
    campaign=setup_main_runs(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['analysis','--campaign',str(campaign),'--selected-run','optimized_lr_decay'])
    with pytest.raises(ValueError,match='C run'):
        a.main()

def test_final_contact_includes_all_seed_view_pairs(tmp_path):
    from PIL import Image
    source=tmp_path/'final/3d_strict_metrics.json'
    images=source.parent/'images'; images.mkdir(parents=True)
    for view in range(4):
        for condition in ('hr','bicubic'):
            Image.new('RGB',(16,16),(view*40,0,0)).save(images/f'view_{view}_{condition}.png')
        for seed in (2201,2202,2203,2204):
            for condition in ('correct','shuffled','disabled'):
                Image.new('RGB',(16,16),(view*40,seed-2200,0)).save(images/f'seed_{seed}_view_{view}_{condition}.png')
    final={'source':str(source),'noise_seeds':[2201,2202,2203,2204]}
    out=tmp_path/'out'; out.mkdir()
    evidence=a.final_qualitative(out,final,[])
    assert len(evidence)==16
    assert {(r['seed'],r['view']) for r in evidence}=={(seed,view) for seed in (2201,2202,2203,2204) for view in range(4)}
    assert all(r['crop_xyxy']==(6,6,10,10) for r in evidence)
    assert Image.open(out/'final_all_seeds_views.png').size==(140+5*16,32+16*(16+24))

def test_failed_final_never_reports_pass(tmp_path,monkeypatch):
    import json
    import sys
    campaign=setup_main_runs(tmp_path,monkeypatch)
    final_dir=campaign/'final'; final_dir.mkdir()
    final_rows=[dict(r,seed=seed) for seed in (2201,2202,2203,2204) for r in rows()]
    (final_dir/'3d_strict_metrics.json').write_text(json.dumps({'metric_rows':final_rows,'verdict':{'passed':False}}))
    monkeypatch.setattr(a,'final_qualitative',lambda *args:[])
    monkeypatch.setattr(sys,'argv',['analysis','--campaign',str(campaign)])
    a.main()
    evidence=json.loads((campaign/'analysis/machine_evidence.json').read_text())
    assert evidence['stage1_pass'] is False
    assert evidence['final_status']=='FAIL'
