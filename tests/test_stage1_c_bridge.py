import copy
import pytest
import torch
from torch import nn
from rl3dsr.models.wan.dit import WanDiT
from rl3dsr.models.wan.lq_conditioning import FrozenLQConditioner, load_adapter_checkpoint, save_adapter_checkpoint


def conditioner(blocks=(0, 2), time=True):
    return FrozenLQConditioner(nn.Linear(3, 4), feature_dim=4,
                               bridge_blocks=blocks, bridge_time_conditioning=time)


def test_initial_noop_gradient_and_time_gate_after_projection_update():
    c = conditioner()
    f = torch.ones(2, 3, 4)
    t = torch.tensor([100., 900.])
    r = c.bridge_residuals(f, t)
    assert set(r) == {0, 2}
    assert all(torch.count_nonzero(x) == 0 for x in r.values())
    sum(x.sum() for x in r.values()).backward()
    for p in c.bridge.projections.values():
        assert p.weight.grad.abs().sum() > 0
    for g in c.bridge.time_gates.values():
        assert torch.count_nonzero(g.weight.grad) == 0
    with torch.no_grad():
        for p in c.bridge.projections.values():
            p.weight.add_(-0.01 * p.weight.grad)
    c.zero_grad()
    sum(x.sum() for x in c.bridge_residuals(f, t).values()).backward()
    assert all(g.weight.grad.abs().sum() > 0 for g in c.bridge.time_gates.values())
    with torch.no_grad():
        for g in c.bridge.time_gates.values():
            g.weight.normal_(std=.1)
    assert not torch.equal(c.bridge_residuals(f, t)[0][0], c.bridge_residuals(f, t)[0][1])
    assert c.bridge_residuals(f, t, enabled=False) == {}
    assert all(p.grad is None and not p.requires_grad for p in c.projector.parameters())
    with pytest.raises(ValueError, match='bridge_residuals'):
        c.bridge_tokens(f)
    c.reset_bridge()
    assert all(torch.count_nonzero(x) == 0 for x in c.bridge_residuals(f, t).values())


class Block(nn.Module):
    def forward(self, x):
        return x * 2


class TinyWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.dim = 4
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.fail = False

    def forward(self, samples, timesteps, contexts, seq_len):
        x = torch.zeros(len(samples), seq_len, self.dim)
        for block in self.blocks:
            x = block(x)
        if self.fail:
            raise RuntimeError('fake failure')
        return [torch.ones_like(s) * x[i].mean() for i, s in enumerate(samples)]


def test_multiblock_distinct_binding_gradient_and_cleanup(monkeypatch):
    model = TinyWan()
    d = WanDiT(model, device='cpu')
    z, t = torch.zeros(1,16,1,2,2), torch.tensor([500.])
    a = torch.ones(1,1,4, requires_grad=True)
    b = torch.full((1,1,4), 3., requires_grad=True)
    y = d(z,t,block_token_residuals={0:a, 2:b})
    assert torch.equal(y, torch.full_like(y,14.))
    y.mean().backward()
    assert torch.all(a.grad == 2.) and torch.all(b.grad == .5)
    assert all(not m._forward_pre_hooks for m in model.blocks)
    model.fail = True
    with pytest.raises(RuntimeError, match='fake failure'):
        d(z,t,block_token_residuals={0:a, 2:b})
    assert all(not m._forward_pre_hooks for m in model.blocks)
    model.fail = False
    def bad_register(*args, **kwargs):
        raise RuntimeError('registration failure')
    monkeypatch.setattr(model.blocks[2], 'register_forward_pre_hook', bad_register)
    with pytest.raises(RuntimeError, match='registration failure'):
        d(z,t,block_token_residuals={0:a, 2:b})
    assert all(not m._forward_pre_hooks for m in model.blocks)


@pytest.mark.parametrize('mapping', [{True:torch.zeros(1,1,4)}, {-1:torch.zeros(1,1,4)}, {3:torch.zeros(1,1,4)}, {0:torch.zeros(1,2,4)}, {0:torch.full((1,1,4),float('nan'))}])
def test_bad_injection_rejected(mapping):
    d = WanDiT(TinyWan(),device='cpu')
    with pytest.raises(ValueError):
        d(torch.zeros(1,16,1,2,2),torch.tensor([500.]),block_token_residuals=mapping)
    assert all(not m._forward_pre_hooks for m in d.model.blocks)


def test_mutually_exclusive_and_shared_prediction_helper():
    from rl3dsr.models.wan.lq_conditioning import conditioned_prediction
    d = WanDiT(TinyWan(),device='cpu')
    z,t = torch.zeros(1,16,1,2,2),torch.tensor([500.])
    with pytest.raises(ValueError,match='mutually exclusive'):
        d(z,t,token_residual=torch.zeros(1,1,4),block_token_residuals={})
    c = conditioner()
    assert torch.equal(conditioned_prediction(d,c,z,t,None,None), d(z,t))
    assert torch.equal(conditioned_prediction(d,c,z,t,None,torch.ones(1,1,4)),d(z,t))
    with torch.no_grad():
        c.bridge.projections['0'].bias.fill_(1.)
        c.bridge.projections['2'].bias.fill_(3.)
    assert torch.equal(conditioned_prediction(d,c,z,t,None,torch.ones(1,1,4)),torch.full_like(z,14.))
    assert torch.equal(conditioned_prediction(d,c,z,t,None,None),d(z,t))


def test_bridge_fp32_under_autocast_and_gate_embedding_wan_units():
    c = conditioner()
    t = torch.tensor([250.,750.])
    features = torch.ones(2,1,4,dtype=torch.bfloat16)
    with torch.no_grad():
        c.bridge.projections['0'].bias.fill_(1.)
        c.bridge.time_gates['0'].weight[:,0].fill_(1.)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        result = c.bridge_residuals(features,t)[0]
    # The first cosine component has unit frequency; sigma instead of Wan t
    # and sin/cos ordering mistakes would both violate this reference.
    expected = 1 + torch.tanh(torch.nn.functional.silu(torch.cos(t.double()).float()))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result[:,0,0],expected)
    stats = c.bridge.last_diagnostics[0]
    assert all(not v.requires_grad for v in stats.values())
    assert stats['gate_min'] >= 0 and stats['gate_max'] <= 2


@pytest.mark.parametrize('blocks', [(), (0,0), (-1,), (True,), (0.5,)])
def test_bridge_rejects_invalid_block_configuration(blocks):
    with pytest.raises(ValueError):
        conditioner(blocks)


def test_legacy_and_v2_checkpoint_roundtrip_and_architecture_mismatch(tmp_path):
    legacy = conditioner((0,),False)
    assert isinstance(legacy.bridge, nn.Linear)
    assert set(legacy.bridge.state_dict()) == {'weight','bias'}
    path = tmp_path/'v1.pt'
    torch.save({'format_version':1,'bridge':legacy.bridge.state_dict(),'config':{'scale':4},'experiment':{}},path)
    load_adapter_checkpoint(path,legacy,expected_config={'scale':4,'bridge_blocks':[0],'bridge_time_conditioning':False,'stop_on_dev_pass':False})
    with pytest.raises(RuntimeError,match='config mismatch'):
        load_adapter_checkpoint(path,legacy,expected_config={'scale':4,'stop_on_dev_pass':True})
    with pytest.raises(RuntimeError,match='architecture'):
        load_adapter_checkpoint(path,conditioner())
    c = conditioner()
    with torch.no_grad():
        for p in c.bridge.parameters():
            p.normal_()
    state = copy.deepcopy(c.bridge.state_dict())
    save_adapter_checkpoint(path,c,config={'scale':4,'bridge_blocks':[0,2],'bridge_time_conditioning':True},experiment={})
    assert torch.load(path,weights_only=True)['format_version'] == 2
    c.reset_bridge()
    load_adapter_checkpoint(path,c,expected_config={'scale':4,'bridge_blocks':(0,2),'bridge_time_conditioning':True})
    assert all(torch.equal(v,c.bridge.state_dict()[k]) for k,v in state.items())
    with pytest.raises(RuntimeError,match='architecture'):
        load_adapter_checkpoint(path,conditioner((0,1),True))
    payload = torch.load(path,weights_only=True)
    del payload['bridge'][next(iter(payload['bridge']))]
    torch.save(payload,path)
    with pytest.raises(RuntimeError):
        load_adapter_checkpoint(path,c)


def test_legacy_cached_features_follow_bridge_device_and_dtype():
    c = conditioner((0,),False)
    features = torch.ones(1,2,4,dtype=torch.bfloat16,device='cpu')
    seen = []
    handle = c.bridge.register_forward_pre_hook(lambda module,args: seen.append((args[0].device,args[0].dtype)))
    output = c.bridge_tokens(features)
    handle.remove()
    assert seen == [(c.bridge.weight.device,torch.float32)]
    assert output.device == c.bridge.weight.device and output.dtype == torch.float32
    # A meta bridge verifies actual device relocation without requiring GPU or
    # allocating model weights. A dtype-only cast fails Linear's device check.
    c.bridge.to(device='meta')
    relocated = c.bridge_tokens(features)
    assert relocated.device.type == 'meta' and relocated.dtype == torch.float32
