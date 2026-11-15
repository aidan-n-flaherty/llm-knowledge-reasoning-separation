import torch

from krs.config import Config
from krs.model.memory import MemoryContext
from krs.model.transformer import Llama3Model


def tiny_model():
    cfg = Config()
    cfg.model.emb_dim, cfg.model.n_heads, cfg.model.n_kv_groups = 32, 4, 2
    cfg.model.n_layers, cfg.model.hidden_dim, cfg.model.context_length = 3, 64, 16
    cfg.memory.kind, cfg.memory.layers, cfg.memory.slots = "entity_block", [1], 2
    cfg.memory.key_dim = cfg.memory.value_dim = 8
    torch.manual_seed(0)
    return Llama3Model(cfg, vocab_size=50, n_entities=10)


def test_memory_is_a_no_op_at_init():
    model = tiny_model()
    tokens = torch.randint(0, 50, (4, 12))
    ids = torch.tensor([0, 3, 7, 9])
    assert torch.equal(model(tokens, MemoryContext(ids)), model(tokens, None))


def test_masked_rows_match_the_base_model_and_active_rows_do_not():
    model = tiny_model()
    for proj in model.memory.out_proj.values():
        torch.nn.init.normal_(proj.weight, std=0.5)
    tokens = torch.randint(0, 50, (4, 12))
    ids = torch.tensor([0, 3, -1, 9])
    masked = torch.tensor([False, True, False, True])

    base = model(tokens, None)
    out = model(tokens, MemoryContext(ids, masked))
    assert not torch.allclose(out[0], base[0])  # active entity: memory changes the output
    assert torch.equal(out[1:], base[1:])  # masked rows and the no-entity row: exactly the base model
    assert torch.equal(model(tokens, MemoryContext.off(ids)), base)
