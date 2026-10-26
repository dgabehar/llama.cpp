import pytest
import re
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 16,
        "return_tokens": True,
    }

    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    # create new server with draft model
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    tokens_draft = res.body["tokens"]

    assert tokens_no_draft == tokens_draft

    server.stop()
    create_server()
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.0] * server.spec_draft_n_max
    server.start()
    res = server.make_request("POST", "/completion", data=request)

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == 0
    assert res.body["tokens"] == tokens_no_draft


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_synth_is_deterministic():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.75 ** (i + 1) for i in range(server.spec_draft_n_max)]
    server.start()

    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 32,
    }
    responses = [server.make_request("POST", "/completion", data=request) for _ in range(2)]

    for res in responses:
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
    assert responses[0].body["timings"]["draft_n"] == responses[1].body["timings"]["draft_n"]
    assert responses[0].body["timings"]["draft_n_accepted"] == responses[1].body["timings"]["draft_n_accepted"]


def test_synth_ignores_target_tokens():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [1.0] * server.spec_draft_n_max
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 32,
    })

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == res.body["timings"]["draft_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 6,
        "grammar": 'root ::= "a"{5,5}',
    })
    assert res.status_code == 200, res.body

    res = server.make_request("POST", "/completion", data={
        "prompt": "Respond with only: OK",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 64,
        "ignore_eos": True,
    })
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 64
    assert res.body["stop_type"] == "limit"


def test_draft_n_ubatch_smaller_than_target():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    }

    # draft with the default ubatch size of the draft context
    server.start()
    res_default = server.make_request("POST", "/completion", data=request)
    server.stop()
    assert res_default.status_code == 200
    assert res_default.body["timings"]["draft_n"] > 0

    # a smaller draft ubatch splits the draft batches into more ubatches, but must not change the result
    server.spec_draft_n_ubatch = 8
    server.start()
    res_small = server.make_request("POST", "/completion", data=request)
    server.stop()
    assert res_small.status_code == 200
    assert res_small.body["timings"]["draft_n"] > 0
    assert res_small.body["tokens"] == res_default.body["tokens"]


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])


def test_draft_token_probs():
    # tokens accepted from the draft must carry the target model's probabilities,
    # same as without speculative decoding (#24271)
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
        "n_probs": 4,
    }

    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    probs_no_draft = res.body["completion_probabilities"]
    server.stop()

    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    probs_draft = res.body["completion_probabilities"]

    assert len(probs_draft) == len(probs_no_draft)
    for tok_draft, tok_no_draft in zip(probs_draft, probs_no_draft):
        assert tok_draft["id"] == tok_no_draft["id"]
        assert len(tok_draft["top_logprobs"]) == len(tok_no_draft["top_logprobs"])
        assert abs(tok_draft["logprob"] - tok_no_draft["logprob"]) < 0.25
    # real logprobs are not all exactly 0.0 (prob 1.0)
    assert any(tok["logprob"] < -1e-4 for tok in probs_draft)


def test_grammar_drafts_only_with_spec_grammar():
    global server
    # lowercase words only, so the grammar rejects most of what the stories model would say
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "n_predict": 32,
        "grammar": "root ::= [a-z]+ (\" \" [a-z]+)*",
        "return_tokens": True,
    }

    server.model_draft = None
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_plain = res.body["tokens"]
    server.stop()

    # default: the guard keeps grammar requests on plain decode
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"].get("draft_n", 0) == 0
    assert res.body["tokens"] == tokens_plain
    server.stop()

    # --spec-grammar: drafts, and greedy output under the grammar is unchanged
    create_server()
    server.spec_grammar = True
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["tokens"] == tokens_plain
    assert re.fullmatch(r"[a-z]+( [a-z]+)*", res.body["content"])


# Hybrid/recurrent targets (qwen35 family) cannot trim a rejected draft out of their recurrent state,
# so with ngram-mod alone (no spare rs sequences) every partial acceptance restores a checkpoint
# and replays: the restore has to bring the grammar back too. Needs a local qwen35-family GGUF,
# e.g. LLAMA_TEST_HYBRID_MODEL=Qwen3.5-0.8B-Q8_0.gguf
@pytest.mark.skipif(not os.environ.get("LLAMA_TEST_HYBRID_MODEL"), reason="LLAMA_TEST_HYBRID_MODEL not set")
def test_grammar_spec_checkpoint_restore_hybrid(tmp_path, monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_TRACE", "1") # logs each "restore checkpoint"
    server = ServerProcess()
    server.log_path = str(tmp_path / "server.log")
    server.model_file = os.environ["LLAMA_TEST_HYBRID_MODEL"]
    server.model_hf_repo = None
    server.model_hf_file = None
    server.n_ctx = 4096
    server.n_slots = 1
    server.spec_type = "ngram-mod"
    server.spec_grammar = True
    server.start()
    n_draft, n_acc = 0, 0
    # sampled, so drafts from the repeated list get rejected part way and the slot restores
    for seed in range(8):
        res = server.make_request("POST", "/completion", data={
            "prompt": "Shopping list, one per line:\n- milk\n- eggs\n- bread\n- milk\n- apples\n- eggs\n- rice\n- bread\n-",
            "temperature": 0.8,
            "seed": seed,
            "n_predict": 128,
            "grammar": "root ::= (\" \" [a-z]+ \"\\n-\")+",
        })
        assert res.status_code == 200
        # n_predict can stop anywhere inside the last item
        assert re.fullmatch(r"( [a-z]+\n-)*( [a-z]*\n?)?", res.body["content"]), repr(res.body["content"])
        n_draft += res.body["timings"].get("draft_n", 0)
        n_acc   += res.body["timings"].get("draft_n_accepted", 0)
    server.stop()
    n_restore = open(server.log_path).read().count("(restore checkpoint)")
    assert n_draft > 0
    assert n_acc < n_draft
    assert n_restore >= 4, f"only {n_restore} checkpoint restores, the replay path was barely exercised"
