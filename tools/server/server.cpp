#include "server-connect.h"
#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// how the server is run; everything is on when it is the llama-server binary
struct server_run_opts {
    bool install_signal_handlers = true; // off when embedded in llama-cli, which handles them itself
    bool resolve_model           = true; // download / resolve -hf and docker models; off when the caller did it already
};

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}


// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const common_json_error & e) {
            // JSON parse and type errors are invalid requests (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

//
// shared by every role
//

static server_role_type server_role_detect(const common_params & params, server_role & role) {
    if (role.is_managed_by_router()) {
        return role.get_role();
    }
    // a server given no model is a router
    const bool has_model = !params.model.path.empty()
                        || !params.model.hf_repo.empty()
                        || !params.model.docker_repo.empty();
    return has_model ? SERVER_ROLE_INSTANCE : SERVER_ROLE_ROUTER;
}

// the GPUs have sparse buffers (CUDA virtual memory), which the DeepSeek V4.1 shared KV pool needs
static bool server_sparse_kv_supported() {
    size_t n_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        auto fn = (ggml_backend_dev_sparse_buffer_type_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_sparse_buffer_type");
        if (fn == nullptr || fn(dev) == nullptr) {
            return false;
        }
        n_gpu++;
    }
    return n_gpu > 0;
}

// the shared KV pool can be used: sparse buffers on the GPUs, and not turned off with LLAMA_DSV4_SHARED_POOL=0
static bool server_kv_pool_available() {
    const char * env = getenv("LLAMA_DSV4_SHARED_POOL");
    return !(env && atoi(env) == 0) && server_sparse_kv_supported();
}

// DeepSeek V4.1 on several GPUs: the settings measured best on 8x RTX 3090 at 512K+ context (see the server
// README), applied to the options the user did not set
static void server_apply_deepseek41_defaults(common_params & params) {
    const auto is_set = [&](const char * name) { return params.args_set.count(name) > 0; };
    std::vector<std::string> applied;

    // the whole context of the model (1M), as with -c 0: --fit then halves the ubatch or measures the smaller prompt
    // graph instead of shrinking the context, which it does when -c is left unset
    if (!is_set("-c") && !is_set("-fitc") && !is_set("--kv-unified-per-slot")) {
        params.n_ctx              = 0;
        params.fit_params_min_ctx = UINT32_MAX;
        applied.push_back("-c 0 (the whole context of the model)");
    }

    // 256-token ubatches: 15% faster prefill than 128 and small enough compute buffers for 512K context; --fit
    // halves them (to 128) rather than keep weights in host memory, as at 1M context
    if (!is_set("-ub")) {
        params.n_ubatch       = 256;
        params.fit_ubatch_min = 128;
        applied.push_back("-ub 256 (128 if needed to fit)");
    }

    // the layer split of the larger prompt graph while it fits: the memory the smaller graph saves moves layers between
    // the GPUs, which slows prefill by 3.5% at 526K; it is used when weights would stay in host memory (1M context)
    params.fit_large_graph = true;

    // 384 MiB free per device: room for the pipeline-parallel buffers next to a DSpark draft model at 512K context
    if (!is_set("-fitt")) {
        std::fill(params.fit_params_target.begin(), params.fit_params_target.end(), (size_t) 384*1024*1024);
        applied.push_back("-fitt 384");
    }

    // the draft model whole on the last GPU, which --fit gives the fewest target layers
    const bool has_draft = !params.speculative.draft.mparams.path.empty();
    if (has_draft && !is_set("-devd")) {
        ggml_backend_dev_t last = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                last = dev;
            }
        }
        if (last != nullptr && params.devices.empty()) {
            params.speculative.draft.devices = { last, nullptr };
            applied.push_back(std::string("-devd ") + ggml_backend_dev_name(last));
        }
    }
    if (has_draft && !is_set("-ngld")) {
        params.speculative.draft.n_gpu_layers = -2; // all
        applied.push_back("-ngld all");
    }

    // 4 draft tokens: +8% decode over 3 at 131K (93% accepted), the same at 16K; 5 is slower
    if (has_draft && !is_set("--spec-draft-n-max")) {
        params.speculative.draft.n_max = 4;
        applied.push_back("--spec-draft-n-max 4");
    }

    // several slots share one context (the shared KV pool) instead of n_ctx/N each, when the GPUs support it;
    // the auto slot count is handled with -np
    if (params.n_parallel > 1 && !is_set("--kv-unified") && server_kv_pool_available()) {
        params.kv_unified = true;
        applied.push_back("--kv-unified (shared KV pool)");
    }

    // the projector on the CPU: on a GPU it takes the room of expert weights, which slows every token down
    // (an image takes 4-5 s on the CPU instead of about 1 s)
    if (!params.mmproj.path.empty() && !is_set("--no-mmproj-offload")) {
        params.mmproj_use_gpu = false;
        applied.push_back("--no-mmproj-offload");
    }

    if (!applied.empty()) {
        std::string s;
        for (const auto & a : applied) {
            s += (s.empty() ? "" : " ") + a;
        }
        SRV_INF("DeepSeek V4.1 defaults for the options not set: %s\n", s.c_str());
    }
}

// adjust the params for the role; returns false on a configuration error
static bool server_prepare_params(common_params & params, server_role_type role_type) {
    if (role_type != SERVER_ROLE_ROUTER) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        const std::string arch = params.model.path.empty() ? std::string() : common_get_gguf_info(params.model.path).arch;

        if (arch == "deepseek41") {
            server_apply_deepseek41_defaults(params);
        }

        if (params.n_parallel < 0 && arch == "deepseek41" && server_kv_pool_available()) {
            // the shared KV pool: 4 slots that share one context (each can grow to all of it), the KV memory of
            // the idle slots is given back as the others need it
            SRV_INF("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true (shared KV pool for deepseek41)\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }

        // the shared KV pool of DeepSeek V4.1 (see llama_kv_cache_dsv4): every slot can grow to the whole context,
        // the server purges idle slots only when the pool is full
        if (arch == "deepseek41" && params.kv_unified && params.n_parallel > 1 && server_kv_pool_available()) {
            params.kv_shared_pool = true;
        }

        if (params.n_parallel < 0) {
            // DeepSeek V4/V4.1 keep one KV stream per sequence: with unified KV every stream would be sized for the
            // whole context (4 slots = 4x the memory, which pushes weights out of the devices), and without it 4
            // slots would each get a quarter of the context. One slot keeps the context that was asked for.
            if (arch == "deepseek4" || arch == "deepseek41") {
                SRV_INF("n_parallel is set to auto, using n_parallel = 1 for %s (one KV stream per slot, use -np N for N slots of n_ctx/N)\n", arch.c_str());

                params.n_parallel = 1;
                params.kv_unified = false;
            } else {
                SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

                params.n_parallel = 4;
                params.kv_unified = true;
            }
        }
    }

    // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
    // or with -c 0 for max context
    const bool ctx_pool_auto_sized = params.kv_unified_per_slot > 0 &&
                                     params.n_ctx == 0 &&
                                     (uint32_t) params.fit_params_min_ctx != UINT32_MAX;

    if (ctx_pool_auto_sized) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
        SRV_INF("--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n", params.n_parallel,
                params.kv_unified_per_slot, params.n_ctx);
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // check early, so a missing llama-connect fails-fast
    if (params.server_connect) {
        const std::string reason = server_connect::unavailable_reason(params);
        if (!reason.empty()) {
            SRV_ERR("--connect is not available: %s\n", reason.c_str());
            return false;
        }
        params.server_connect_code = server_connect::resolve_code(params.server_connect_code);
    } else if (!params.server_connect_code.empty()) {
        // the code is only meaningful while the tunnel runs, and /props keys off it being set
        SRV_WRN("%s", "--connect-code is ignored without --connect\n");
        params.server_connect_code.clear();
    }

    return true;
}

// the API of every server; a router points the handlers in routes at its proxies before calling this
static void server_register_routes(server_http_context & ctx_http, server_routes & routes) {
    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/tts",                      ex_wrapper(routes.post_tts));
    ctx_http.post("/v1/audio/speech",          ex_wrapper(routes.post_speech_oai));
    ctx_http.post("/audio/speech",             ex_wrapper(routes.post_speech_oai));
    ctx_http.ws  ("/v1/realtime",              routes.ws_realtime);
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/systemone",             ex_wrapper(routes.post_systemone));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    ctx_http.get ("/steering",                 ex_wrapper(routes.get_steering));
    ctx_http.post("/steering",                 ex_wrapper(routes.post_steering));
    ctx_http.get ("/speculative",              ex_wrapper(routes.get_speculative));
    ctx_http.post("/speculative",              ex_wrapper(routes.post_speculative));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();
}

// resumable streaming, see server-stream.h: an instance binds the local session factories,
// a router binds proxies that resolve the owning child
static void server_register_stream_routes(
        server_http_context & ctx_http,
        const server_http_context::handler_t & stream_get,
        const server_http_context::handler_t & streams_lookup,
        const server_http_context::handler_t & stream_delete) {
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete));
}

// the features behind a flag, and the warning about the ones that must not face untrusted clients
// returns false when one of them fails to start
static bool server_setup_features(
        server_http_context & ctx_http,
        const common_params & params,
        server_mcp & mcp_mgr,
        server_tools & tools,
        server_role_type role_type) {
    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)\n");
    }

    std::vector<std::string> warn_names;
    if (role_type == SERVER_ROLE_ROUTER) {
        warn_names.push_back("router mode");
    }

    if (params.server_connect) {
        warn_names.push_back("peer-to-peer tunnel (llama-connect, experimental)");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return false;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return false;
        }
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("server tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    if (!warn_names.empty()) {
        std::string features;
        for (const auto & name : warn_names) {
            if (!features.empty()) features += ", ";
            features += name;
        }
        SRV_WRN("security: %s enabled - do not expose to untrusted environments\n", features.c_str());
    }

    return true;
}

static void server_install_signal_handlers() {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
}

// once the HTTP server listens: log the addresses and spawn llama-connect if asked, so the port is
// bound and the child startup health check reaches us. the destructor of connect_proc stops it
static bool server_on_listening(server_http_context & ctx_http, const common_params & params, server_connect & connect_proc) {
    for (const auto & address : ctx_http.listening_addresses) {
        SRV_INF("listening on %s\n", address.c_str());
    }

    if (params.server_connect && !connect_proc.start(params, ctx_http.port)) {
        SRV_ERR("%s", "exiting due to llama-connect error\n");
        return false;
    }
    return true;
}

//
// download worker: a child of the router that pulls a model into the cache and reports the progress
//

struct server_download_state : public common_download_callback {
    server_role * role;
    std::function<bool()> should_stop;
    std::atomic<int64_t> last_progress_time{0}; // multiple files downloading in different threads
    bool is_ok = false;

    server_download_state(server_role * r) : role(r) {}

    bool run(common_params & params) {
        try {
            common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            common_models_handler_apply(handler, params, this);
            is_ok = true;
        } catch (const std::exception & e) {
            auto model_name = params.model.get_name();
            SRV_ERR("download failed for model name=%s: %s\n", model_name.c_str(), e.what());
            is_ok = false;
        }
        return is_ok;
    }
    void on_progress(const common_download_progress & p) {
        json data = {
            {"url", p.url},
            {"downloaded", p.downloaded},
            {"total", p.total},
        };
        role->notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), data);
    }
    void on_start(const common_download_progress & p) override {
        on_progress(p);
    }
    void on_update(const common_download_progress & p) override {
        int64_t now = ggml_time_ms();
        // throttle progress updates to avoid flooding logs
        if (now - last_progress_time.load(std::memory_order_relaxed) >= 100) {
            on_progress(p);
            last_progress_time.store(now, std::memory_order_relaxed);
        }
    }
    void on_done(const common_download_progress & p, bool) override {
        on_progress(p);
    }
    bool is_cancelled() const override {
        return should_stop ? should_stop() : false;
    }
};

static int server_run_download(common_params & params, server_role & role) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    // monitor stdin for cancellation command from the router
    std::thread signal_thread = role.setup([cancelled](int) {
        cancelled->store(true, std::memory_order_relaxed);
    });

    server_download_state dl(&role);
    dl.should_stop = [cancelled]() {
        return cancelled->load(std::memory_order_relaxed);
    };

    bool ok = dl.run(params);

    role.notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), {
        {"result", ok ? "download_finished" : "download_failed"},
    });

    // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
    if (signal_thread.joinable()) {
        signal_thread.join();
    }

    SRV_INF("download completed %s\n", ok ? "successfully" : "with errors");
    return 0;
}

//
// measure worker: a child of the router, reports the memory a model needs without loading it
//

static int server_run_measure(common_params & params, server_role & role, common_models_handler & models_handler) {
    // the router sets --offline: only resolve the local path, never download
    try {
        common_models_handler_apply(models_handler, params);
    } catch (const std::exception & e) {
        SRV_ERR("failed to resolve model: %s\n", e.what());
        return 1;
    }

    const std::string model_name = params.model.get_name();

    // TODO: mmproj is not in llama_get_memory_breakdown, a multimodal model is underestimated
    json payload;
    try {
        common_memory_reports m = common_measure_memory(params);
        payload = {
            {"result",      "measured"},
            {"n_ctx_train", m.n_ctx_train},
            {"min",         server_model_memory(m.n_ctx_min, m.info_min).to_json()},
            {"max",         server_model_memory(m.n_ctx_max, m.info_max).to_json()},
        };
        SRV_INF("memory measured for model name=%s at n_ctx=%u (min) and n_ctx=%u (max)\n",
            model_name.c_str(), m.n_ctx_min, m.n_ctx_max);
    } catch (const std::exception & e) {
        SRV_ERR("failed to measure memory for model name=%s: %s\n", model_name.c_str(), e.what());
        payload = {
            {"result", "failed"},
            {"error",  e.what()},
        };
    }

    role.notify_to_router(server_state_to_str(SERVER_STATE_MEASURING), payload);
    return 0;
}

//
// router
//

static int server_run_router(common_params & params, int argc, char ** argv, const server_run_opts & opts) {
    SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // the router loads no model, its context stays empty; the routes it does not proxy still need one
    server_context ctx_server;
    server_routes  routes(params, ctx_server);
    server_tools   tools;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    // setup server instances manager
    std::optional<server_models_routes> models_routes;
    try {
        models_routes.emplace(params, argc, argv);
    } catch (const std::exception & e) {
        SRV_ERR("failed to initialize router models: %s\n", e.what());
        return 1;
    }

    // proxy handlers
    // note: routes.get_health stays the same
    routes.get_metrics                 = models_routes->proxy_get;
    routes.post_props                  = models_routes->proxy_post;
    routes.post_completions            = models_routes->proxy_post;
    routes.post_completions_oai        = models_routes->proxy_post;
    routes.post_chat_completions       = models_routes->proxy_post;
    routes.post_control                = models_routes->proxy_post;
    routes.post_responses_oai          = models_routes->proxy_post;
    routes.post_transcriptions_oai     = models_routes->proxy_post;
    routes.post_tts                    = models_routes->proxy_post;
    routes.post_speech_oai             = models_routes->proxy_post;
    routes.ws_realtime                 = models_routes->proxy_ws;
    routes.post_anthropic_messages     = models_routes->proxy_post;
    routes.post_anthropic_count_tokens = models_routes->proxy_post;
    routes.post_infill                 = models_routes->proxy_post;
    routes.post_embeddings             = models_routes->proxy_post;
    routes.post_embeddings_oai         = models_routes->proxy_post;
    routes.post_rerank                 = models_routes->proxy_post;
    routes.post_systemone              = models_routes->proxy_post;
    routes.post_tokenize               = models_routes->proxy_post;
    routes.post_detokenize             = models_routes->proxy_post;
    routes.post_apply_template         = models_routes->proxy_post;
    routes.post_chat_completions_tok   = models_routes->proxy_post;
    routes.post_responses_tok_oai      = models_routes->proxy_post;
    routes.get_lora_adapters           = models_routes->proxy_get;
    routes.post_lora_adapters          = models_routes->proxy_post;
    routes.get_steering                = models_routes->proxy_get;
    routes.post_steering               = models_routes->proxy_post;
    routes.get_speculative             = models_routes->proxy_get;
    routes.post_speculative            = models_routes->proxy_post;
    routes.get_slots                   = models_routes->proxy_get;
    routes.post_slots                  = models_routes->proxy_post;

    // custom routes for router
    routes.get_props                   = models_routes->get_router_props;
    routes.get_models                  = models_routes->get_router_models;

    ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
    ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
    ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
    ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
    ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));

    server_register_routes(ctx_http, routes);
    server_register_stream_routes(ctx_http,
        models_routes->router_stream_get,
        models_routes->router_streams_lookup,
        models_routes->router_stream_delete);

    if (!server_setup_features(ctx_http, params, mcp_mgr, tools, SERVER_ROLE_ROUTER)) {
        return 1;
    }

    auto clean_up = [&models_routes, &mcp_mgr]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        // stop the session GC first, it finalizes live sessions and wakes pending readers
        server_stream_session_manager_stop();
        if (models_routes.has_value()) {
            models_routes->stopping.store(true); // maybe redundant, but just to be safe
            models_routes->models.unload_all();
        }
        mcp_mgr.shutdown();
        llama_backend_free();
    };

    if (!ctx_http.start()) {
        clean_up();
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }
    ctx_http.is_ready.store(true);

    shutdown_handler = [&](int) {
        if (models_routes.has_value()) {
            // important to disconnect any SSE clients
            models_routes->stopping.store(true);
        }
        mcp_mgr.shutdown();
        ctx_http.stop();
    };

    try {
        models_routes->models.load_startup_models();
    } catch (const std::exception & e) {
        SRV_ERR("failed to load models on startup: %s\n", e.what());
        ctx_http.stop();
        ctx_http.join();
        clean_up();
        return 1;
    }

    if (opts.install_signal_handlers) {
        server_install_signal_handlers();
    }

    server_connect connect_proc;
    if (!server_on_listening(ctx_http, params, connect_proc)) {
        ctx_http.stop();
        ctx_http.join();
        clean_up();
        return 1;
    }

    if (!params.models_preset_hf.empty()) {
        SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
        SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
    }

    ctx_http.join(); // keep the main thread alive

    // when the HTTP server stops, clean up and exit
    clean_up();

    return 0;
}

//
// instance
//

static int server_run_instance(common_params & params, server_role & role, common_models_handler & models_handler, const server_run_opts & opts) {
    // -hf / docker: download if needed and resolve the local path. a child resolves what the router downloaded
    if (opts.resolve_model) {
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;
    server_routes  routes(params, ctx_server);
    server_tools   tools;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    server_register_routes(ctx_http, routes);
    server_register_stream_routes(ctx_http,
        server_stream_make_get_handler(),
        server_stream_make_lookup_handler(),
        server_stream_make_delete_handler());

    if (!server_setup_features(ctx_http, params, mcp_mgr, tools, SERVER_ROLE_INSTANCE)) {
        return 1;
    }

    // setup clean up function, to be called before exit
    auto clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        // stop the session GC first, it finalizes live sessions and wakes pending readers
        server_stream_session_manager_stop();
        ctx_http.stop();
        ctx_server.terminate();
        mcp_mgr.shutdown();
        llama_backend_free();
    };

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!ctx_http.start()) {
        clean_up();
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }

    // setup communication child --> router if necessary
    if (role.is_managed_by_router()) {
        ctx_server.set_state_callback([&](server_state state, json payload) {
            role.notify_to_router(server_state_to_str(state), payload);
        });
    }

    if (!ctx_server.load_model(params)) {
        clean_up();
        ctx_http.join();
        SRV_ERR("%s", "exiting due to model loading error\n");
        return 1;
    }

    routes.update_meta(ctx_server);
    ctx_http.is_ready.store(true);

    SRV_INF("%s", "model loaded\n");

    shutdown_handler = [&](int) {
        mcp_mgr.shutdown();
        // this will unblock start_loop()
        ctx_server.terminate();
    };

    if (opts.install_signal_handlers) {
        server_install_signal_handlers();
    }

    server_connect connect_proc;
    if (!server_on_listening(ctx_http, params, connect_proc)) {
        ctx_http.stop();
        ctx_http.join();
        clean_up();
        return 1;
    }

    // optionally, notify router server that this instance is ready
    std::thread monitor_thread;
    if (role.is_managed_by_router()) {
        monitor_thread = role.setup(shutdown_handler);
        json info = routes.get_model_info();
        // the router tracks device memory from these reports
        if (const auto * ll_ctx = ctx_server.get_llama_context()) {
            info["memory"] = server_model_memory(llama_n_ctx(ll_ctx), common_memory_breakdown_rows(ll_ctx)).to_json();
        }
        role.notify_to_router(server_state_to_str(SERVER_STATE_READY), info);
    }

    // this call blocks the main thread until queue_tasks.terminate() is called
    ctx_server.start_loop();

    clean_up();
    ctx_http.join();
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }

    auto * ll_ctx = ctx_server.get_llama_context();
    if (ll_ctx != nullptr) {
        common_memory_breakdown_print(ll_ctx);
    }

    return 0;
}

//
// entry points
//

static int server_run(common_params & params, int argc, char ** argv, server_role & role, const server_run_opts & opts) {
    common_models_handler models_handler;

    // -hf may name a preset repo, which makes this a router: settle that before deciding the role
    if (opts.resolve_model && !params.model.hf_repo.empty()) {
        try {
            models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            if (common_models_handler_is_preset_repo(models_handler)) {
                // apply the preset and start the server in router mode
                common_models_handler_apply(models_handler, params);
            }
        } catch (const std::exception & e) {
            SRV_ERR("failed to fetch model metadata: %s\n", e.what());
            return 1;
        }
    }

    const server_role_type role_type = server_role_detect(params, role);

    // only an instance loads a model; the others must not touch the GPU, so skip the
    // device enumeration for them and the CUDA primary context stays uncreated
    common_params_print_info(params, role_type == SERVER_ROLE_INSTANCE);

    if (!server_prepare_params(params, role_type)) {
        return 1;
    }

    switch (role_type) {
        case SERVER_ROLE_DOWNLOAD: return server_run_download(params, role);
        case SERVER_ROLE_MEASURE:  return server_run_measure(params, role, models_handler);
        case SERVER_ROLE_ROUTER:   return server_run_router(params, argc, argv, opts);
        case SERVER_ROLE_INSTANCE: return server_run_instance(params, role, models_handler, opts);
    }

    GGML_ABORT("invalid server role");
}

int llama_server(int argc, char ** argv) {
    server_role role;

    std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
    // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
    signal(SIGPIPE, SIG_IGN);
#endif

    // own arguments required by this example
    common_params params;

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    server_stream_session_manager_start();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    SRV_INF("%s", "initializing ...\n");

    llama_backend_init();
    llama_numa_init(params.numa);

    const int result = server_run(params, argc, argv, role, server_run_opts{});
    common_log_flush(common_log_main());
    return result;
}

int llama_server(common_params & params, int argc, char ** argv) {
    // embedded in llama-cli: it handles the signals, and the model is resolved already
    server_run_opts opts;
    opts.install_signal_handlers = false;
    opts.resolve_model           = false;

    server_role role;
    return server_run(params, argc, argv, role, opts);
}
