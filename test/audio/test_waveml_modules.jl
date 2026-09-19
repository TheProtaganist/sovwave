"""
    test_waveml_modules.jl

Comprehensive unit test suite for all WaveML deep learning modules:
- WaveTokenizer & WaveForm continuous acoustic tokenization
- WaveDataset, WaveDataLoader, and multi-modal formatters (tabular, text, image, series, Jev)
- HuggingFace hub integration & auth token resolution
- WaveHead task-specific heads (classification, regression, decision, embedding)
- 5 Native Model Types (LLM, Image, 3D, Jev, Video)
- Model Introspection & Reading API
"""

using Test
using Statistics
using LinearAlgebra
using Sovwave

@testset "WaveML Modules Test Suite" begin

    @testset "Harmonic Wave Tokenizer & WaveForm" begin
        tok = default_tokenizer()
        @test tok isa WaveTokenizer
        @test length(tok.inv_vocab) > 100
        @test tok.carrier_frequency == 432.0

        # Tokenize produces Vector{WaveForm}
        wfs = tokenize(tok, "harmonic wave computing")
        @test wfs isa Vector{WaveForm}
        @test !isempty(wfs)
        
        # Test WaveForm attributes (sound & frequency)
        wf1 = wfs[1]
        @test wf1.frequency > 0.0
        @test 0.0 <= wf1.phase <= 2π
        @test length(wf1.samples) == 32
        @test length(wf1.harmonics) == 4
        @test wf1.energy > 0.0

        # Audio synthesis
        audio = to_audio(wfs)
        @test audio isa Vector{Float64}
        @test length(audio) == length(wfs) * 32
        @test maximum(abs.(audio)) <= 1.0

        # Token sonification WAV export
        tmp_wav = tempname() * ".wav"
        try
            sonify_tokens(tok, "sound wave", path=tmp_wav)
            @test isfile(tmp_wav)
            @test filesize(tmp_wav) > 44 # WAV header is 44 bytes
        finally
            isfile(tmp_wav) && rm(tmp_wav, force=true)
        end

        # Decoding
        decoded = decode(tok, wfs)
        @test decoded == "harmonic wave computing"

        # Wave packets
        pkt = to_wave_packet(tok, 1, 32)
        @test length(pkt) == 32
        @test isapprox(norm(pkt), 1.0; atol=1e-4)

        # Sequence matrix encoding
        seq_mat = encode_sequence(tok, "wave"; max_len=8, embed_dim=16)
        @test size(seq_mat) == (16, 8)
    end

    @testset "Multi-Modal Dataset Formatting" begin
        # Tabular
        X_tab = rand(20, 6)
        y_tab = rand(20)
        ds_tab = from_tabular(X_tab, y_tab; task=:regression, embed_dim=16)
        @test num_samples(ds_tab) == 20
        @test length(ds_tab.inputs[1]) == 16
        @test length(ds_tab.targets[1]) == 1

        # Text
        texts = ["quantum harmonic", "neural wave packet", "continuous computing"]
        labels = [1, 2, 1]
        ds_txt = from_text(texts, labels; embed_dim=16)
        @test num_samples(ds_txt) == 3

        # Images
        imgs = [rand(8, 8) for _ in 1:4]
        img_labels = [0, 1, 0, 1]
        ds_img = from_image(imgs, img_labels; embed_dim=16)
        @test num_samples(ds_img) == 4

        # Time-Series
        series = sin.(range(0, 10π, length=100))
        ds_ts = from_timeseries(series; window_size=16, horizon=1, embed_dim=16)
        @test num_samples(ds_ts) == 100 - 16 - 1 + 1

        # Jev Decision States
        states = [Dict("temp" => 20.0, "active" => true), Dict("temp" => 95.0, "active" => false)]
        questions = ["System normal?", "Overheat risk?"]
        decisions = [[1.0, 0.0], [0.0, 1.0]]
        ds_jev = from_jev_state(states, questions, decisions; embed_dim=16)
        @test num_samples(ds_jev) == 2

        # DataLoader batching
        loader = WaveDataLoader(ds_tab; batch_size=5, shuffle=true)
        batches = collect(loader)
        @test length(batches) == 4
        @test length(batches[1][1]) == 5
    end

    @testset "HuggingFace Hub Integration" begin
        # Token resolution test
        ENV["HF_TOKEN"] = "hf_test_token_12345"
        @test hf_auth_token() == "hf_test_token_12345"
        @test hf_auth_token("explicit_token") == "explicit_token"
        delete!(ENV, "HF_TOKEN")

        # Dataset loader fallback / synthetic generation
        hf_ds = load_hf_dataset("imdb"; limit=10, embed_dim=16)
        @test hf_ds isa WaveDataset
        @test num_samples(hf_ds) == 10
        @test hf_ds.modality == :text
    end

    @testset "WaveHead Task Heads" begin
        # Classification head
        head_cls = create_head(:classification, 32, 5; temperature=1.0)
        x = randn(32)
        probs = apply_head(head_cls, x)
        @test length(probs) == 5
        @test isapprox(sum(probs), 1.0; atol=1e-5)
        @test all(probs .>= 0.0)

        # Regression head
        head_reg = create_head(:regression, 32, 2)
        y_pred = apply_head(head_reg, x)
        @test length(y_pred) == 2

        # Decision head
        head_dec = create_head(:decision, 32, 3)
        d_out = apply_head(head_dec, x)
        @test isapprox(sum(d_out), 1.0; atol=1e-5)

        # Embedding head (unit sphere)
        head_emb = create_head(:embedding, 32, 16)
        e_out = apply_head(head_emb, x)
        @test isapprox(norm(e_out), 1.0; atol=1e-5)

        # Head mutation
        old_w = copy(head_cls.weights)
        mutate_head!(head_cls; mutation_rate=1.0, mutation_scale=0.1)
        @test head_cls.weights != old_w
    end

    @testset "Model Introspection API" begin
        cfg = default_config()
        model = WaveModel(cfg)

        @test num_layers(model) == cfg.model.layers
        @test parameter_count(model) > 1000
        @test parameter_count(get_layer(model, 1)) > 0

        # Layer details
        details = layer_details(model, 1)
        @test haskey(details, :amplitudes)
        @test haskey(details, :phases)
        @test haskey(details, :frequencies)
        @test details[:nodes] == cfg.model.nodes

        # Model summary & full inspection
        summary_str = model_summary(model)
        @test occursin("SOVWAVE NEURAL MODEL SUMMARY", summary_str)
        @test occursin("Total Parameters", summary_str)

        info = inspect_model(model)
        @test info[:num_layers] == cfg.model.layers
        @test length(info[:layers]) == cfg.model.layers
    end

    @testset "5 Native Model Architectures" begin
        cfg = default_config()
        model = WaveModel(cfg)
        tok = default_tokenizer()

        # 1. LLM
        text_out = generate_text(model, tok, "quantum wave", max_new_tokens=2)
        @test text_out isa String
        @test length(text_out) >= length("quantum wave")

        # 2. Image Generation
        img = generate_image(model, (8, 8))
        @test size(img) == (8, 8)
        @test all(0.0 .<= img .<= 1.0)

        # 3. Text-to-3D
        vol = generate_3d(model, 4)
        @test size(vol) == (4, 4, 4, 4)
        @test all(0.0 .<= vol .<= 1.0)

        # 4. Jev Decision Engine
        state = Dict("latency" => 12.4, "alert" => false)
        questions = ["System healthy?", "Risk score?"]
        decisions = jev_decide(model, state, questions; question_types=[:boolean, :rubric])
        @test length(decisions) == 2
        @test decisions[1].decision isa Bool
        @test decisions[2].decision isa Real
        @test 0.0 <= decisions[2].decision <= 1.0

        # Extensible registry
        types = list_model_types()
        @test haskey(types, :llm)
        @test haskey(types, :image_generation)
        @test haskey(types, :text_to_3d)
        @test haskey(types, :jev)
        @test haskey(types, :text_to_video)
    end

end
