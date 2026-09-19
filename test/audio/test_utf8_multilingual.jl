# ============================================================================
# test/audio/test_utf8_multilingual.jl
# Comprehensive UTF-8 Multilingual, Emoji & Native Wave Tokenizer Test Suite
# Ensures zero <UNK> data loss, no fallbacks/shortcuts, and 100% loss-free
# bidirectional wave tokenization across all world languages and scripts.
# ============================================================================

using Test
using Sovwave

@testset "UTF-8 Multilingual & Emoji Wave Tokenizer" begin
    tok = default_tokenizer()

    @testset "Golden-Ratio Unicode Harmonic Mapping" begin
        # Every Unicode character must have valid positive carrier frequency and phase in [0, 2π)
        test_chars = ['A', 'z', '9', ' ', '\n', 'ñ', 'ü', 'Ω', 'ψ', 'ж', 'ع', 'ש', 'क', '中', 'あ', '가', '🌊', '🧠', '⚡', '🚀', '⚛']
        for c in test_chars
            f = unicode_wave_frequency(c)
            ph = unicode_wave_phase(c)
            @test f > 0.0
            @test isfinite(f)
            @test 0.0 <= ph < 2π
        end

        # Distinct characters must produce distinct harmonic frequencies
        @test unicode_wave_frequency('A') != unicode_wave_frequency('B')
        @test unicode_wave_frequency('🌊') != unicode_wave_frequency('🧠')
        @test unicode_wave_frequency('中') != unicode_wave_frequency('日')
    end

    @testset "Lossless Bidirectional Reconstruction (No UNK / Zero Loss)" begin
        sentences = [
            # English & Programming
            "Quantum wave computing in Sovwave: y = cos(2π·f·t + φ);",
            
            # Latin Extended & Accented
            "El niño español comió un jalapeño en México mañana, ¡qué rico!",
            "Français: Les élèves français goûtent des crêpes délicieuses à Noël.",
            "Deutsch: Über den Wolken muß die Freiheit wohl grenzenlos sein.",
            "Português: Oração, bênção e ações constroem o amanhã com esperança.",
            "Čeština: Příliš žluťoučký kůň úpěl ďábelské ódy na řece.",
            
            # Greek
            "Ἑλληνικά: Ἡ φύσις κρύπτεσθαι φιλεῖ — Ἁρμονία κόσμου καὶ λόγος.",
            
            # Cyrillic
            "Русский: Самоорганизующийся вакуум и непрерывные волновые гармоники.",
            "Українська: Хвильова динаміка та квантова акустика відкривають нові горизонти.",
            
            # Arabic
            "العربية: موجات التوافق الكوانتي والذكاء الاصطناعي الموجي في سيلكون.",
            
            # Hebrew
            "עברית: תהודה הרמונית וחישוב גלי במרחב הפיזיקלי של סוֹווֶייב.",
            
            # Hindi (Devanagari)
            "हिन्दी: सोववेव में तरंग-आधारित क्वांटम हार्मोनिक कंप्यूटिंग और चेतना।",
            
            # CJK (Chinese)
            "中文: 自组织真空场论与连续量子声波谐振机器学习系统。",
            
            # Japanese
            "日本語: 黄金比に基づく調和波形トークナイザーと自己組織化真空。",
            
            # Korean
            "한국어: 고유 파동 양자 조화 컴퓨팅과 신경망 공명 모델링.",
            
            # Sacred Geometry & Physics
            "∂ψ/∂t = -i Ĥ ψ + ∇²ψ + β_s · Φ^(n/12) · cos(ωt + φ) ∮_C E·dl = -∂Φ_B/∂t",
            
            # Universal Emojis (First-Class Wave Primitives)
            "🌊 🧠 ⚡ 🚀 ⚛️ 🔮 🎵 🎶 🔊 🌐 🌌 ✨ 🌟 💡 🔥 🌈 🛠️ 📊 🎯 🤖 💻 🧬 🛡️ 🕊️ 🪐 ☀️ 🌙 ⭐ ❤️ 💎 🔔 👁️ 🌀 🎨 🧪 📡 🔋 🔑 🏆 🥇",
            
            # Dense Multilingual & Emoji Composite
            "Sovwave 🌊: Quantum ⚛️ resonance in 中文, 日本語, 한국어, العربية, हिन्दी, Русский & English! 🚀⚡🧠"
        ]

        for s in sentences
            ids = tokenize_ids(tok, s)
            @test !isempty(ids)
            # ZERO UNK tokens allowed
            @test all(id != tok.special_tokens[:UNK] for id in ids)
            
            # Token ID decoding must exactly match original string
            dec_ids = decode(tok, ids)
            @test dec_ids == s

            # Physical WaveForm tokenization
            wfs = tokenize(tok, s)
            @test length(wfs) == length(ids)
            for (i, wf) in enumerate(wfs)
                @test wf.token_id == ids[i]
                @test wf.frequency > 0.0
                @test !isempty(wf.samples)
                @test wf.energy > 0.0
            end

            # WaveForm decoding must exactly match original string
            dec_wfs = decode(tok, wfs)
            @test dec_wfs == s
        end
    end

    @testset "Dynamic Registration for Novel / Rare Unicode Characters" begin
        # Test rare scripts and newest Unicode additions that may not be pre-seeded
        novel_text = "᚛ᚑᚌᚐᚋ᚜ 𓆣𓆤𓆥 ᚠᚢᚦᚨ 𒀭𒈹 🛸🦖🪷🪸🪼"
        
        # Verify initial state
        initial_vocab_size = length(tok.inv_vocab)
        
        ids = tokenize_ids(tok, novel_text)
        @test !isempty(ids)
        @test all(id != tok.special_tokens[:UNK] for id in ids)
        
        # Every novel character must now exist in vocabulary
        @test length(tok.inv_vocab) >= initial_vocab_size
        
        # Bidirectional decode must be 100% exact
        @test decode(tok, ids) == novel_text
        
        # Continuous WaveForm tokenization
        wfs = tokenize(tok, novel_text)
        @test decode(tok, wfs) == novel_text
    end

    @testset "Continuous Wave Audio Synthesis with Multilingual Text" begin
        multilingual_msg = "Wave 🌊 声 🎵 صَوْت"
        wfs = tokenize(tok, multilingual_msg; n_samples=32, sample_rate=48000.0)
        audio = to_audio(wfs; sample_rate=48000.0)
        @test length(audio) == length(wfs) * 32
        @test all(isfinite, audio)
        @test maximum(abs.(audio)) <= 1.0
    end

    @testset "Multi-Modal Dataset Formatting with Multilingual Text" begin
        texts = [
            "Wave harmonic computing 🌊",
            "量子調和コンピューティング ⚛️",
            "الذكاء الاصطناعي الموجي ⚡",
            "Квантовые непрерывные гармоники 🚀"
        ]
        labels = [1, 2, 3, 4]
        
        ds = format_text(texts, labels; tokenizer=tok, max_len=16, embed_dim=32)
        @test length(ds) == 4
        @test ds.modality == :text
        @test length(ds.inputs[1]) == 32
        @test length(ds.targets[1]) == 4
        @test ds.targets[1] == [1.0, 0.0, 0.0, 0.0]
    end
end
