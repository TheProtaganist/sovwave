using Sovwave

println("================================================================================")
println(" 🌊 TESTING WAVE TOKENIZER: TOKENS ARE PHYSICAL WAVE FREQUENCIES (HZ) 🌊")
println("================================================================================")

# 1. Instantiate Tokenizer
tok = default_tokenizer()

# 2. Tokenize into Physical Wave Frequencies
text = "The universe operates on harmonic wave interference."
wave_freqs = tokenize_frequencies(tok, text)

println("\n[1] Input Text: \"$text\"")
println(" • Tokens as Physical Wave Frequencies (Hz):")
for (i, f) in enumerate(wave_freqs[1:min(8, length(wave_freqs))])
    println("   - Token $i: $(round(f, digits=3)) Hz")
end
if length(wave_freqs) > 8
    println("   ... ($(length(wave_freqs) - 8) more wave frequencies)")
end

# 3. Decode Directly from Wave Frequencies
decoded_text = decode(tok, wave_freqs)
println("\n[2] Lossless Wave Frequency Decoding:")
println(" • Decoded from frequencies: \"$decoded_text\"")
@assert strip(decoded_text) == strip(text) "Decoded text does not match original!"
println(" ✅ 100% Lossless Wave Frequency Roundtrip Verified!")

# 4. Multilingual & Emoji Test with Wave Frequencies
multilingual_texts = [
    "Quantum Resonance 🌊⚛️⚡",
    "الذكاء الموجي والرنين الصوتي",
    "波浪智能与量子共振",
    "Волновая интерференция и гармоники"
]

println("\n[3] Multilingual Wave Frequency Encoding & Decoding:")
for s in multilingual_texts
    f_seq = tokenize_frequencies(tok, s)
    d_str = decode(tok, f_seq)
    println(" • Original: $s")
    println("   Wave Frequencies: [$(join([string(round(f, digits=1)) * " Hz" for f in f_seq[1:min(4, length(f_seq))]], ", "))...]")
    println("   Decoded:  $d_str")
    @assert strip(d_str) == strip(s) "Mismatch for: $s"
end
println(" ✅ All Multilingual & Emoji Wave Frequency Roundtrips Exact!")

# 5. Custom Tokenizer with Explicit Frequencies
custom_freqs = Dict(
    "solfeggio_ut" => 396.0,
    "solfeggio_re" => 417.0,
    "solfeggio_mi" => 528.0,
    "solfeggio_fa" => 639.0,
    "solfeggio_sol" => 741.0,
    "solfeggio_la" => 852.0
)
tok_custom = custom_tokenizer(custom_freqs)
phrase = "solfeggio_ut solfeggio_mi solfeggio_la"
f_phrase = tokenize_frequencies(tok_custom, phrase)
println("\n[4] Custom Tokenizer with Exact Physical Frequencies:")
println(" • Phrase: \"$phrase\"")
println(" • Emitted Frequencies: $f_phrase")
@assert f_phrase[1] == 396.0
@assert f_phrase[3] == 528.0
@assert f_phrase[5] == 852.0
d_phrase = decode(tok_custom, f_phrase)
println(" • Decoded: \"$d_phrase\"")
@assert strip(d_phrase) == strip(phrase)

# 6. Save & Load Tokenizer with Wave Frequencies
save_path = "/tmp/wave_freq_tokenizer.json"
save_tokenizer(tok_custom, save_path)
println("\n[5] Saved Tokenizer with Wave Frequencies to: $save_path")
tok_reloaded = load_tokenizer(save_path)
f_reloaded = tokenize_frequencies(tok_reloaded, phrase)
@assert f_reloaded ≈ f_phrase
d_reloaded = decode(tok_reloaded, f_reloaded)
@assert strip(d_reloaded) == strip(phrase)
println(" ✅ Reloaded Tokenizer Matches Exact Wave Frequencies: $f_reloaded")

println("\n================================================================================")
println(" 🌊 ALL WAVE FREQUENCY TOKENIZER TESTS PASSED (100% LOSSLESS) 🌊")
println("================================================================================")
