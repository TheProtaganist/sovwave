"""
    WaveML.TextValidation

Coherence validation and quality checking for generated text outputs.
Detects gibberish, word salad, and malformed outputs from wave-based language models.
"""

export check_text_coherence, CoherenceResult

"""
    CoherenceResult

Struct containing coherence validation results with detailed diagnostics.
"""
struct CoherenceResult
    is_coherent::Bool
    score::Float64
    issues::Vector{String}
    metrics::Dict{Symbol, Any}
end

function Base.show(io::IO, r::CoherenceResult)
    status = r.is_coherent ? "COHERENT" : "INCOHERENT"
    @printf(io, "CoherenceResult(%s, score=%.3f, issues=%d)", status, r.score, length(r.issues))
end

"""
    check_text_coherence(output::String, prompt::String=""; strict::Bool=false)::CoherenceResult

Validates text coherence using multiple linguistic and statistical checks.
Returns a `CoherenceResult` with detailed diagnostics.

Parameters:
- `output`: The generated text to validate
- `prompt`: Optional prompt that was used to generate the text
- `strict`: If true, applies stricter thresholds for coherence detection

The checker detects:
1. Repeated characters (3+ consecutive)
2. Random capitalization patterns
3. Numbers mixed inappropriately with letters
4. Weird punctuation attached to words
5. Excessive special characters
6. Word salad (too many function words)
7. Insufficient real vocabulary words
8. Too short outputs
"""
function check_text_coherence(output::String, prompt::String=""; strict::Bool=false)::CoherenceResult
    issues = String[]
    metrics = Dict{Symbol, Any}()
    
    # 1. Repeated characters (3+)
    has_repeated_chars = occursin(r"(.)\1{2,}", output)
    if has_repeated_chars
        push!(issues, "repeated_characters")
    end
    metrics[:has_repeated_chars] = has_repeated_chars
    
    # 2. Random caps (3+ consecutive capitals)
    has_random_caps = occursin(r"[A-Z]{3,}", output)
    if has_random_caps
        push!(issues, "random_capitalization")
    end
    metrics[:has_random_caps] = has_random_caps
    
    # 3. Numbers mixed with letters (even with punctuation between)
    has_numbers_mixed = occursin(r"[A-Za-z][.,!?:;]*\d|\d[.,!?:;]*[A-Za-z]", output)
    if has_numbers_mixed
        push!(issues, "numbers_mixed_with_letters")
    end
    metrics[:has_numbers_mixed] = has_numbers_mixed
    
    # 4. Obvious gibberish patterns
    has_ffff_pattern = occursin(r"FFFF|GGGG|HHHH|IIII|JJJJ|KKKK", output)
    if has_ffff_pattern
        push!(issues, "gibberish_pattern")
    end
    metrics[:has_gibberish_pattern] = has_ffff_pattern
    
    # 5. Weird punctuation attached to words (is* or language()
    has_weird_punctuation = occursin(r"[A-Za-z][*#\$%^&<>]|[*#\$%^&<>][A-Za-z]", output)
    if has_weird_punctuation
        push!(issues, "weird_punctuation")
    end
    metrics[:has_weird_punctuation] = has_weird_punctuation
    
    # 6. Too short
    too_short = !isempty(prompt) && length(output) <= length(prompt) + 2
    if too_short
        push!(issues, "output_too_short")
    end
    metrics[:too_short] = too_short
    
    # 7. Check for meaningful words
    words = split(output, r"\s+")
    meaningful_words = Set(["is", "are", "was", "were", "the", "a", "an", "in", "on", "at", 
                           "to", "for", "with", "and", "or", "but", "equals", "plus", "minus"])
    has_meaningful_word = any(w -> lowercase(w) in meaningful_words, words)
    if !has_meaningful_word
        push!(issues, "no_meaningful_words")
    end
    metrics[:has_meaningful_words] = has_meaningful_word
    
    # 8. Excessive punctuation
    special_char_count = count(c -> c in "()[]{}!@#\$%^&*<>?/\\|;:,.", output)
    total_chars = length(output)
    excessive_threshold = strict ? 0.12 : 0.15
    excessive_special_chars = total_chars > 0 && (special_char_count / total_chars) > excessive_threshold
    if excessive_special_chars
        push!(issues, "excessive_punctuation")
    end
    metrics[:special_char_ratio] = total_chars > 0 ? special_char_count / total_chars : 0.0
    
    # 9. Weird spacing
    word_count = length(words)
    avg_word_length = word_count > 0 ? total_chars / word_count : 0
    weird_spacing = avg_word_length > 15 || avg_word_length < 2
    if weird_spacing
        push!(issues, "weird_spacing")
    end
    metrics[:avg_word_length] = avg_word_length
    
    # 10. Extract generated part (if prompt provided)
    generated_part = isempty(prompt) ? output : replace(output, prompt => "", count=1)
    generated_words = filter(w -> length(w) > 0, split(generated_part, r"\s+"))
    metrics[:generated_word_count] = length(generated_words)
    
    # 11. Word salad detection - too many consecutive function words
    function_words = Set(["is", "are", "was", "were", "the", "a", "an", "in", "on", "at", "to", "for", "with"])
    consecutive_function = 0
    max_consecutive_function = 0
    for word in generated_words
        clean_word = lowercase(strip(word, [',', '.', '!', '?', ':', ';', '(', ')', '[', ']']))
        if clean_word in function_words
            consecutive_function += 1
            max_consecutive_function = max(max_consecutive_function, consecutive_function)
        else
            consecutive_function = 0
        end
    end
    salad_threshold = strict ? 2 : 3
    has_word_salad = max_consecutive_function >= salad_threshold
    if has_word_salad
        push!(issues, "word_salad")
    end
    metrics[:max_consecutive_function_words] = max_consecutive_function
    
    # 12. Too many function words overall
    function_word_count = count(w -> lowercase(strip(w, [',', '.', '!', '?', ':', ';', '(', ')', '[', ']'])) in function_words, generated_words)
    function_word_ratio = length(generated_words) > 0 ? function_word_count / length(generated_words) : 0.0
    ratio_threshold = strict ? 0.30 : 0.45  # Allow up to 45% function words in normal speech
    too_many_function_words = function_word_ratio > ratio_threshold
    if too_many_function_words
        push!(issues, "too_many_function_words")
    end
    metrics[:function_word_ratio] = function_word_ratio
    
    # 13. Real word count - check for actual vocabulary
    common_words = Set([
        "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "a", "an", "and",
        "or", "but", "if", "then", "than", "when", "where", "who",
        "what", "which", "how", "why", "this", "that", "these", "those",
        "in", "on", "at", "to", "for", "from", "with", "about", "by",
        "of", "as", "into", "through", "during", "before", "after",
        "above", "below", "up", "down", "out", "over", "under",
        "again", "further", "once", "here", "there", "all", "both",
        "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "one", "two", "three", "four", "five",
        "first", "second", "last", "long", "great", "little", "good",
        "new", "old", "right", "big", "high", "small", "large",
        "next", "early", "young", "important", "public", "bad",
        "same", "able", "blue", "red", "green", "black", "white",
        "fast", "slow", "hot", "cold", "warm", "cool", "light", "dark",
        "sky", "sun", "moon", "star", "earth", "water", "fire", "air",
        "tree", "flower", "grass", "mountain", "river", "ocean", "lake",
        "dog", "cat", "bird", "fish", "animal", "animals", "people",
        "person", "man", "woman", "child", "boy", "girl", "friend",
        "time", "day", "night", "year", "week", "month", "hour",
        "morning", "evening", "today", "tomorrow", "yesterday",
        "hello", "hi", "goodbye", "thanks", "please", "yes", "no",
        "plus", "equals", "minus", "times", "divided", "number",
        "science", "math", "mathematics", "language", "music", "art",
        "book", "read", "write", "learn", "teach", "study", "school",
        "work", "play", "run", "walk", "see", "look", "hear", "listen",
        "say", "tell", "ask", "answer", "know", "think", "feel",
        "want", "need", "like", "love", "hate", "happy", "sad",
        "good", "bad", "better", "best", "worse", "worst",
        "go", "come", "bring", "take", "give", "get", "make",
        "find", "use", "show", "help", "move", "turn", "start",
        "stop", "open", "close", "rise", "rises", "fall", "falls",
        "flow", "flows", "travel", "travels", "bring", "brings",
        "together", "forward", "backward", "upward", "downward",
        "east", "west", "north", "south", "loyal", "downhill",
        "hope", "today", "doing", "well"
    ])
    
    real_word_count = count(w -> lowercase(strip(w, [',', '.', '!', '?', ':', ';'])) in common_words, generated_words)
    min_real_words = strict ? 4 : 3
    has_enough_real_words = real_word_count >= min_real_words
    if !has_enough_real_words
        push!(issues, "insufficient_real_words")
    end
    metrics[:real_word_count] = real_word_count
    
    # Calculate overall coherence score (0.0 to 1.0)
    # Each issue reduces the score
    max_issues = 10  # Maximum number of critical issues to check
    score = 1.0 - (length(issues) / max_issues)
    score = clamp(score, 0.0, 1.0)
    
    # Output is coherent if it passes ALL critical checks
    is_coherent = !has_repeated_chars && 
                  !has_random_caps && 
                  !has_numbers_mixed && 
                  !has_ffff_pattern && 
                  !has_weird_punctuation &&
                  !too_short &&
                  has_meaningful_word &&
                  !excessive_special_chars &&
                  !weird_spacing &&
                  !has_word_salad &&
                  !too_many_function_words &&
                  has_enough_real_words
    
    return CoherenceResult(is_coherent, score, issues, metrics)
end

# Note: check_text_coherence(output::String) works via default parameter prompt=""
# No separate definition needed to avoid method overwriting during precompilation
