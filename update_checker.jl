#!/usr/bin/env julia
"""
Debug specific failing cases
"""

function check_coherence_debug(output::String, prompt::String)
    # Check if output contains gibberish
    
    # 1. Repeated characters (3+)
    has_repeated_chars = occursin(r"(.)\1{2,}", output)
    println("  has_repeated_chars: $has_repeated_chars")
    
    # 2. Random caps (3+ consecutive capitals, excluding proper nouns at start)
    has_random_caps = occursin(r"[A-Z]{3,}", output)
    println("  has_random_caps: $has_random_caps")
    
    # 3. Numbers mixed with letters (catches both ways, even with punctuation)
    has_numbers_mixed = occursin(r"[A-Za-z][.,!?:;]*\d|\d[.,!?:;]*[A-Za-z]", output)
    println("  has_numbers_mixed: $has_numbers_mixed")
    
    # 4. Obvious gibberish patterns  
    has_ffff_pattern = occursin(r"FFFF|GGGG|HHHH|IIII|JJJJ|KKKK", output)
    println("  has_ffff_pattern: $has_ffff_pattern")
    
    # 4b. Special char attached to words (is* or language()
    has_weird_punctuation = occursin(r"[A-Za-z][*#\$%^&<>]|[*#\$%^&<>][A-Za-z]", output)
    println("  has_weird_punctuation: $has_weird_punctuation")
    
    # 5. Too short (generated less than 3 characters)
    too_short = length(output) <= length(prompt) + 2
    println("  too_short: $too_short (output=$(length(output)), prompt=$(length(prompt)))")
    
    # 6. Check for meaningful words
    words = split(output, r"\s+")
    meaningful_words = ["is", "are", "was", "were", "the", "a", "an", "in", "on", "at", 
                       "to", "for", "with", "and", "or", "but", "equals", "plus", "minus"]
    has_meaningful_word = any(w -> lowercase(w) in meaningful_words, words)
    println("  has_meaningful_word: $has_meaningful_word")
    
    # 7. Check for excessive punctuation or special chars
    special_char_count = count(c -> c in "()[]{}!@#\$%^&*<>?/\\|;:,.", output)
    total_chars = length(output)
    excessive_special_chars = total_chars > 0 && (special_char_count / total_chars) > 0.15
    println("  excessive_special_chars: $excessive_special_chars ($(special_char_count)/$total_chars = $(round(special_char_count/total_chars*100, digits=1))%)")
    
    # 8. Check for word boundaries
    word_count = length(words)
    avg_word_length = word_count > 0 ? total_chars / word_count : 0
    weird_spacing = avg_word_length > 15 || avg_word_length < 2
    println("  weird_spacing: $weird_spacing (avg=$(round(avg_word_length, digits=1)))")
    
    # 9. Remove prompt from output to see what was generated
    generated_part = replace(output, prompt => "", count=1)
    generated_words = filter(w -> length(w) > 0, split(generated_part, r"\s+"))
    println("  generated_words: $(length(generated_words)) words")
    println("  generated_part: \"$generated_part\"")
    
    # 10. Check for word salad
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
    has_word_salad = max_consecutive_function >= 2  # Even 2 consecutive function words is suspicious
    println("  has_word_salad: $has_word_salad (max_consecutive=$max_consecutive_function)")
    
    # 10b. Check for nonsensical word order - too many function words relative to content
    function_word_count = count(w -> lowercase(strip(w, [',', '.', '!', '?', ':', ';', '(', ')', '[', ']'])) in function_words, generated_words)
    function_word_ratio = length(generated_words) > 0 ? function_word_count / length(generated_words) : 0
    too_many_function_words = function_word_ratio > 0.35  # More than 35% function words is suspicious
    println("  too_many_function_words: $too_many_function_words (count=$function_word_count, ratio=$(round(function_word_ratio*100, digits=1))%)")
    
    # 11. Check for semantic coherence
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
    has_enough_real_words = real_word_count >= 4
    println("  has_enough_real_words: $has_enough_real_words (count=$real_word_count)")
    
    # Output is coherent if it passes ALL checks
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
    
    println("  => is_coherent: $is_coherent")
    return is_coherent
end

# Failing gibberish cases (detected as coherent)
println("="^80)
println("FAILING CASE 1: Should be GIBBERISH")
println("-"^80)
println("Prompt: \"Hello\"")
println("Output: \"Hello in is  forward you? are moves\"")
check_coherence_debug("Hello in is  forward you? are moves", "Hello")

println("\n" * "="^80)
println("FAILING CASE 2: Should be GIBBERISH")
println("-"^80)
println("Prompt: \"The sky\"")
println("Output: \"The sky the language ( is* science loyal.4 well fast\"")
check_coherence_debug("The sky the language ( is* science loyal.4 well fast", "The sky")

println("\n" * "="^80)
println("FAILING CASE 3: Should be COHERENT")
println("-"^80)
println("Prompt: \"Hello\"")
println("Output: \"Hello, how are you doing today? I hope well\"")
check_coherence_debug("Hello, how are you doing today? I hope well", "Hello")
