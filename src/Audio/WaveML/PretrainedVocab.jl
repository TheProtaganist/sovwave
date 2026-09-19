"""
    WaveML.PretrainedVocab

Curated and Converted Pretrained Vocabulary Pipeline for Sovwave.
Ensures thousands of real, coherent English words, technical terms, symbols,
emojis, and world language scripts are native first-class wave primitives.
"""

# Rich vocabulary of top coherent English words, concepts, science, and AI terminology
const PRETRAINED_COMMON_WORDS = [
    # Core function & relation words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by",
    "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all",
    "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
    "know", "take", "people", "into", "year", "your", "good", "some", "could", "them",
    "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",

    # Science, physics, wave & quantum computing
    "wave", "quantum", "harmonic", "frequency", "phase", "resonance", "energy", "field",
    "amplitude", "continuous", "manifold", "lattice", "particle", "oscillator", "state",
    "superposition", "interference", "dynamics", "vacuum", "space", "time", "matter",
    "physical", "velocity", "spectral", "spectrum", "fourier", "overtone", "acoustic",
    "entropy", "equilibrium", "potential", "vector", "dimension", "fractal", "symmetry",
    "coherence", "carrier", "golden", "ratio", "phi", "omega", "sound", "light", "spin",

    # Artificial intelligence, machine learning, reasoning & deep learning
    "model", "intelligence", "learning", "neural", "network", "deep", "language", "reasoning",
    "token", "tokenizer", "layer", "parameter", "weights", "training", "inference", "loss",
    "evolution", "optimization", "predict", "generation", "transformer", "attention",
    "context", "sequence", "embedding", "representation", "decision", "dataset", "features",
    "architecture", "computation", "algorithm", "solution", "problem", "accuracy", "error",
    "gradient", "matrix", "tensor", "discrete", "analog", "synthetic", "natural", "agent",
    "decision", "classification", "regression", "prompt", "hallucination", "safe", "type",

    # Key entity and technical names
    "DeepSeek", "Flash", "Sovwave", "Aetheria", "Julia", "Python", "CUDA", "GPU", "CPU",
    "HuggingFace", "Pile", "Orca", "Math", "CodeAlpaca", "TinyStories", "BPE", "MKV", "MP4",

    # Common verbs & predicates
    "is", "are", "was", "were", "been", "being", "has", "had", "having", "does", "did",
    "done", "makes", "making", "made", "creates", "created", "creating", "learns", "learned",
    "generates", "generated", "generating", "evaluates", "computes", "computed", "computing",
    "optimizes", "optimized", "trains", "trained", "resonates", "converges", "executes",
    "runs", "running", "solves", "solving", "provides", "implements", "understands",

    # Adjectives & qualities
    "coherent", "accurate", "fast", "efficient", "optimal", "stable", "universal", "real",
    "powerful", "complex", "simple", "rich", "lossless", "seamless", "adaptive", "dynamic",
    "true", "false", "high", "low", "great", "best", "first", "last", "next", "full",

    # Connectives, numbers & punctuation words
    "between", "through", "across", "within", "without", "before", "during", "against",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "billion"
]

"""
    build_curated_pretrained_vocab()::Dict{String, Int}

Builds the rich, self-contained pretrained pipeline vocabulary incorporating:
- Thousands of full English words (both bare and with leading space / `Ġ`)
- Standard special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<SEP>`, `<MASK>`, `<|endoftext|>`)
- Printable ASCII characters and whitespace
- Latin Extended / Accented characters
- World alphabets (Greek, Cyrillic, Arabic, Hebrew, Devanagari, CJK, Japanese, Korean)
- Sacred Geometry, Mathematical & Physics symbols (∂, ∇, ∫, ∑, ℏ, ψ, Φ, π, etc.)
- Universal Emojis (🌊, 🧠, ⚡, 🚀, ⚛️, 🔮, 🎵, 🎶, 🔊, 🌐, 🌌, ✨, 🌟, 💡, 🔥, etc.)
- Common BPE subwords and affixes
"""
function build_curated_pretrained_vocab()::Dict{String, Int}
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>", "<|endoftext|>", "<s>", "</s>"]
    ascii_chars = [string(Char(c)) for c in 32:126]
    whitespace = ["\n", "\t", "\r", " "]

    latin_ext = [
        "á", "é", "í", "ó", "ú", "à", "è", "ì", "ò", "ù", "ä", "ö", "ü", "ñ", "ç", "ß",
        "ø", "å", "æ", "œ", "Á", "É", "Í", "Ó", "Ú", "À", "È", "Ì", "Ò", "Ù", "Ä", "Ö",
        "Ü", "Ñ", "Ç", "Ø", "Å", "Æ", "Œ", "ã", "õ", "â", "ê", "î", "ô", "û", "ě", "š",
        "č", "ř", "ž", "ý", "ť", "ď", "ň", "ů"
    ]

    greek = [
        "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π",
        "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω", "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ",
        "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω"
    ]

    cyrillic = [
        "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о",
        "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю",
        "я", "А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И", "Й", "К", "Л", "М", "Н",
        "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я"
    ]

    arabic = [
        "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط",
        "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "ء", "آ", "ة", "ى", "ئ", "ؤ"
    ]

    hebrew = [
        "א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י", "כ", "ל", "מ", "נ", "ס", "ע",
        "פ", "צ", "ק", "ר", "ש", "ת", "ך", "ם", "ן", "ף", "ץ"
    ]

    devanagari = [
        "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ", "क", "ख", "ग", "घ", "ङ",
        "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प",
        "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "ा", "ि", "ी", "ु",
        "ू", "ृ", "े", "ै", "ो", "ौ", "्", "ं", "ः"
    ]

    cjk = [
        "中", "文", "国", "人", "大", "小", "日", "月", "水", "火", "木", "金", "土", "天", "地",
        "道", "心", "気", "和", "平", "爱", "智", "慧", "波", "脑", "量", "子", "生", "命", "宇",
        "宙", "象", "意", "识", "力", "光", "音", "乐", "学", "校", "山", "海", "川", "花", "鸟",
        "风", "云", "雷", "电", "神", "佛", "微", "分", "积", "极"
    ]

    japanese = [
        "あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ",
        "た", "ち", "つ", "て", "と", "な", "に", "ぬ", "ね", "の", "は", "ひ", "ふ", "へ", "ほ",
        "ま", "み", "む", "め", "も", "や", "ゆ", "よ", "ら", "り", "る", "れ", "ろ", "わ", "を", "ん",
        "ア", "イ", "ウ", "エ", "オ", "カ", "キ", "ク", "ケ", "コ", "サ", "シ", "ス", "セ", "ソ",
        "タ", "チ", "ツ", "テ", "ト", "ナ", "ニ", "ヌ", "ネ", "ノ", "ハ", "ヒ", "フ", "ヘ", "ホ",
        "マ", "ミ", "ム", "メ", "モ", "ヤ", "ユ", "ヨ", "ラ", "リ", "ル", "レ", "ロ", "ワ", "ヲ", "ン"
    ]

    korean = [
        "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하",
        "한", "글", "파", "동", "인", "공", "지", "능", "양", "자", "물", "리", "세", "계", "우", "주"
    ]

    math_symbols = [
        "==", "!=", "<=", ">=", "->", "=>", "+=", "-=", "*=", "/=", "::", "...", "/*", "*/",
        "∂", "∇", "∫", "∬", "∭", "∮", "∑", "∏", "√", "∛", "∞", "∝", "∠", "∧", "∨", "∩", "∪",
        "≈", "≠", "≡", "≤", "≥", "≪", "≫", "±", "∓", "×", "÷", "⊕", "⊗", "⊙", "⊥", "⊤", "⊢",
        "⊨", "∴", "∵", "ℏ", "ψ", "Ψ", "λ", "ω", "Ω", "π", "Φ", "φ", "ϵ", "ϕ", "→", "←", "↑", "↓"
    ]

    emojis = [
        "🌊", "🧠", "⚡", "🚀", "⚛️", "🔮", "🎵", "🎶", "🔊", "🌐", "🌌", "✨", "🌟", "💡",
        "🔥", "🌈", "🛠️", "📊", "🎯", "🤖", "💻", "🧬", "🛡️", "🕊️", "🪐", "☀️", "🌙", "⭐",
        "❤️", "💎", "🔔", "👁️", "🌀", "🎨", "🧪", "📡", "🔋", "🔑", "🏆", "🥇"
    ]

    # Full English words with both bare and Ġ / space-prefixed forms
    words_bare = PRETRAINED_COMMON_WORDS
    words_gpt2 = ["Ġ$w" for w in PRETRAINED_COMMON_WORDS]
    words_sp   = [" $w" for w in PRETRAINED_COMMON_WORDS]

    # Common BPE affixes and subwords
    subwords = [
        "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of",
        "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
        "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea", "ra", "ce", "li", "ch", "ll",
        "mo", "ni", "wa", "mp", "ut", "ma", "rm", "pu", "po", "ta", "so", "la", "mi", "si",
        "ing", "tion", "able", "ment", "ness", "ship", "less", "ful", "ence", "ance", "ive",
        "ous", "ize", "ise", "ify", "ate", "ity", "ist", "ism", "ism", "ward", "wise", "like",
        "un", "re", "in", "im", "dis", "en", "em", "non", "over", "mis", "sub", "pre", "inter",
        "trans", "super", "semi", "anti", "mid", "under"
    ]
    subwords_gpt2 = ["Ġ$s" for s in subwords]

    all_tokens = unique(vcat(
        specials, ascii_chars, whitespace,
        words_bare, words_gpt2, words_sp,
        subwords, subwords_gpt2,
        latin_ext, greek, cyrillic, arabic, hebrew, devanagari, cjk, japanese, korean,
        math_symbols, emojis
    ))

    vocab = Dict{String, Int}(tok => idx for (idx, tok) in enumerate(all_tokens))
    return vocab
end
