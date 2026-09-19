# evolve_harness.jl - shared multi-metric scorer for Alpha-Evolve tournaments.
# Reads specs/settings.json (LIVING config, see specs/Guide.md sec4). Zero deps:
# ships its own minimal JSON-subset parser (stdlib only).
#
# Usage (from l-2 / l-1 tasks in specs/Tasks.md):
#   cfg  = EvolveHarness.load_settings()
#   ws, higher = EvolveHarness.metric_weights(cfg, "ALG-ENC")
#   composite = EvolveHarness.composite_score(scores, ws, higher)
#   EvolveHarness.write_round_log("ALG-ENC", 1, "A1", rows)
#   EvolveHarness.print_scoreboard(rows_namedtuples)
module EvolveHarness

using Dates

export load_settings, metric_weights, normalize_metric, composite_score,
       write_round_log, print_scoreboard

# --- minimal recursive JSON-subset parser ----------------------------------
function _json_value(txt::AbstractString, i::Ref{Int})
    while i[] <= lastindex(txt) && txt[i[]] in (' ', '\t', '\n', '\r')
        i[] = nextind(txt, i[])
    end
    i[] <= lastindex(txt) || error("JSON: unexpected end of input")
    c = txt[i[]]
    if c == '{'
        obj = Dict{String,Any}()
        i[] = nextind(txt, i[])
        while true
            while i[] <= lastindex(txt) && txt[i[]] in (' ', '\t', '\n', '\r'); i[] = nextind(txt, i[]); end
            if i[] > lastindex(txt); error("JSON: unterminated object"); end
            if txt[i[]] == '}'; i[] = nextind(txt, i[]); return obj; end
            txt[i[]] == '"' || error("JSON: expected string key, got $(txt[i[]])")
            key = _json_string(txt, i)
            while i[] <= lastindex(txt) && txt[i[]] in (' ', '\t', '\n', '\r'); i[] = nextind(txt, i[]); end
            (i[] <= lastindex(txt) && txt[i[]] == ':') || error("JSON: expected ':'")
            i[] = nextind(txt, i[])
            obj[key] = _json_value(txt, i)
            while i[] <= lastindex(txt) && txt[i[]] in (' ', '\t', '\n', '\r'); i[] = nextind(txt, i[]); end
            i[] <= lastindex(txt) || error("JSON: unterminated object")
            sep = txt[i[]]
            i[] = nextind(txt, i[])
            sep == ',' && continue
            sep == '}' && return obj
            error("JSON: expected ',' or '}' got '$sep'")
        end
    elseif c == '['
        arr = Any[]
        i[] = nextind(txt, i[])
        while true
            while i[] <= lastindex(txt) && txt[i[]] in (' ', '\t', '\n', '\r'); i[] = nextind(txt, i[]); end
            if i[] > lastindex(txt); error("JSON: unterminated array"); end
            if txt[i[]] == ']'; i[] = nextind(txt, i[]); return arr; end
            push!(arr, _json_value(txt, i))
            while i[] <= lastindex(txt) && txt[i[]] in (' ', '\t', '\n', '\r'); i[] = nextind(txt, i[]); end
            i[] <= lastindex(txt) || error("JSON: unterminated array")
            sep = txt[i[]]
            i[] = nextind(txt, i[])
            sep == ',' && continue
            sep == ']' && return arr
            error("JSON: expected ',' or ']' got '$sep'")
        end
    elseif c == '"'
        return _json_string(txt, i)
    else
        start = i[]
        while i[] <= lastindex(txt) && !(txt[i[]] in (' ', '\t', '\n', '\r', ',', ']', '}'))
            i[] = nextind(txt, i[])
        end
        word = txt[start:prevind(txt, i[])]
        word == "true" && return true
        word == "false" && return false
        word == "null" && return nothing
        return tryparse(Float64, word)
    end
end

function _json_string(txt::AbstractString, i::Ref{Int})
    txt[i[]] == '"' || error("JSON: expected opening quote")
    i[] = nextind(txt, i[])
    buf = IOBuffer()
    while i[] <= lastindex(txt)
        c = txt[i[]]
        if c == '"'
            i[] = nextind(txt, i[])
            return String(take!(buf))
        elseif c == '\\'
            i[] = nextind(txt, i[])
            esc = txt[i[]]
            esc == 'n' && write(buf, '\n')
            esc == 't' && write(buf, '\t')
            esc == '"' && write(buf, '"')
            esc == '\\' && write(buf, '\\')
            esc == '/' && write(buf, '/')
            esc == 'u' && error("JSON: \\u escapes not supported by harness parser")
            i[] = nextind(txt, i[])
        else
            write(buf, c)
            i[] = nextind(txt, i[])
        end
    end
    error("JSON: unterminated string")
end

"""
    load_settings(path = default) -> Dict{String,Any}

Parse `specs/settings.json` (paths resolved relative to this file). Returns a
nested `Dict{String,Any}` mirroring the JSON exactly.
"""
function load_settings(path::AbstractString = joinpath(@__DIR__, "..", "..", "specs", "settings.json"))
    p = abspath(path)
    isfile(p) || error("settings.json not found: $p")
    return _json_value(read(p, String), Ref(1))
end

"""
    metric_weights(cfg, alg="") -> (weights, higher)

Return `(metric->weight, metric->higher_is_better)` merged with
`cfg["per_alg_overrides"][alg]["metrics"]` when present. Weights renormalized
to sum 1.0 (per-ALG, honoring the living-config rule).
"""
function metric_weights(cfg::Dict{String,Any}, alg::AbstractString = "")
    ws = Dict{String,Float64}()
    higher = Dict{String,Bool}()
    for (k, v) in cfg["metrics"]
        kk = String(k)
        startswith(kk, "_") && continue   # skip _comment etc. metadata keys
        ws[kk] = Float64(v["weight"])
        higher[kk] = Bool(v["higher_is_better"])
    end
    ov = get(cfg, "per_alg_overrides", Dict{String,Any}())
    if isa(ov, Dict) && haskey(ov, alg)
        mo = ov[alg]
        if haskey(mo, "metrics")
            for (k, v) in mo["metrics"]
                kk = String(k)
                startswith(kk, "_") && continue
                haskey(v, "weight") && (ws[kk] = Float64(v["weight"]))
                haskey(v, "higher_is_better") && (higher[kk] = Bool(v["higher_is_better"]))
            end
        end
    end
    s = sum(values(ws))
    s > 0 || error("metric_weights: weights sum to zero")
    for k in keys(ws); ws[k] /= s; end
    return ws, higher
end

"""Map a raw metric value to a 0..1 score honoring higher_is_better."""
normalize_metric(raw::Real, higher_is_better::Bool) =
    clamp(higher_is_better ? Float64(raw) : 1.0 - Float64(raw), 0.0, 1.0)

"""
    composite_score(scores, weights, higher) -> Float64

Weighted composite: `sum(w[i] * normalize(raw[i]))` over metrics present in
`scores`. Raw lower-is-better metrics (speed, complexity, energy) are inverted
(`1 - raw`) before weighting.
"""
function composite_score(scores::Dict{<:AbstractString,<:Real},
                         weights::Dict{<:AbstractString,<:Real},
                         higher::Dict{<:AbstractString,Bool})
    total = 0.0
    for (k, raw) in scores
        kk = String(k)
        w = get(weights, kk, 0.0)
        total += w * normalize_metric(raw, get(higher, kk, true))
    end
    return total
end

"""
    write_round_log(alg, round, variant, lines; logdir) -> path

Append a timestamped block to `logs/theory/<ALG>/round<R>-<variant>.log`.
Creates directories as needed.
"""
function write_round_log(alg::AbstractString, round::Integer, variant::AbstractString,
                         lines::Vector{<:AbstractString};
                         logdir::AbstractString = joinpath(@__DIR__, "..", "..", "logs", "theory"))
    d = joinpath(logdir, alg)
    mkpath(d)
    path = joinpath(d, "round$(round)-$(variant).log")
    open(path, "a") do io
        println(io, "=== run $(now()) | round $round variant $variant ===")
        for l in lines
            println(io, l)
        end
        println(io)
    end
    return path
end

"""Print a markdown scoreboard table (headers from the first row keys)."""
function print_scoreboard(rows::Vector{<:NamedTuple})
    isempty(rows) && return
    cols = collect(keys(rows[1]))
    println("| " * join(string.(cols), " | ") * " |")
    println("|" * join(("---" for _ in cols), "|") * "|")
    for r in rows
        println("| " * join((string(getfield(r, c)) for c in cols), " | ") * " |")
    end
end

end # module EvolveHarness
