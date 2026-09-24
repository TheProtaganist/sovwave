"""
Test Export Module - Intentionally has issues for validator testing
"""
module TestExportModule

export calculate_sum, multiply_values!, DataPoint, CONSTANT_PI
export nonexistent_function  # ERROR: Exported but not defined
export unused_helper  # Will be unused
export process_data  # New function with multiple typed parameters

# Struct definition
struct DataPoint
    x::Float64
    y::Float64
    label::String
end

# Const definition
const CONSTANT_PI = 3.14159

# Function with parameters
function calculate_sum(a::Float64, b::Float64)::Float64
    return a + b
end

# Function with ! and typed parameters
function multiply_values!(arr::Vector{Float64}, factor::Float64)::Nothing
    for i in 1:length(arr)
        arr[i] *= factor
    end
    return nothing
end

# Function with multiple parameters of different types
function process_data(id::Int64, value::Float64, name::String, active::Bool)::String
    return "ID: $id, Value: $value, Name: $name, Active: $active"
end

# Short form function
square(x) = x * x

# Helper function (exported but will be unused)
function unused_helper(x::Int)::Int
    return x + 1
end

end # module
