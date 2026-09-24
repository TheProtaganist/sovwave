"""
Test Import File - Intentionally has issues for validator testing
"""

# Correct import
using .TestExportModule: calculate_sum, DataPoint, CONSTANT_PI, process_data

# WARNING: Imported but never used
using .TestExportModule: unused_helper

function main()
    # ERROR: Wrong number of parameters (expects 2, got 1)
    result = calculate_sum(10.0)
    println("Sum: $result")
    
    # ERROR: Wrong parameter types (expects Float64, got String)
    bad_result = calculate_sum("hello", "world")
    println("Bad sum: $bad_result")
    
    # ERROR: Wrong number of parameters (expects 2, got 3)
    over_result = calculate_sum(1.0, 2.0, 3.0)
    println("Over: $over_result")
    
    # ERROR: Wrong struct field types
    point = DataPoint("not a float", 2.0, "origin")  # x should be Float64, not String
    println("Point: $(point.x), $(point.y)")
    
    # ERROR: Missing struct field
    bad_point = DataPoint(1.0, 2.0)  # Missing label parameter
    
    # ERROR: Wrong order of struct fields (String, Float64, Float64 instead of Float64, Float64, String)
    wrong_order = DataPoint("label", 1.0, 2.0)
    
    # ERROR: process_data wrong types (expects Int64, Float64, String, Bool but got String, Int64, Float64, String)
    bad_process = process_data("123", 456, 78.9, "yes")
    
    # ERROR: process_data wrong number of args (expects 4, got 2)
    incomplete = process_data(1, 2.5)
    
    # Correct usage
    correct = calculate_sum(5.0, 10.0)
    println("Correct: $correct")
    
    # Correct process_data usage
    good_process = process_data(42, 3.14, "test", true)
    println(good_process)
    
    # Correct const usage
    area = CONSTANT_PI * 5.0^2
    println("Area: $area")
    
    # ERROR: Using multiply_values! but never imported it
    arr = [1.0, 2.0, 3.0]
    multiply_values!(arr, 2.0)
    println("Multiplied: $arr")
    
    # ERROR: Wrong parameter type for multiply_values! (expects Vector{Float64}, got String array literal)
    string_arr = ["a", "b", "c"]
    multiply_values!(string_arr, 2.0)
    
    # ERROR: Using square() but never imported it (and not exported)
    val = square(5)
    println("Square: $val")
    
    # ERROR: Using nonexistent_function that's exported but not defined
    broken = nonexistent_function(100)
end

main()
