#!/usr/bin/env julia

using Test
using JuMP
using Alpine
using HiGHS
using Ipopt
using Juniper
using MathOptInterface

const MOI = MathOptInterface

function json_escape(s::AbstractString)
    escaped = replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
    return escaped
end

function write_record(io, record::Dict{String, Any})
    parts = String[]
    for key in sort!(collect(keys(record)))
        value = record[key]
        encoded = if value === nothing
            "null"
        elseif value isa Bool
            value ? "true" : "false"
        elseif value isa Integer || value isa AbstractFloat
            string(value)
        else
            "\"" * json_escape(string(value)) * "\""
        end
        push!(parts, "\"" * key * "\":" * encoded)
    end
    write(io, "{" * join(parts, ",") * "}\n")
end

function build_optimizer()
    ipopt = MOI.OptimizerWithAttributes(
        Ipopt.Optimizer,
        MOI.Silent() => true,
        "sb" => "yes",
        "max_iter" => 9999,
    )
    highs = MOI.OptimizerWithAttributes(
        HiGHS.Optimizer,
        "presolve" => "on",
        "log_to_console" => false,
    )
    juniper = MOI.OptimizerWithAttributes(
        Juniper.Optimizer,
        MOI.Silent() => true,
        "mip_solver" => highs,
        "nl_solver" => ipopt,
    )
    return JuMP.optimizer_with_attributes(
        Alpine.Optimizer,
        "nlp_solver" => ipopt,
        "mip_solver" => highs,
        "minlp_solver" => juniper,
    )
end

function run_case(optimizer, problem_id::AbstractString, symbol_name::AbstractString, minlptests)
    problem_id_str = String(problem_id)
    symbol_name_str = String(symbol_name)
    f = getfield(minlptests, Symbol(symbol_name_str))
    ts = Test.DefaultTestSet(problem_id_str)
    Test.push_testset(ts)
    err_text = nothing
    wall_time = @elapsed begin
        try
            Base.invokelatest(
                f,
                optimizer,
                minlptests.OPT_TOL,
                minlptests.PRIMAL_TOL,
                minlptests.DUAL_TOL,
                minlptests.TERMINATION_TARGET_GLOBAL,
                minlptests.PRIMAL_TARGET_GLOBAL,
            )
        catch err
            err_text = sprint(showerror, err, catch_backtrace())
        end
    end
    Test.pop_testset()

    counts = Test.get_test_counts(ts)
    outcome = (err_text === nothing && counts.fails == 0 && counts.errors == 0 && counts.broken == 0) ? "pass" : "fail"
    note = err_text === nothing ? "" : split(err_text, '\n')[1]
    return Dict(
        "problem_id" => problem_id_str,
        "symbol" => symbol_name_str,
        "outcome" => outcome,
        "passes" => counts.passes,
        "fails" => counts.fails,
        "errors" => counts.errors,
        "broken" => counts.broken,
        "wall_time_sec" => wall_time,
        "note" => note,
    )
end

function main()
    if length(ARGS) != 3
        error("usage: alpine_minlptests_status.jl REQUEST.tsv OUTPUT.jsonl MINLPTESTS_PATH")
    end

    request_path = ARGS[1]
    output_path = ARGS[2]
    minlptests_path = ARGS[3]

    push!(LOAD_PATH, minlptests_path)
    Base.eval(Main, :(using MINLPTests))
    minlptests = Base.invokelatest(() -> getfield(Main, :MINLPTests))

    optimizer = build_optimizer()

    open(output_path, "w") do io
        for line in eachline(request_path)
            isempty(strip(line)) && continue
            fields = split(line, '\t')
            if length(fields) != 3
                error("expected 3 tab-separated fields per line in request file")
            end
            problem_id, category, symbol_name = fields
            record = run_case(optimizer, problem_id, symbol_name, minlptests)
            record["category"] = category
            write_record(io, record)
        end
    end
end

main()
