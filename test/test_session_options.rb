# frozen_string_literal: true

require "test_helper"

class TestSessionOptions < Minitest::Test
  include TestHelper

  def setup
    skip "simple.onnx not found" unless model_exists?("simple.onnx")
  end

  def test_optimization_level_all
    session = OnnxRuby::Session.new(simple_model_path, optimization_level: :all)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_optimization_level_none
    session = OnnxRuby::Session.new(simple_model_path, optimization_level: :none)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_optimization_level_basic
    session = OnnxRuby::Session.new(simple_model_path, optimization_level: :basic)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_optimization_level_extended
    session = OnnxRuby::Session.new(simple_model_path, optimization_level: :extended)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_threading_options
    session = OnnxRuby::Session.new(simple_model_path,
                                    intra_threads: 2,
                                    inter_threads: 2)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_memory_pattern_disabled
    session = OnnxRuby::Session.new(simple_model_path, memory_pattern: false)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_cpu_mem_arena_disabled
    session = OnnxRuby::Session.new(simple_model_path, cpu_mem_arena: false)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_parallel_execution_mode
    session = OnnxRuby::Session.new(simple_model_path,
                                    execution_mode: :parallel,
                                    inter_threads: 2)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_sequential_execution_mode
    session = OnnxRuby::Session.new(simple_model_path, execution_mode: :sequential)
    result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert result.key?("output")
  end

  def test_log_levels
    %i[verbose info warning error fatal].each do |level|
      session = OnnxRuby::Session.new(simple_model_path, log_level: level)
      result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
      assert result.key?("output"), "Failed with log_level: #{level}"
    end
  end

  def test_dynamic_batch_size
    session = OnnxRuby::Session.new(simple_model_path)

    # Single item batch
    r1 = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    assert_equal 1, r1["output"].length

    # Batch of 3
    r3 = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0],
                                    [5.0, 6.0, 7.0, 8.0],
                                    [9.0, 10.0, 11.0, 12.0]] })
    assert_equal 3, r3["output"].length
  end
end
