# frozen_string_literal: true

require "test_helper"

class TestLazySession < Minitest::Test
  include TestHelper

  def setup
    skip "simple.onnx not found" unless model_exists?("simple.onnx")
  end

  def test_not_loaded_initially
    lazy = OnnxRuby::LazySession.new(simple_model_path)
    refute lazy.loaded?
  end

  def test_loads_on_first_run
    lazy = OnnxRuby::LazySession.new(simple_model_path)
    result = lazy.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })

    assert lazy.loaded?
    assert result.key?("output")
  end

  def test_loads_on_inputs
    lazy = OnnxRuby::LazySession.new(simple_model_path)
    inputs = lazy.inputs

    assert lazy.loaded?
    assert_equal 1, inputs.length
  end

  def test_loads_on_outputs
    lazy = OnnxRuby::LazySession.new(simple_model_path)
    outputs = lazy.outputs

    assert lazy.loaded?
    assert_equal 1, outputs.length
  end

  def test_reuses_session
    lazy = OnnxRuby::LazySession.new(simple_model_path)
    lazy.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
    lazy.run({ "input" => [[5.0, 6.0, 7.0, 8.0]] })

    assert lazy.loaded?
  end

  def test_thread_safe_loading
    lazy = OnnxRuby::LazySession.new(simple_model_path)
    results = []

    threads = 4.times.map do
      Thread.new do
        r = lazy.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
        results << r
      end
    end
    threads.each(&:join)

    assert_equal 4, results.length
    results.each { |r| assert r.key?("output") }
  end
end
